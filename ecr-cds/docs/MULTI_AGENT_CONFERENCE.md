# Multi-agent case conference (rev 2.5)

`ecr_cds.agents` turns the single-pass pipeline into a conference of specialised agents that
reason over the visual information extracted from the CBCT volume and produce a **suggested
diagnosis and treatment plan** with a complete evidence trail. Nothing in this layer is
clinically validated; the output is a structured suggestion for the treating clinician.

## Architecture

```
DICOM/ZIP ──► ImagingQAAgent ──► AnatomyAgent ──► LesionScreeningAgent
                                                      │  candidates + 25 visual descriptors + tiles
                                                      ▼
                         ┌──────── debate (≤ N rounds) ────────┐
                         │  RadiologistAgent  ⇄  SkepticAgent   │   (+ optional LLM / VLM second opinion)
                         └──────────────────────────────────────┘
                                                      ▼
                             ModeratorAgent (consensus, thresholds, actions required)
                                   │ ECR suggested / possible
                                   ▼
                      ClassifierAgent (Patel 3D) ──► TreatmentPlannerAgent (matrix + RL policy)
                                                      ▼
                            conference_report.json / conference_summary.md / masks / tiles
```

All agents read and write a shared **blackboard** (`agents/blackboard.py`). Each `Evidence`
item carries agent, kind, payload, basis (`measurement` | `rule` | `llm` | `clinician`),
confidence, caveats, candidate id and debate round. Nothing is overwritten; disagreements are
added as counter-evidence and the moderator resolves them. The blackboard is serialised into
the report (`conference` key) together with a human-readable transcript.

### Agents

| agent | input | output | basis |
|---|---|---|---|
| `ImagingQAAgent` | volume geometry, QA result | fitness for ECR assessment (voxel size, FOV, hyperdense material) | measurement |
| `AnatomyAgent` | tooth segmenter, seed | crop, tooth/canal/bone masks, thresholds, canal mode, reliability caveats | measurement |
| `LesionScreeningAgent` | crop | `CervicalResorptionDetector` candidates, `CandidateDescriptors`, orthogonal tiles | measurement |
| `RadiologistAgent` | descriptors (+ tiles for a VLM) | per-candidate differential over 6 hypotheses, criteria for/against, band | rule (+ llm) |
| `SkepticAgent` | descriptors + radiologist evidence | challenges with severity (landmarks, artefacts, mimics, size, truncation) | rule (+ llm) |
| `ClassifierAgent` | accepted candidate mask | Patel height/spread/proximity with margins and borderline flag | measurement |
| `TreatmentPlannerAgent` | Patel code, clinical inputs, policy | matrix primary/alternatives, constrained-bandit ranking, contingencies, follow-up | rule + policy |
| moderator (`CaseConference._moderate`) | all of the above | status, suggested diagnosis/plan, clinician actions | rule |

### Visual descriptors (`agents/visual_features.py`)

Per candidate: volume; coronal/apical extent from the CEJ; fraction and extent below the bone
crest; external and cervical surface connection; portal area; maximum internal cross-section;
**undermining ratio** (max cross-section / portal area); maximum depth; angular spread; minimum
canal distance and canal-wall integrity; margin roughness at 0.3 mm; axial elongation; mean
intensity and heterogeneity relative to dentine; radiopaque inclusion fraction inside the defect
envelope (repair-tissue proxy); dense-material contact; near-metal fraction (root fillings in the
canal proxy excluded); threshold stability; crop-boundary contact; landmark source and flags.

### Question-driven diagnosis (rev 2.5, `agents/criteria.py`, `agents/clinical.py`)

The radiologist agent answers the eight questions a clinician works through, each from named
descriptors with the evidence string kept in the report:

| # | question | descriptors used |
|---|---|---|
| Q1 | Arising from the external cervical root? | surface connection, portal persistence across levels, eccentricity to the canal, cervical band |
| Q2 | Independent of the canal (external, not internal)? | canal traceable fraction, canal-space expansion ratio |
| Q3 | ECR growth pattern (enters one side, wraps)? | portal angular width vs total spread (wrap ratio), undermining ratio |
| Q4 | Preserved pericanalar dentine while wrapping? | minimum canal distance, canal wall intact, spread > 180° |
| Q5 | Resorption rather than caries / smooth cavity? | roughness, asymmetry, inclusions, apical extent, dense-material contact (score 0–5) |
| Q6 | Persists in three planes as a 3-D defect? | axial levels, in-plane extents, threshold stability, metal proximity |
| Q7 | How extensive? | Patel height / spread / pulp relationship (classifier agent) |
| Q8 | Does the clinical picture fit? | clinician-entered `ClinicalFindings` (pink spot, hard irregular defect, bleeding on probing, pulp sensibility, history, restoration at defect) |

Seven cue-level criteria back these questions in the posterior (external contour interrupted,
eccentric to canal, canal traceable, canal balloons into lesion, enters-one-side-and-wraps,
asymmetric infiltrative, persists in three planes) and **internal resorption** was added to the
differential. Summed log-likelihood ratios are tempered (÷1.6) because the cues are correlated.
Q8 shifts the posterior only for findings that were actually entered and reports
`consistent_with_ecr` / `inconsistent_with_ecr` / `mixed` / `not_assessed`.

### Differential diagnosis (`agents/criteria.py`)

Hypotheses: ECR, internal resorption, caries, non-carious cervical lesion, restoration margin /
cervical burnout, anatomical concavity / furcation, imaging artefact. Twenty-one criteria map descriptors to
log-likelihood-ratio weights per hypothesis (e.g. *small portal with wide interior*: ECR +1.5,
NCCL −1.5; *near metal*: artefact +2.0). Sources: Patel et al. 2018 (IEJ 51:1205, 51:1224),
ESE 2023 position statement (IEJ 56:1183), Mavridou et al. 2016 (JOE 42:1737), DMFR 2023
adolescent mimics study. Weights are **expert-set engineering priors, not calibrated**; the
report prints `weights_version` and `calibrated: false`.

Confidence band: `low` < 0.35 ≤ `moderate` < 0.60 ≤ `high`, and the band is **capped at
moderate** whenever the CEJ/apex are automatic, the canal is a proxy, or landmark flags exist.

### Debate

Round r: radiologist assesses → skeptic challenges → if the skeptic produced new challenges,
the radiologist re-assesses with a log-odds penalty per challenge (major 0.8, moderate 0.4,
minor 0.15). Model-generated challenges are capped at `moderate`. The loop stops when the
skeptic has nothing new, or after `max_rounds` with one closing reply.

### Model backends (`agents/llm_backend.py`)

* `NullBackend` – rules only (default when nothing is configured).
* `PerplexitySDKBackend` – text-only structured extraction (`pplx_sdk.llm`) inside Perplexity Computer.
* `OpenAICompatibleBackend` – any `/v1/chat/completions` endpoint; sends the candidate tiles as
  images when `supports_images=True` (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`).

The model's differential is averaged 50/50 with the rule posterior and both are stored; the
model can lower the final confidence but cannot lift the band above the anatomy cap.

### Status values

| status | meaning |
|---|---|
| `suggested_ecr_with_plan` | leading hypothesis ECR with P ≥ 0.50 → Patel code + plan suggested |
| `ecr_possible_conditional_plan` | 0.35 ≤ P(ECR) < 0.50 or ECR not leading → plan shown *conditional on clinical confirmation* |
| `no_ecr_suggested` | leading alternative with generic work-up steps; **does not exclude ECR** |
| `no_candidates` | screen retained nothing; **does not exclude ECR** |

### Learning over time

* **Treatment policy**: unchanged constrained Thompson-sampling bandit with doubly-robust
  off-policy evaluation and a manual promotion gate (`docs/ECR_CDS_Design.md`).
* **Diagnosis**: `record_diagnosis_feedback()` logs the clinician-confirmed diagnosis for the
  conference's leading candidate (idempotent per case). `RadiologistAgent.update_weights_from_feedback()`
  refines only the ECR column of the criterion weights by ridge-penalised logistic descent toward
  the expert prior, and only once ≥ 20 independent confirmed cases exist. Tuned weights are
  versioned (`feedback_tuned_nNN`) and can be passed back via `CaseConference(radiologist_weights=...)`.

## Results on phantoms (software tests, `tests/test_agents.py`)

| phantom | conference status | Patel | matrix primary |
|---|---|---|---|
| ECR morphology, 3 mm, 150°, no perforation | `suggested_ecr_with_plan` | 2Bd (truth 2Bd) | external_repair, contingencies on pulp vitality / symptoms |
| ECR morphology, 9 mm, 150°, perforating | `suggested_ecr_with_plan` | 3Bp (truth 3Bp) | intentional_replantation (irreversible → confirmation flagged) |
| NCCL saucer, 1 mm, 80° | `no_ecr_suggested` (leading NCCL) | – | – |
| intact tooth | `no_candidates` | – | – |
| internal resorption (canal-centred balloon) | `no_ecr_suggested`, leading internal_resorption; Q1 no, Q2 no | – | RCT work-up suggested |
| open wedge + pink spot, bleeding, hard defect | `suggested_ecr_with_plan`, Q8 consistent | 2Bp | external_repair_rct |
| scripted model voting "caries" | logged as second opinion; P(ECR) falls, band ≠ high | | |

Open wedge without findings sits near the decision threshold by design. Phantom geometry is synthetic; these tests show the code behaves as designed, not that it
detects ECR in patients.

## Result on the supplied 501-slice scan (case 81a510c8d8ca, rev 2.5, text LLM second opinion, 20/20 model calls)

Four components were examined on the automatically chosen root-filled tooth (two cervical candidates
plus two deep external components outside the estimated cervical band). For the largest cervical
candidate (11.0 mm³) the questions answered: Q1 external origin **yes**, Q2 independent of the canal
**no** (canal-space expansion 1.5 at a filled canal), Q3 growth pattern **uncertain** (180° spread but
the opening is as wide as the spread, undermining ratio 0.08), Q4 preserved pericanalar dentine **no**,
Q5 resorption morphology **uncertain**, Q6 multiplanar persistence **uncertain** (66 % near metal), Q8
**not assessed** (no clinical findings entered). Leading hypothesis restoration margin / cervical
burnout (P≈0.23) with ECR ≈0.19 and artefact ≈0.16; the other components favoured caries or root
anatomy with P(ECR) ≤ 0.17 → status `no_ecr_suggested`, no Patel code or plan. The posterior is flat
because the discriminating cues (narrow portal, wrapping, preserved dentine) are absent and the main
confounders (unmeasured CEJ, root filling, metal) are present; entering the clinical findings (Q8) and
reviewed CEJ/apex points is the way to resolve it.

## Rev 2.7 additions

Four roles were added and the conference now runs on top of an explicit perception toolkit (see
`docs/PERCEPTION_AGENT_ARCHITECTURE.md`): **context** (Diagnocat findings + patient chart -> prior shifts and input
hints), **verifier** (critic re-inspection of each candidate under alternative processing, tooth-axis persistence,
provenance-stamped multiplanar evidence frame; challenges are placed before the radiologist's first assessment),
**coverage** (examined vs complaint teeth, missing prerequisites) and **knowledge** (ESE / Patel / AAE citations per
answered question, plan step and gap; CDT codes). The patient-level entry point is
`ecr_cds.diagnocat.pipeline.run_diagnocat_pipeline` (see `docs/DIAGNOCAT_INTEGRATION.md`).

## Rev 2.8 additions

The radiologist role can now be backed by a **local** vision model (NVIDIA Nemotron 3.5 Bolt 74B-A9B via llama.cpp,
Ollama or vLLM; `docs/NEMOTRON_LOCAL.md`). Its input is the verifier's provenance-stamped 8-panel evidence frame of
the candidate (never the raw volume), one image per request. Its output is fused with a weight

    w = 0.5 x (1 - DSE / 0.3),   w = 0 if DSE >= 0.3 (abstention),   w <= 0.2 if the counterfactual test is weak

where DSE is the discrete semantic entropy of five structured answers sampled at temperature 1.0, and the
counterfactual test re-asks the model on a frame in which the candidate voxels were filled with the volume's own
dentine level. Abstention leaves the rule posterior unchanged. The verifier gained a `tta` mode (gain, noise,
threshold-jitter and smoothing paths, reporting the minimum Dice) and an unsharp-mask display option; both are off by
default and were ablated against the frozen baseline. The knowledge agent cites the `vlm_limits` statements whenever a
model opinion was used, and the consensus markdown gets a "Vision-language second opinion (local model)" section with
per-candidate weight, entropy and grounding verdict.

## Limitations

* Everything downstream of the automatic masks inherits their errors; the CEJ is the weakest link
  (supply `CEJ_ZYX`/`APEX_ZYX` to remove the major challenge).
* Criteria weights, thresholds and the 50/50 model blend are design choices without clinical
  calibration; posteriors are not probabilities of disease.
* Text-only backends never see the images; vision backends see one multiplanar evidence frame per candidate
  (rev 2.8; earlier revisions sent three 12 mm tiles). The local Nemotron path was tested only against a scripted
  stand-in server, not the real model.
* The planner reuses the Patel→ESE matrix; its recommendations remain provisional until the
  clinician enters reviewed components and clinical inputs (notebook Section 5).
