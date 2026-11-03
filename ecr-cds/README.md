# ECR-CDS — CBCT-based clinical decision support for external cervical resorption

Revision 2.3 defaults to experimental ECR candidate screening, not diagnosis.
See `docs/ECR_DETECTION_2.3.md`. The earlier rule-derived 2Bp interpretation
is superseded: no expert-reviewed ECR diagnosis was established on that scan.

See `docs/ECR_CDS_Design.md` for the full design. Quick start:

```bash
pip install pydicom scikit-learn scipy numpy
python tests/test_pipeline.py          # builds a 500-slice DICOM phantom and runs every stage
```

```python
from ecr_cds import run_pipeline, ClinicalContext
from ecr_cds.decision_matrix import ClinicalModifiers
report = run_pipeline("/path/to/dicom_folder", tooth_segmenter, lesion_segmenter,
                      modifiers=ClinicalModifiers(symptomatic=False, pulp_vital=True),
                      tooth_type="premolar", age_years=38, expected_slices=500)
print(report["status"])  # ecr_not_confirmed until explicitly reviewed
```

For the cervical screen use the notebook's default `LESION_SOURCE = "ecr_candidate"`.
`ECR_CONFIRMED` must not be set from an automatic mask or a model prediction.
It is a clinician attestation. The candidate screen itself never issues treatment.

## No label map? (`ecr_cds.auto_segment`, rev 2.2)

`auto_segment()` produces a classical teeth/bone label map, proposes the seed (radiolucency evidence, else FOV centre), detects a root filling as canal, and `RadiolucencyLesionDetector` offers a model-free lesion fallback. All outputs are labelled `auto_classical` / `rule_based_radiolucency`; see `docs/REVIEW_RESPONSE_2.1.md`, section 2.2, for behaviour on a patient scan and limitations.

## Multi-agent case conference (rev 2.4)

`ecr_cds.agents` runs the agents (imaging QA, anatomy, lesion screening, radiologist, skeptic, feature measurement,
extent summary, treatment planner, moderator; the Patel classifier only as a legacy opt-in since rev 2.9) over a
shared blackboard and returns a **suggested** diagnosis and treatment plan with a full evidence trail. Optional
LLM/VLM second opinion via `auto_backend()`; rules-only otherwise.

```bash
python -m ecr_cds.cli conference --dicom scan.zip --out conference_out --llm auto   # or --llm none
```

See `docs/MULTI_AGENT_CONFERENCE.md`. Posteriors are uncalibrated and the output is not a diagnosis.

## Diagnocat -> agentic system, perception/agent split (rev 2.7)

* `ecr_cds.diagnocat`: ingest a Diagnocat study (Partner API client with configurable endpoints, JSON export, report
  PDF, STL voxeliser), map per-tooth conditions and a `PatientChart` to prior shifts / clinical-input hints / triage
  priority, and run prioritised per-tooth conferences (`run_diagnocat_pipeline`, CLI `diagnocat-run`). Diagnocat has
  no ECR class; it supplies anatomy, numbering and context, our screen and conference do the resorption reasoning.
* `ecr_cds.perception`: the only route to the voxels - segment, propose, delineate, measure, volume-relative
  reference levels (no absolute CBCT cut-offs), metal/streak mask, CLAHE for display only, provenance-stamped
  multiplanar evidence frames (axial neighbours, coronal, sagittal, tooth-axis reformats, full-slice context), and
  `reproduce_candidate` for critic re-inspection. Every call is logged into the report.
* New agents: context, verifier (detect -> verify -> reason), coverage (missing prerequisites, unexamined
  complaint teeth) and knowledge (ESE 2023 / Patel / AAE CDT citations per question, plan step and gap; CDT codes).
* `ecr_cds.evaluation`: perception metrics and report metrics kept separate, ablation against a frozen baseline
  (`python -m ecr_cds.evaluation --n-cases 8`).

```bash
python -m ecr_cds.cli diagnocat-run --dicom scan.zip --diagnocat-json export.json --chart chart.json --out out
```

See `docs/DIAGNOCAT_INTEGRATION.md` and `docs/PERCEPTION_AGENT_ARCHITECTURE.md`.

## Local vision model and recognition-accuracy work (rev 2.8)

* `NemotronLocalBackend` (`ecr_cds/agents/llm_backend.py`): NVIDIA Nemotron 3.5 Bolt 74B-A9B (or any local VLM)
  behind an OpenAI-compatible server (llama.cpp / Ollama / vLLM). One provenance-stamped evidence frame per request,
  five samples at the model-card temperature, discrete-semantic-entropy abstention, thinking toggle, JSON output.
  `--llm nemotron` or `NEMOTRON_BASE_URL=...`. No data leaves the machine.
* The model is a second opinion: weight capped at 0.5, 0 on abstention, 0.2 when a candidate-filled counterfactual
  frame fails to change its answer (`--counterfactual`). It never detects, measures or stages.
* Verifier `--verify-mode tta` (gain / noise / threshold-jitter / smoothing paths), `--sharpen-display` unsharp mask
  (display only), review-budget and AUROC metrics, three cited `vlm_limits` knowledge statements.
* The build sandbox had no GPU: the backend is tested against a scripted stand-in server only
  (`tests/test_nemotron_backend.py`). Run `python -m ecr_cds.evaluation --llm nemotron` on your hardware first.

See `docs/NEMOTRON_LOCAL.md` (serving recipes, guards, caveats) and `docs/RECOGNITION_ACCURACY_RESEARCH.md`
(evidence table, adopted vs rejected, ablation).

## VLM feature measurement and measurement-based extent summary (rev 2.9)

* `ecr_cds.perception.vlm_features` + `feature_schema`: for every candidate, a typed `FeatureBundle` of categorical
  **visual features** read by a local OpenAI-compatible VLM (radiolucency texture, borders, fibro-osseous interior,
  pericanal line, portal visibility / surface, restoration-margin relation, burnout, beam hardening) and 13
  **physical measurements** computed from the voxel geometry (long-axis / MD / BL extents, limits from the CEJ and
  bone crest, depth, circumferential extent, dentine volume fraction, canal distance / communication, volume), each
  with confidence and provenance. The model never sees the numbers; its estimates are cross-checked, geometry
  always wins, and values geometry cannot verify are flagged `vlm_unverified` and excluded from planning.
* `FeatureMeasurementAgent` posts the bundle; radiologist, skeptic, clinical-correlation and planner consume it.
  `ExtentSummaryAgent` replaces the Patel classifier on the default path with a descriptive extent summary
  (extent / pulp relationship / access descriptors + narrative, **no staging codes**), and the planner's safe set is
  keyed by that summary through a documented measurement-indexed mapping (`docs/measurement_matrix.csv`).
  `--staging-scheme patel_3d` reproduces the rev <= 2.8 outputs.

```bash
NEMOTRON_BASE_URL=http://127.0.0.1:8080/v1 python -m ecr_cds.cli conference --dicom scan.zip --out out --llm nemotron --buccal 0,1,0
python tests/test_vlm_features.py   # schema, geometry vs phantom truth, flagging, mapping, conference e2e with a fake VLM (~4 min)
```

See `docs/VLM_FEATURE_MEASUREMENT.md`.

## Tests

```bash
python tests/test_pipeline.py      # pipeline on a 500-slice phantom
python tests/test_robustness.py    # DICOM/ZIP/mask geometry, clinical controls, feedback, policy audit
python tests/test_ecr_detection.py # cervical-resorption screen logic
python tests/test_agents.py        # multi-agent conference on phantoms (~9 min)
python tests/test_finetune.py      # fine-tuning data pipeline (~1.5 min, no GPU)
python tests/test_diagnocat.py     # Diagnocat ingestion, context priors, perception tools, knowledge/CDT (~1.5 min)
python tests/test_ecr_detection.py # cervical vs noncervical/internal fixtures and abstention
python tests/test_vlm_features.py  # rev 2.9 feature-measurement layer (fake VLM server, ~4 min)
```

## Colab notebook

`ECR_CDS_Colab.ipynb` is standalone (package + model embedded) and runs on a patient DICOM folder/ZIP by default; phantoms are opt-in. See `docs/REVIEW_RESPONSE_2.1.md` for the changes made after the external review.
The bundled model `ecr_cds/models/lesion_svm_phantom.joblib` (+ `.card.json`) was trained on five synthetic
phantom cases only. Training is in Section 6; clinically labelled positives,
negatives and mimics plus independent validation are needed before clinical deployment.

## Fine-tuning the second-opinion model (rev 2.6)

`ecr_cds/finetune/` generates supervised data from the conference itself (synthetic phantoms with known truth and
injected confounders, plus clinician-confirmed cases), trains a LoRA adapter (`train_lora.py`, Colab GPU) or exports
for hosted fine-tuning (`export_openai.py`), and scores any backend on held-out prompts (`evaluate.py`).
`LLM_LOCAL_MODEL=<adapter>` makes the conference use the tuned model. See `docs/FINETUNING.md` for what this does and
does not improve: it is reasoning/format fidelity on synthetic truth, not clinical validation.
