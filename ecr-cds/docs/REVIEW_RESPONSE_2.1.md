# Response to review report 2.0.0 — changes in ecr_cds 2.1

Revision 2.3 correction: the patient-specific 2Bp interpretation and generic
radiolucency treatment output later in this historical document are superseded.
They were not validated ECR findings. See `ECR_DETECTION_2.3.md`; automatic
detection now returns review candidates and withholds diagnosis and treatment.

Each finding from `ECR_CDS_Review_Report.md` (4 Oct 2026) is mapped to the change made and the test that covers it. Tests: `tests/test_pipeline.py` (pipeline, 5 tests) and `tests/test_robustness.py` (8 tests, 8 DICOM rejection classes); both pass.

| # | Review finding | Change in 2.1 | Where | Test |
|---|---|---|---|---|
| 1 | Required package absent; notebook could not initialise | Package, phantom model and training card are **embedded in the notebook** (base64, 89 KB) and extracted on first run; an external `ecr_cds.zip` is optional | `build_notebook.py`, Section 1 | Notebook executed end-to-end from the embedded copy in a clean directory |
| 2 | Defaults to synthetic cases | Patient scan (`SCAN_PATH`, folder or ZIP) is the default; phantoms require `USE_DEMO_PHANTOM = True` and every phantom output is labelled | Section 2–3 | Both modes executed |
| 3 | `CASES[1]` fixed selection → IndexError | Single-case flow; no list indexing. Series chosen by largest series, `series_uid` override available | Section 2, `load_dicom_series` | — |
| 4 | Model / training card unavailable; metrics unverified | Card shipped alongside the model (`models/lesion_svm_phantom.card.json`) with recipe, LOPO CV, Youden table and library versions; the pipeline derives an `evidence_level` from the card (`research_phantom_model`, `internal_model`, `validated`, `expert_mask`, `unvalidated_model`) and sets report status to `provisional` unless validated | `pipeline._evidence_level` | `test_end_to_end_pipeline_on_dicom` |
| 5 | Empty lesion overlay crashes (NaN centroid) | Empty mask → `no_lesion_detected` status with note; overlay falls back to the seed slice | `pipeline.run_pipeline`, Section 4 overlay | Notebook path |
| 6 | Phantom defaults contaminate patient reports | All `ClinicalModifiers` default `None`; the matrix lists `missing_inputs`, flags `incomplete_clinical_inputs`, sets `requires_confirmation=True` and status `provisional` | `decision_matrix.recommend_from_matrix` | `test_clinical_controls`, `test_matrix_complete_and_rules` |
| 7 | Automatic invented outcome logging | Feedback off by default; `Feedback` validates that outcomes carry an observed follow-up in (0, 240] months and that a lost tooth cannot be a success; event IDs `case:decision` / `case:outcome:<months>` make updates idempotent; `policy.n_cases` counts independent cases; `audit_log.jsonl` written | `rl_policy.Feedback`, `ConstrainedBanditPolicy.update`, Section 7 | `test_feedback_idempotent_and_validated` |
| 8 | In-sample threshold selection | `training.nested_patient_cv`: outer leave-one-patient-out, inner patient-fold threshold selection, duplicate-data guard, ≥4 patients required; notebook training uses it exclusively | `training.py`, Section 6 | `test_nested_cv_guard`, notebook run (5 phantom "patients") |
| 9 | Policy evaluation not independent | `split_log_by_case` holds out the most recent 30 % of cases; candidate refit on the rest; DR estimate with **case-level** bootstrap; gate requires ≥50 independent held-out cases; promotion is manual | `rl_policy.offline_evaluate / promotion_gate`, `cli evaluate`, Section 8 | `test_heldout_policy_evaluation` |
| 10 | Geometry safeguards unverifiable | Loader: physical sorting along the slice normal, rejection of gaps, duplicate locations, duplicate SOP UIDs, orientation/spacing/dimension changes, in-plane drift, multi-frame, invalid rescale; per-slice rescale; `PixelPaddingValue` handled separately; anisotropic spacing kept per-axis. Masks: NIfTI affine → RAS→LPS axis reordering/flips, shape/spacing/origin mismatch rejection, explicit opt-in resampling | `dicom_loader.py`, `adapters.load_labelmap` | `test_physical_sort_rescale_padding_deid`, `test_rejections`, `test_mask_geometry` |
| 11 | Exports assume prior training | Export lists only files generated in the current run; raw DICOM and scan cache excluded | Section 9 | Notebook path |
| 12 | Archive handling | ZIP members read with path sanitisation; `__MACOSX/`, `._*`, `.DS_Store`, non-DICOM files skipped and counted in `Housekeeping` | `dicom_loader.extract_dicom_zip` | `test_zip_housekeeping_and_unsafe_paths` |
| 13 | De-identification | Reports store `PatientID`, Study/Series UIDs only as 12-char salted SHA-256 hashes; names never read | `dicom_loader._hash_id` | `test_physical_sort_rescale_padding_deid` |
| 14 | Incomplete observations must withhold a code | `manual_patel_code` returns `None` if any component is unknown → `insufficient_evidence`, no recommendation | `decision_matrix.manual_patel_code`, Section 5 | `test_clinical_controls` |

## Not changed, deliberately

* The ExtraTrees baseline from the review bundle is not adopted; the texture-SVM remains the reference segmenter because its behaviour is documented in Ravikumar et al. Any model can be plugged in through the `LesionSegmenter` hook.
* Widgets are not used; seed and ROI are explicit variables with a static three-plane check, so behaviour is identical in Colab, Jupyter and headless execution.

## What still cannot be concluded

The bundled model has seen synthetic data only. No Patel code, Dice, sensitivity or treatment recommendation produced from it constitutes clinical evidence. Clinical validation requires expert tooth/canal/lesion masks with stable patient identifiers (including negatives), nested evaluation with an external cohort, reader-agreement analysis, and, for the treatment policy, authentic longitudinal outcomes.

---

# Revision 2.2 — first run on a patient scan (case `81a510c8d8ca`)

## Inputs received
* 501-slice CBCT ZIP (J. Morita unit, 89 kVp, anonymised with DicomCleaner; 501 `__MACOSX` entries skipped). Loader QA: 501 × 503 × 503 voxels, 0.08 mm isotropic, FOV 40 mm, slice-position residual 1.8e-15 mm, rescale 1/0, intensities −11 629…28 426 (not HU-calibrated).
* Carestream CS 3D Imaging cache files (`FilesManager.xml`, `*.serie`, `*_2.dat`): viewer metadata and a 2× sub-sampled volume cache — **not** a label map or canal mask. No DentalSegmentator output or expert masks were available, and no seed was known.

## What was added so the system can proceed without external masks (`ecr_cds/auto_segment.py`)
| Component | Method | Label in report |
|---|---|---|
| Teeth/bone label map | Otsu thresholds (`t_hard` soft/mineralised, `t_dense` dentine/enamel-filling), 0.8 mm opening on a 0.16 mm grid to detach roots from cortical plates, component filters (≥50 mm³, ≥6 mm long, contains dense material), coarse closing, full-resolution refinement inside the teeth bounding box; written to NIfTI for review in Slicer | `auto_classical` |
| Seed proposal | Candidates ranked by in-plane distance to the FOV centre; overridden by the tooth with the largest external radiolucent defect (≥5 mm³) unless flagged as metal/artefact | `seed_source = auto_fov_centre` / `auto_radiolucency` |
| Canal | Hyperdense central material (root filling / post) when it spans ≥25 % of the tooth length, else dark pulp (Otsu) in the central 45 % radius; expert mask overrides | `canal_mode = hyperdense_filling` / `dark_pulp_otsu` / `expert_mask` |
| Lesion (model-free) | Hypodense (< `t_tooth`) voxels inside the tooth hull (per-slice per-root convex hull ∪ 2.5 mm closing), outside the dilated canal and outside the central dark core, opened 3×, ≥1 mm³, touching the external surface | `rule_based_radiolucency` |
| Pipeline | Crops to the target tooth + 3 mm before normalisation/texture features (a 500-slice float pipeline previously exceeded 7 GB); guideline order is retained when the policy has no feedback for a code (Thompson noise no longer re-ranks) | — |

## Result on the patient scan
* Five tooth candidates. The system selected the tooth nearest the FOV centre, which was also the only non-metal tooth with a radiolucent defect (13.5 mm³ before post-processing): seed `(z, y, x) = (333, 221, 229)`, 301 mm³, 18.7 mm long, a 12.9 mm hyperdense root filling used as canal (`seed_source = auto_radiolucency`).
* Rule-based detector after post-processing: one 5.7 mm³ defect at the cervical margin, buccal/distal aspect at z≈391, coded **Patel 2Bp — provisional, borderline on all three components** (height margin 0.6 mm to level 3, spread 91° vs the 90° A/B boundary, canal distance 0.24 mm vs the 0.30 mm d/p threshold). Matrix cell 2Bp → external repair + RCT (ESE 2023); four clinical inputs missing, so confirmation is required and the policy kept the guideline order (no feedback for this code).
* Three sub-threshold components are listed for review: 3.5 mm³ at (298, 221, 215) — mid-root, at an apparent root bifurcation; 7.1 mm³ at (392, 235, 215) — cervical margin under the crown bulge (the source of the 2Bp call; the classic ECR portal location, but equally consistent with a restoration margin or cervical abrasion); 2.9 mm³ at (390, 190, 231).
* The metal-crowned tooth at (381, 407, 157) produced 107.6 mm³ of "defect" from beam-hardening bands and was excluded by the metal flag.
* The bundled phantom texture-SVM, run for comparison, labelled 75 % of the dentine as lesion (Patel 4Dp); the report flagged it `research_phantom_model`/`provisional`. The failure is the expected domain shift (phantom dentine uniform and bright; real dentine ≈2 200 vs filling ≈11 000 in scanner units).
* Two geometry defects were found and fixed during this run: the enamel-based CEJ estimate locked onto the hyperdense root filling (CEJ placed 5 mm from the apex; now the canal is excluded, only the outer 35 % radial shell is used, and a CEJ more than 3 mm apical to the bone crest falls back to crest − 1.5 mm), and the apex = narrow-end heuristic can invert on crown-truncated teeth (now overridden by the bone-coverage cue when it is unambiguous).

## Limitations established in this run
* The rule-based detector is a fallback: on phantoms it finds small/moderate defects (Dice ≈0.65, ~40 % under-sized) but misses large wedge lesions (Dice <0.15) because the hull of what remains no longer approximates the original contour; proximity `p` can be missed because the central core is excluded. Large lesions are obvious to the clinician and should be entered through the reviewed Patel components (Section 5).
* `auto_classical` segmentation is expected to fail near large metal restorations and for teeth cut by the FOV border; its label map must be checked against the overlay.
* Nothing in this run is a diagnosis: status `no_lesion_detected` means "no defect found by a rule with unknown sensitivity", not "no ECR".

## Revision 2.4 - multi-agent case conference

Request: "a fully multi-agent system that uses the visual information gathered from the scan and suggests a
diagnosis and a treatment plan". Implemented in `ecr_cds/agents/` (design in `docs/MULTI_AGENT_CONFERENCE.md`):

* Eight agents over a shared blackboard with a serialised evidence trail and transcript.
* 25 quantitative visual descriptors per candidate (portal area, undermining ratio, subcrestal extent, canal-wall
  integrity, margin roughness, radiopaque inclusions, dense-material and metal proximity, threshold stability ...)
  plus orthogonal image tiles for vision models.
* Thirteen literature-derived criteria give an **uncalibrated** differential over ECR, caries, NCCL, restoration
  margin / burnout, anatomical concavity and artefact; a skeptic agent debates the radiologist for up to two rounds.
* Patel 3D code and treatment plan (matrix + constrained RL policy + contingencies for missing clinical inputs) are
  produced only when the conference favours ECR, and are labelled conditional on clinician review.
* Optional LLM/VLM second opinion (`NullBackend`, `PerplexitySDKBackend`, `OpenAICompatibleBackend`); failures fall
  back to rules and are reported in `backend_status`.
* Diagnosis feedback log with gated, ridge-penalised re-tuning of the radiologist's ECR criterion weights (>= 20
  independent confirmed cases).
* Phantom generator gained `morphology="ecr" | "nccl"` to exercise the discriminating criteria; 7 new tests in
  `tests/test_agents.py` (27 test functions in total).

Honesty constraints kept: `requires_confirmation` always true, `clinical_validation` false, posteriors flagged as
uncalibrated, confidence band capped while the CEJ/apex/canal are automatic, negative results explicitly non-exclusionary.

## Revision 2.5 - question-driven diagnosis

Request: focus on getting the diagnosis and treatment right from the clinician's visual cues rather than on delineating
the lesion. Changes:

* Eleven cue-level descriptors added per candidate: eccentricity to the canal, canal traceable fraction, canal-space
  expansion ratio, portal angular width, wrap ratio (spread / portal width), external-contour interruption persistence,
  axial levels spanned, in-plane extents, 3-D persistence, angular asymmetry.
* The radiologist agent answers the clinician's questions Q1-Q7 (external origin, independence from the canal, growth
  pattern, preserved pericanalar dentine, resorption-vs-caries morphology, multiplanar persistence, extent) with the
  evidence string recorded; a clinical-correlation agent answers Q8 from clinician-entered `ClinicalFindings`.
* Internal resorption added to the differential (canal-centred, ballooning, no surface breach); large internal
  components and deep external components outside the estimated cervical band are now examined for the differential
  instead of being silently dropped.
* Summed criterion log-LRs tempered (÷1.6) so correlated cues cannot saturate the posterior; deep-root location
  criterion added so root caries is not proposed for defects far from the CEJ.
* Phantom morphologies `internal` added; 9 agent tests (29 test functions in total).

The output remains a suggestion: `requires_confirmation` true, posteriors uncalibrated, band capped while anatomy is
automatic, negative results explicitly non-exclusionary.

## Revision 2.6 - fine-tuning pipeline for the model roles

Request: fine-tune the LLM so it is more accurate and fits the task. Delivered `ecr_cds/finetune/`: a recording
backend that captures the exact run-time prompts; a synthetic case generator with labelled confounders (noise, metal
streaks, partial volume, grey-level drift, restoration band); label-smoothed differential targets decided by overlap
with the true lesion; a converter for clinician-confirmed conference reports; case-wise splitting; a LoRA/QLoRA
trainer and a hosted-API exporter; an evaluator with the same metrics for rules, hosted and tuned models; and
`LocalHFBackend` so the adapter plugs into the conference unchanged. Honesty constraints are unchanged - the tuned
model remains a 50/50 second opinion that cannot raise the confidence band, and synthetic metrics are explicitly
not clinical validation.

## Revision 2.7 - Diagnocat intake, perception/agent split, critic re-inspection, knowledge and CDT

Requests: (a) "a system that uses Diagnocat to scan CT scans for ECR and takes the findings from Diagnocat into an
agentic AI system with knowledge to provide diagnosis and a treatment plan"; (b) architecture guidance - separate
seeing from reasoning, preserve image evidence, no universal CBCT intensity rule, detect-verify-reason critic loop,
RAG with the patient chart and contextual filtering, test perception separately from report quality.

Finding that shapes (a): Diagnocat has **no external / cervical resorption class** (AI 4.0 manual), and in an
independent CBCT validation (the attached JCM paper) the 14 external-resorption cases produced no AI output. It is
therefore used as the upstream anatomy / numbering / per-tooth context / clinician-flag provider, and the ECR
reasoning stays in this package. The Partner API is partner-gated (Postman collection via api-support@diagnocat.com),
so the client is configurable and unverified; JSON export, report-PDF text and STL meshes are the file routes.

Delivered:

* `ecr_cds/diagnocat/` - interchange schema with condition-name normalisation, loaders (JSON with field mapping,
  PDF text, STL voxeliser), configurable Partner API client, per-tooth `ToothContext` (prior shifts, clinical-input
  hints, canal hint, triage priority), patient-level `run_diagnocat_pipeline` and CLI `diagnocat-run`.
* `ecr_cds/agents/chart.py` - `PatientChart` with per-tooth retrieval (history events, complaint focus, symptoms),
  FDI / Universal numbering; complaint teeth are conferenced first and unexamined ones are flagged.
* `ecr_cds/knowledge/` - 20 cited statements (ESE 2023, Patel 2018, DMFR 2023, AAE CDT 2024, Diagnocat docs) and
  CDT-code mapping per plan option and tooth type.
* `ecr_cds/perception/` - toolkit with call log; volume-relative reference levels replace the last absolute offset
  (metal level); metal core + streak zone; CLAHE for display only; provenance-stamped multiplanar evidence frames
  with tooth-axis reformats; `reproduce_candidate` (alternative-processing Dice, axis persistence).
* Agents: ContextAgent, VerifierAgent (challenges on low reproducibility or single-plane extent, before the
  radiologist reasons), CoverageAgent (missing prerequisites with how-to-resolve), KnowledgeAgent (citations per
  question / plan / gap, CDT codes). RadiologistAgent accepts the context prior and records it.
* `ecr_cds/evaluation/` - perception vs report metrics, ablation against the frozen rev-2.6 baseline on identical
  synthetic cases (results in `ecr_ablation_v27/`; phantom-only, hypotheses for clinical testing).
* Synthetic metal in the fine-tuning generator raised from 3500 to 6000 so it sits above the metal reference level.
* Tests: `tests/test_diagnocat.py` (11 functions). Notebook sections 4b (context, verifier, evidence frames) and
  4c (Diagnocat patient-level pipeline).

Unchanged honesty constraints: suggestions only, `requires_confirmation=True`, uncalibrated posteriors, band capped
while anatomy is automatic, negative screen is not exclusion, Diagnocat does not detect ECR, API schema unverified.

## Revision 2.8 - local Nemotron 3.5 Bolt second opinion, entropy abstention, grounding test, TTA verifier

Request: "improve recognition accuracy, use the local model Nemotron 3.5 Bolt 74B instead of an LLM."

* **Local model instead of a hosted LLM.** `NemotronLocalBackend` talks to an OpenAI-compatible server on the
  clinician's GPU (llama.cpp GGUF 24 GB build, Ollama, or vLLM BF16). Verified against the model cards: released
  15 Oct 2026, hybrid Mamba-2/Transformer MoE VLM, 74B total / 9B active, 2-bit 24 GB build described by NVIDIA as
  aggressive and lossy, no medical evaluation. `docs/NEMOTRON_LOCAL.md` has the recipes and flags.
* **Recognition accuracy, honestly scoped.** The literature (see `docs/RECOGNITION_ACCURACY_RESEARCH.md`) does not
  support a VLM as a detector (17% accuracy / 74% hallucination on radiology images; text preferred over image). The
  perception layer therefore stays the detector and the model's influence is governed by three guards: discrete
  semantic entropy over 5 samples with abstention at 0.3 (published protocol), a counterfactual candidate-filled
  frame that caps the weight at 0.2 when the model's answer does not react to the image, and a hard 0.5 cap.
* **Perception-side changes, ablated not assumed.** TTA verifier mode (six processing paths, `dice_min`), optional
  unsharp-mask display (recorded in provenance), review-budget metrics and AUROC. Results versus the frozen rev 2.7
  baseline are in the rev 2.8 case review archive; on the 6 phantoms the TTA mode did not change the ranking or the
  diagnosis relative to the baseline (same top-1, recall and Dice), which is reported as "no measured gain" rather than
  as an improvement.
* **Knowledge.** Three `vlm_limits` statements (VLM accuracy/hallucination, entropy abstention, 2-bit quantization
  loss) are cited automatically whenever a model opinion entered the posterior.
* **Not done.** No GPU in the build environment, so Nemotron itself was exercised only through a scripted stand-in
  server (30 software tests). Nothing here is clinical validation; `requires_confirmation` stays true on every output.
