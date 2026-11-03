# Seeing vs reasoning: the two-layer architecture (rev 2.7)

Principles adopted from the user's architecture notes, and how each is implemented. None of the changes is assumed
to improve the clinical task; `ecr_cds.evaluation` compares them with a frozen baseline on synthetic cases, and the
same harness is meant to be run on labelled clinical scans.

## 1. Perception layer (`ecr_cds/perception/`) - only route to the voxels

| tool | does | returns |
|---|---|---|
| `segment_teeth` | data-driven (Otsu) tooth / bone / dense labelling | `AutoSegmentation` with thresholds in scanner units |
| `propose_regions` | ranks teeth by radiolucent defect volume, excludes metal-flagged teeth | scores, seed |
| `delineate_tooth` | tooth + canal masks for a seed (expert canal mask honoured) | `ToothSegmentation` |
| `reference_levels` | soft-tissue, dentine, enamel and metal reference levels **relative to this volume** | levels + method string |
| `metal_mask` | metal core and 3 mm streak zone; root filling excluded from the core | masks, fractions |
| `evidence_frame` | linked multiplanar evidence with provenance (below) | `EvidenceFrame` (PNG + JSON) |
| `reproduce_candidate` | re-delineates a candidate under other smoothing paths; Dice with the original; extent along the tooth axis | reproducibility, axis persistence |

Every call is logged with arguments and a summary (`PerceptionToolkit.log`) and the log is attached to the report,
so the transcript shows which tools were used and why.

Intensity handling: CBCT grey values are not Hounsfield units, so no absolute cut-off is used anywhere; tooth,
dense and metal levels come from Otsu splits, tissue medians, hard-voxel modes and percentiles of the volume itself
(the previous `t_dense + 3000` metal level is replaced by `enamel_mode + 2 x dentine-enamel contrast`). Display
transforms (percentile window, optional CLAHE for viewing) are recorded with every panel and never feed a
measurement. No generative enhancement exists in the evidence path.

### Evidence frames (`perception/evidence.py`)

Per candidate: three axial neighbours (centre +-0.8 mm), coronal, sagittal, a tooth long-axis reformat and the
perpendicular plane, and the unmodified full axial slice with the crop box drawn. The JSON records source id, voxel
spacing, crop offset, every panel's slice index and extent in **volume** voxel coordinates, the oblique frame
vectors, and the display transform. A single slice never represents the volume.

## 2. Agent layer (`ecr_cds/agents/`) - selects tools, verifies, tracks coverage, retrieves knowledge

```
ContextAgent      Diagnocat + chart -> prior shifts, input hints (never findings)
ImagingQAAgent    acquisition / quality checks
AnatomyAgent      tooth, canal proxy, thresholds, root-filling flag
LesionScreening   candidates + Q1-Q7 descriptors (perception measurements)
VerifierAgent     critic re-inspection: reproduce_candidate + evidence_frame; low reproducibility or
                  single-plane extent -> challenge (never silent deletion)
RadiologistAgent  posterior from criteria with the context prior; answers Q1-Q7; optional model second opinion
SkepticAgent      adversarial challenges (artefacts, landmarks, mimics); penalises ECR log-odds
ClinicalCorrelationAgent  Q8 from ClinicalFindings (chart hints fill unset fields)
FeatureMeasurementAgent / ExtentSummaryAgent   VLM visual features + voxel measurements -> descriptive extent (rev 2.9)
TreatmentPlannerAgent                     measurement-indexed safe set + constrained policy
                                          (legacy opt-in: ClassifierAgent Patel 3-D code + matrix)
CoverageAgent     examined vs chief-complaint teeth, missing prerequisites with how-to-resolve
KnowledgeAgent    guideline citations for each answered question, plan step and gap; CDT codes
Moderator         status, statement, actions required
```

The critic loop is the detect -> verify -> reason order: screening detects, the verifier re-measures the same
region through a different processing path and in the tooth-axis frame, and only then does the radiologist
reason, with the verifier's challenges already on the table.

Knowledge is retrieval for interpretation: the KnowledgeAgent can only attach statements to findings that the
imaging layer established or to prerequisites that are missing. It has no path to add a candidate.

## 3. Context (RAG with the chart) and contextual filtering

`PatientChart` (age, chief complaint with FDI teeth / quadrants, symptoms, history events per tooth) is retrieved
per tooth. Teeth in the complaint focus and teeth with Diagnocat flags are conferenced first (`triage_order` in the
patient report); teeth below the `max_teeth` cut are listed as not examined, and the CoverageAgent flags
complaint teeth that were not conferenced. History fills `ClinicalModifiers` / `ClinicalFindings` only where the
clinician left them unset, and is shown in the report as "Upstream context".

## 4. Evaluate the layers separately (`ecr_cds/evaluation/`)

* Perception: best-candidate Dice with the true lesion, centroid hit within 1 mm, false candidates per case,
  verifier reproducibility on true vs false candidates.
* Report: top-1 diagnosis vs label, ECR recall / false-positive rate, Brier of P(ECR), plan primary.
* `run_ablation(variants, n_cases)` runs each variant and the **frozen baseline** (rev 2.6 behaviour: no verifier,
  no context, one round) on identical synthetic cases with injected confounders and reports paired deltas.
  `python -m ecr_cds.evaluation --n-cases 8` reproduces the table in `ecr_ablation_v27/`.

A convincing explanation is not evidence that the perception was right: the report metrics are only meaningful
conditioned on the perception metrics, which is why both are printed side by side.

## 5. Preprocessing additions used for display / QA (not as disease rules)

* CLAHE (`perception.display.clahe2d`, numpy): sharpens enamel / dentine / bone boundaries for viewing and VLM
  prompts when `clahe_display=True`; parameters recorded in each panel.
* Metal masking (`perception.display.metal_mask`): relative metal level, core + streak zone; the SkepticAgent's
  near-metal challenge and the Diagnocat `metal_expected` flag both use it.
* Orientation: DICOM orientation and spacing are kept from the loader; oblique reformats are computed from the
  tooth axis rather than by resampling the whole volume.

## 6. What is still open

* Dental VLMs report task-dependent failures and tooth-localisation limits; the model second opinion stays a 50/50
  blend that cannot raise the confidence band, and images given to it are the provenance-stamped frames.
* The fallback seeding without Diagnocat numbering is by radiolucency rank, not by FDI number.
* Ablation results are on phantoms. Real labelled scans are required before any variant is adopted as default.
