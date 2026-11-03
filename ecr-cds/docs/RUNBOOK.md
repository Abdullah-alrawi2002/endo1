# ECR-CDS Runbook — running the system with all features

This is the operational guide. Design rationale is in `ECR_CDS_Design.md`.

## 0. Install

> Revision 2.2: without a DentalSegmentator label map set `AUTO_SEGMENT = True` in the notebook (or use `ecr_cds.auto_segment` in code). The system segments teeth/bone, proposes the seed and detects a root filling as canal; `LESION_SOURCE = "radiolucency_rule"` gives a model-free lesion fallback. Confirm the seed and label map on the overlay; see the review response for limitations.

> Revision 2.1 (after review 2.0.0): scans may be a DICOM folder **or ZIP**; geometry is validated strictly (see `docs/REVIEW_RESPONSE_2.1.md`); clinical inputs have no defaults; feedback requires a `case_id` and observed outcomes; `evaluate` scores a candidate on held-out cases only; reports carry an `evidence_level`. Run `tests/test_robustness.py` in addition to `tests/test_pipeline.py`.


```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pydicom scikit-learn nibabel joblib
python tests/test_pipeline.py        # phantom self-test, ~1 min
python examples/full_demo.py         # fabricates every input and drives all CLI commands
```

Verified on Python 3.14, numpy 2.5, scipy 1.18, pydicom 3.0, scikit-learn 1.9.

## 1. Inputs you need per case

| Input | How to get it | Format |
|---|---|---|
| CBCT DICOM folder | Export the native series (≈500 axial slices, isotropic voxel ≤0.1 mm preferred for d/p calls) | one `.dcm` per slice |
| Tooth / bone label map | 3D Slicer → install **DentalSegmentator** extension → load the DICOM → run segmentation → `Segmentations` module → export as label map → save `.nii.gz` **on the same reference volume** | NIfTI, labels 1 = upper skull, 2 = mandible, 3 = upper teeth, 4 = lower teeth, 5 = mandibular canal (verify in the segment table and pass `--teeth-labels/--bone-labels` if renumbered) |
| Seed voxel | Hover over the target tooth in Slicer; the data probe shows IJK. Pass as `z,y,x` (slice, row, column) | `--seed 125,80,105` |
| Lesion model **or** lesion mask | Either a trained texture-SVM (`train`, §2) or an expert-drawn lesion segment exported as NIfTI (review mode) | `--lesion-model file.joblib` or `--lesion-mask file.nii.gz` |
| Clinical modifiers | From the chart / clinical exam / intra-operative findings | CLI flags (§3) |
| Policy file | Created automatically on first run; persist it per clinic | `--policy policy.json` |

The tooth segmenter separates the target tooth from neighbours (erosion → seed component → Voronoi split), and extracts the root canal as the Otsu-dark core along the PCA tooth axis. If the canal is missed on a heavily calcified tooth, draw the canal as an extra segment and replace `extract_canal` with a label (label hook is `SlicerLabelmapToothSegmenter`).

## 2. Train the lesion segmenter (once per site, retrain as cases accrue)

Create `train_manifest.json` from expert-labelled cases:

```json
[{"dicom": "cases/001/dicom", "labelmap": "cases/001/dentalsegmentator_labels.nii.gz",
  "seed": [125, 80, 105], "lesion_mask": "cases/001/expert_lesion_mask.nii.gz"}, ...]
```

```bash
python -m ecr_cds.cli train --manifest train_manifest.json --out lesion_svm.joblib --radius 5
```

Leave-one-patient-out Dice should be reported before clinical use (Ravikumar et al. reached 0.59 ± 0.19 on 6 scans; expect to need dozens of cases). For a stronger model, train nnU-Net on the same masks and wrap it in a class with the `LesionSegmenter` signature `(volume_norm, seg) -> probability map`.

**Threshold calibration.** Do not keep 0.50 blindly: compute the Youden-optimal threshold and calibration slope on held-out cases and pass it as `--threshold`. Over-segmentation at lesion edges can push the circumferential spread across a category boundary (in the demo the model-driven run measured 181° for a 150° lesion → 3Cp flagged `borderline`, while the expert-mask run gave 3Bp). The borderline flag exists precisely for this; treat it as a prompt to review the mask.

## 3. Run a case

```bash
python -m ecr_cds.cli run \
  --dicom cases/017/dicom --expected-slices 500 \
  --labelmap cases/017/dentalsegmentator_labels.nii.gz --seed 125,80,105 \
  --lesion-model lesion_svm.joblib --threshold 0.42 \
  --tooth-type molar --age 52 --buccal 0,-1,0 \
  --symptomatic yes --pulp-vital yes --restorable yes --perio yes \
  --multirooted yes --lesion-on-root DB --bleeding yes \
  --policy clinic_policy.json --out reports/017.json --case-id 017
```

Modifier flags (`yes/no`, omit if unknown): `--symptomatic --pulp-vital --restorable --perio --multirooted --lesion-on-root --accessible --bleeding --growing --aesthetic --fracture-risk`. `--buccal` is the buccal direction in voxel axes (z,y,x) so the portal-of-entry angle becomes clinically oriented (0° buccal, 180° lingual/palatal) and accessibility can be inferred; without it, set `--accessible` manually.

Console output shows the Patel code with its measurements, the ranked recommendation, propensities, every rule that fired, and QA issues. The JSON report contains acquisition provenance, QA, Patel parameters with margins, modifiers, the rule safe set, the RL decision, and the end-stage matrix when applicable.

Review mode (expert lesion mask instead of the model): replace `--lesion-model` with `--lesion-mask cases/017/expert_lesion_mask.nii.gz`.

Reinspection: if the report says `borderline=True`, correct the masks or re-slice in Slicer, re-run with the same `--out`, and pass `--reinspection-changed yes/no` in the feedback step so the conditioned reinspection reward is credited correctly.

## 4. Close the loop (RL learning)

Immediately after the clinician decides (event id `<case_id>:decision`; re-running is a no-op):

```bash
python -m ecr_cds.cli feedback --report reports/017.json --policy clinic_policy.json \
  --clinician-choice root_resection_hemisection
```

At an observed recall (event id `<case_id>:outcome:<months>`; `--months` is mandatory with an outcome and must be in (0, 240]; a lost tooth cannot be recorded as success):

```bash
python -m ecr_cds.cli feedback --report reports/017.json --policy clinic_policy.json \
  --clinician-choice root_resection_hemisection --success yes --survival yes --months 12
```

Each call computes the hybrid reward (rubric + format + conditioned reinspection), updates the per-cell Beta posterior and the per-action linear model, and appends to the policy's log with the recorded propensity.

## 5. Evaluate before promoting a policy

Keep two files: the deployed `clinic_policy.json` (used by `run`) and a candidate trained offline (for example with a different prior strength or after merging logs from several clinics).

```bash
python -m ecr_cds.cli evaluate --policy clinic_policy.json --candidate candidate_policy.json
```

The doubly-robust estimate with bootstrap CI is printed; `PROMOTE` appears only if the CI lower bound is at least the deployed policy's logged value and n ≥ 50. Copy the candidate over the deployed file only then.

## 6. Inspect the matrices

```bash
python -m ecr_cds.cli matrix               # 32 cells with alternatives, priors, evidence tag
python -m ecr_cds.cli matrix --end-stage   # end-stage management matrix
```

## 7. Python API (same features, no CLI)

```python
from ecr_cds import run_pipeline
from ecr_cds.adapters import SlicerLabelmapToothSegmenter, PretrainedLesionSegmenter
from ecr_cds.decision_matrix import ClinicalModifiers
from ecr_cds.rl_policy import ConstrainedBanditPolicy, Feedback, build_reward, ClinicalContext
from ecr_cds.patel_classifier import PatelParameters

policy = ConstrainedBanditPolicy.load("clinic_policy.json")
report = run_pipeline("cases/017/dicom",
                      SlicerLabelmapToothSegmenter("cases/017/labels.nii.gz", seed=(125, 80, 105)),
                      PretrainedLesionSegmenter(model_path="lesion_svm.joblib"),
                      modifiers=ClinicalModifiers(symptomatic=True, multirooted=True, lesion_on_root="DB"),
                      policy=policy, lesion_threshold=0.42, tooth_type="molar", age_years=52,
                      expected_slices=500)
policy.save("clinic_policy.json")
```

## 8. Operational guardrails

* Irreversible options are returned with `requires_confirmation=True`; your UI must require an explicit confirm and should log who confirmed.
* A `format` reward component of −1.0 in the feedback output means a guardrail was violated; investigate before continuing to learn.
* QA issues (voxel > 0.2 mm, anisotropy, missing slices) are printed and stored; a d/p call made on a > 0.2 mm voxel scan should be treated as provisional.
* Cells tagged `expert-extrapolation` in `decision_matrix_32.csv` (1Cd, 1Dd, 1Cp, 1Dp, 2Cd, 2Dd) have no guideline backing; have an endodontist sign off on them before go-live.
* Reports store `PatientID` and UIDs only as salted hashes; images and masks still contain anatomy, so treat exported crops as patient data.
* `status: provisional` means either the lesion model is not clinically validated (`evidence_level`) or clinical inputs are missing (`missing_inputs`); act only on `reviewed`/`ok` outputs with all inputs entered.
