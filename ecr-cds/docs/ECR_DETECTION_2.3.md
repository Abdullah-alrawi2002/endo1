# ECR candidate detection, revision 2.3

This revision implements an experimental cervical-resorption candidate screen.
It does not install a clinically trained ECR model. It supersedes the previous
claim that the scan had been classified as ECR 2Bp: that was a generic
radiolucency-to-treatment mapping without diagnostic validation.

## What changes

- **Cervical connection:** a candidate needs a surface connection in a configurable
  band around a reviewed or estimated CEJ plane. Connected extension outside that
  band is retained, rather than clipping an entire lesion to the neck of the tooth.
- **External origin:** internal dark components without a surface connection are
  screened out. The automatic canal is an imperfect proxy, never an expert mask.
- **Physical geometry:** component volume, depth into the reconstructed contour,
  and axial extent use millimetres. Each component is retained independently;
  the old largest-component post-processing is not used in this mode.
- **Stability:** lower-threshold retained fraction and upper-threshold component
  expansion are reported. These are robustness diagnostics, not ECR probabilities.
- **Restoration and anatomy flags:** nearby dense material, missing/automatic
  canal, unreviewed CEJ, obliquity and crop boundaries are reported.
- **No anatomical rewriting:** this detector does not silently add its predicted
  defect to the tooth mask to make classification succeed.
- **Abstention:** ECR diagnosis, Patel classification and treatment are withheld.
  A negative screen is not evidence that ECR is absent.

The clinical motivation is the external portal and three-dimensional spread
described by [Patel et al.](https://onlinelibrary.wiley.com/doi/10.1111/iej.12942).
The need for differential review is illustrated by a study in which suspected
ECR was instead explained by caries, root anatomy and other mimics
([CBCT differential study](https://academic.oup.com/dmfr/article/52/2/20220223/7280316?login=false)).
Neither publication validates this code or its numeric cutoffs.

## How to run

Open the attached Colab notebook in a fresh runtime, set `SCAN_PATH` or upload the
DICOM ZIP, and run the cells. The default `LESION_SOURCE = "ecr_candidate"` uses
automatic tooth selection when no seed is entered.

Optional `CEJ_ZYX` and `APEX_ZYX` are clinician-reviewed points in the original
volume's `(z, y, x)` voxel coordinates. Supply both or neither. They define an
axial CEJ plane, not a complete anatomical surface. Do not copy the unverified
automatic estimate into these fields and call it reviewed.

The scan is screened on the automatically selected tooth, not exhaustively on
every tooth. Candidate-tooth selection remains heuristic. A clinician should
review the selection or set a different seed; metal flags do not prove another
tooth is free of disease.

## Outputs

- **`auto_report.json`:** component coordinates, flags, configuration, rejections,
  and explicit null ECR/Patel/treatment fields.
- **`ecr_review_candidates.nii.gz`:** component-ID map on the original scan grid,
  not an expert lesion label.
- **`automatic_canal_proxy.nii.gz`:** machine-derived canal/filling proxy,
  not a full anatomical canal mask.
- **`auto_labels.nii.gz`:** classical teeth/bone labels, not DentalSegmentator output.
- **Executed notebook and overlay:** inspect the contour alignment and uncertainty.

## Verification and limits

Logic tests cover an engineered external cervical defect, intact tooth, internal
void, noncervical surface defect, dense-material adjacency, missing landmarks,
empty masks and blocking treatment from unconfirmed segmentation. These fixtures
do not estimate diagnostic sensitivity, specificity, Dice on patients or
clinical performance. Legacy phantom and robustness suites are also retained.

Engineering defaults include a 2 mm CEJ half-band, 0.25 mm³ minimum candidate,
0.2 mm contour depth, 0.3 mm axial extent and ±10% threshold perturbation.
They are not clinical criteria. Intensity units are scanner-relative, not HU.

Limitations include the classical tooth segmentation, approximate CEJ, slice-wise
convex contour assumption, multiple roots, severe contour loss, oblique axes,
reparative/mineralized disease and metal artifacts. The screen requires
near-isotropic input. It cannot resolve ECR versus caries or restoration defects.

For diagnostic detection, collect expert-confirmed ECR positives and hard
negatives/mimics, tooth/canal/lesion labels and diagnostic adjudication. Develop
and evaluate a lesion model with patient-level splits and an untouched external
cohort; evaluate tooth-level sensitivity, specificity and false positives as well
as segmentation. The patient scan in this conversation was not used to train a
model or establish performance.
