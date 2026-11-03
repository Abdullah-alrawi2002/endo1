"""Full-feature demo: fabricates every input a real run needs, then drives the
CLI exactly as you would on clinical data.

  1. two phantom CBCT cases -> DICOM folders (500 slices each)
  2. DentalSegmentator-style label maps (1 maxilla, 3 upper teeth, incl. a
     neighbouring tooth so tooth separation is exercised) -> NIfTI
  3. expert lesion masks -> NIfTI
  4. train the texture-SVM lesion model on case A
  5. run case B through `ecr_cds run` with the trained model + clinical modifiers
  6. record clinician feedback + 12-month outcome with `ecr_cds feedback`
  7. off-policy evaluate with `ecr_cds evaluate`
"""
import os, sys, json, subprocess, shutil
import numpy as np
import nibabel as nib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from ecr_cds.phantom import make_phantom, write_dicom_series, VOX, SHAPE

WORK = os.path.join(ROOT, "examples", "demo_data")
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)


def save_nifti(arr_zyx, path):
    aff = np.diag([VOX, VOX, VOX, 1.0])
    nib.save(nib.Nifti1Image(np.transpose(arr_zyx, (2, 1, 0)).astype(np.uint8), aff), path)


def build_case(name, h, s, perf):
    vol, seg, lesion, truth = make_phantom(h, s, perf)
    # add a neighbouring tooth touching the target (tests tooth separation)
    Z, Y, X = SHAPE
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    r2 = np.hypot(yy - Y / 2, xx - (X / 2 + 88)) * VOX
    z_mm = zz * VOX
    nb = (r2 <= 4.0) & (z_mm >= 4) & (z_mm <= 30) & ~seg.tooth
    vol[nb] = 1600
    labels = np.zeros(SHAPE, np.uint8)
    labels[seg.bone] = 1                 # upper skull / maxilla
    labels[seg.tooth | nb] = 3           # upper teeth (single class, as DentalSegmentator outputs)
    d = os.path.join(WORK, name); os.makedirs(d)
    os.makedirs(os.path.join(d, "dicom")); write_dicom_series(vol, os.path.join(d, "dicom"))
    save_nifti(labels, os.path.join(d, "dentalsegmentator_labels.nii.gz"))
    save_nifti(lesion, os.path.join(d, "expert_lesion_mask.nii.gz"))
    seed = [int(12.5 / VOX), SHAPE[1] // 2, SHAPE[2] // 2 + 25]   # inside the target tooth, mid-crown dentine
    json.dump({"truth": truth, "seed": seed}, open(os.path.join(d, "meta.json"), "w"))
    return d, seed


def sh(args):
    print("\n$", " ".join(args)); subprocess.run(args, check=True, cwd=ROOT)


caseA, seedA = build_case("caseA_2Bp", 3.0, 150, True)
caseB, seedB = build_case("caseB_3Bp", 9.0, 150, True)

# 4. train lesion model on case A
manifest = [{"dicom": f"{caseA}/dicom", "labelmap": f"{caseA}/dentalsegmentator_labels.nii.gz",
             "seed": seedA, "lesion_mask": f"{caseA}/expert_lesion_mask.nii.gz"}]
json.dump(manifest, open(f"{WORK}/train_manifest.json", "w"))
sh([sys.executable, "-m", "ecr_cds.cli", "train", "--manifest", f"{WORK}/train_manifest.json",
    "--out", f"{WORK}/lesion_svm.joblib"])

# 5. run case B with the trained model and clinical modifiers (RJME case 2-like: #17 DB root)
policy = f"{WORK}/policy.json"; report = f"{WORK}/caseB_report.json"
sh([sys.executable, "-m", "ecr_cds.cli", "run", "--dicom", f"{caseB}/dicom",
    "--ecr-confirmed",  # synthetic known-positive fixture, NOT clinical confirmation
    "--labelmap", f"{caseB}/dentalsegmentator_labels.nii.gz", "--seed", ",".join(map(str, seedB)),
    "--lesion-model", f"{WORK}/lesion_svm.joblib", "--threshold", "0.5",
    "--expected-slices", "500", "--tooth-type", "molar", "--age", "52",
    "--symptomatic", "yes", "--pulp-vital", "yes", "--multirooted", "yes", "--lesion-on-root", "DB",
    "--bleeding", "yes", "--policy", policy, "--out", report, "--case-id", "caseB"])

# 5b. same case in review mode with the expert mask instead of the model
sh([sys.executable, "-m", "ecr_cds.cli", "run", "--dicom", f"{caseB}/dicom",
    "--ecr-confirmed",  # synthetic known-positive fixture
    "--labelmap", f"{caseB}/dentalsegmentator_labels.nii.gz", "--seed", ",".join(map(str, seedB)),
    "--lesion-mask", f"{caseB}/expert_lesion_mask.nii.gz", "--tooth-type", "molar", "--age", "52",
    "--symptomatic", "no", "--pulp-vital", "yes", "--accessible", "no",
    "--policy", policy, "--out", f"{WORK}/caseB_review_report.json"])

# 6. clinician feedback: accepted hemisection; 12-month outcome success
sh([sys.executable, "-m", "ecr_cds.cli", "feedback", "--report", report, "--policy", policy,
    "--clinician-choice", "root_resection_hemisection", "--success", "yes", "--survival", "yes", "--months", "12"])

# 7. off-policy evaluation of the (updated) policy against its own log
sh([sys.executable, "-m", "ecr_cds.cli", "evaluate", "--policy", policy])
print("\nDemo artefacts in", WORK)
