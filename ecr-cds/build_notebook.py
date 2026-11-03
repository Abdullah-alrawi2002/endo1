"""Generate ECR_CDS_Colab.ipynb (revision 2.1 - addresses review report 2.0.0).

The package (code + phantom-trained model + training card) is embedded in the
notebook as base64, so the notebook is standalone; an external ecr_cds.zip is
optional and overrides the embedded copy.
"""
import base64
import io
import os
import zipfile

import nbformat as nbf

ROOT = os.path.dirname(os.path.abspath(__file__))


def package_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("ecr_cds", "tests", "docs"):
            for root, _, files in os.walk(os.path.join(ROOT, sub)):
                if "__pycache__" in root:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    zf.write(fp, os.path.relpath(fp, os.path.dirname(ROOT)))
        zf.write(os.path.join(ROOT, "README.md"), "ecr_cds/README.md")
        split_dir = os.path.join(ROOT, "finetune_data", "split")   # shipped SFT data (synthetic, compresses well)
        if os.path.isdir(split_dir):
            for f in sorted(os.listdir(split_dir)):
                zf.write(os.path.join(split_dir, f), f"ecr_cds/finetune_data/split/{f}")
    return buf.getvalue()


PKG_B64 = base64.b64encode(package_bytes()).decode()

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "colab": {"name": "ECR_CDS_Colab.ipynb", "provenance": []}}
C = []

C.append(nbf.v4.new_markdown_cell("""# ECR-CDS: cervical resorption candidate screening (revision 2.3)

Revision 2.3 supersedes the earlier automatic 2Bp interpretation. Default output is a
non-diagnostic ECR candidate screen: external cervical connection, physical depth,
3D extent, threshold stability and anatomy/restoration flags. It retains separate
components, exports review masks, and withholds Patel classification and treatment.
The bundled model has not been replaced with a clinically trained ECR model.

Standalone notebook for the `ecr_cds` package: patient CBCT (DICOM folder or ZIP) → geometry-validated volume → optional tooth/canal/lesion masks → Patel 3D code (height 1–4, spread A–D, proximity d/p) → 32-cell decision matrix → constrained-bandit recommendation → clinician review → explicit feedback events → audited policy evaluation.

**What changed after the review (4 Oct 2026):** the package, phantom model and training card are embedded (no external ZIP needed); the patient scan is the default input and phantoms are opt-in; the lesion overlay tolerates empty masks; clinical inputs default to *unknown* and must be entered; feedback is off by default, requires observed outcomes with validated follow-up and uses idempotent event IDs; thresholds are selected inside patient-grouped folds; policy evaluation uses held-out cases and never auto-promotes; DICOM/mask geometry is checked (physical slice sorting, gaps, duplicates, orientation/spacing changes, drift, multiframe, rescale, padding, RAS→LPS, shape/spacing/shift mismatch); reports omit identifiers.

**Bundled model status:** `lesion_svm_phantom.joblib` was trained on five synthetic phantoms only. Its output is labelled `research_phantom_model` and cannot establish ECR. Classification is withheld without explicit clinical confirmation.

Revision 2.2 adds `ecr_cds.auto_segment`: classical teeth/bone segmentation, system-proposed seed (radiolucency evidence, else FOV centre), root-filling-aware canal, and a model-free radiolucent-defect detector, all labelled as automatic/rule-based evidence. The pipeline now crops to the target tooth so 500-slice volumes run within a few GB.\n\nSections: 1 Setup · 2 Load scan + QA · 3 Target tooth and masks · 4 Automatic measurement · 5 Manual Patel entry + clinical inputs · 6 Training (off) · 7 Feedback (off) · 8 Policy audit · 9 Export"""))

C.append(nbf.v4.new_markdown_cell("## 1. Setup"))
C.append(nbf.v4.new_code_cell(f'''import os, sys, subprocess, importlib, base64, io, zipfile, json
def _ensure(pkgs):
    missing = [p for p, m in pkgs if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
_ensure([("numpy", "numpy"), ("scipy", "scipy"), ("pydicom", "pydicom"), ("scikit-learn", "sklearn"),
         ("nibabel", "nibabel"), ("joblib", "joblib"), ("matplotlib", "matplotlib")])

EXTERNAL_ZIP = None        # optional: path to a newer ecr_cds.zip; None = use the embedded package
_EMBEDDED_PKG = "{PKG_B64}"
if not os.path.exists("ecr_cds/ecr_cds"):
    data = open(EXTERNAL_ZIP, "rb").read() if EXTERNAL_ZIP else base64.b64decode(_EMBEDDED_PKG)
    zipfile.ZipFile(io.BytesIO(data)).extractall(".")
PKG = os.path.abspath("ecr_cds"); sys.path.insert(0, PKG); os.chdir(PKG)
import numpy as np, ecr_cds
from ecr_cds.dicom_loader import load_dicom_series, DicomGeometryError
print("ecr_cds ready at", PKG, "| source:", "external zip" if EXTERNAL_ZIP else "embedded")'''))

C.append(nbf.v4.new_code_cell("""# Optional self-tests (~2 min): phantom pipeline + robustness suite
RUN_SELF_TESTS = False
if RUN_SELF_TESTS:
    subprocess.run([sys.executable, "tests/test_pipeline.py"], check=True)
    subprocess.run([sys.executable, "tests/test_robustness.py"], check=True)
    subprocess.run([sys.executable, "tests/test_ecr_detection.py"], check=True)"""))

C.append(nbf.v4.new_markdown_cell("""## 2. Load the patient scan and check geometry

Set `SCAN_PATH` to a DICOM folder or a ZIP (macOS metadata and non-DICOM files are skipped; unsafe archive paths are rejected). If `SCAN_PATH` is `None` in Colab you will be prompted to upload. Set `USE_DEMO_PHANTOM = True` only to exercise the workflow on a synthetic scan — the output is then labelled as phantom data."""))
C.append(nbf.v4.new_code_cell("""SCAN_PATH = None            # e.g. "/content/drive/MyDrive/ecr/scan.zip" or "/content/scan_dicom/"
USE_DEMO_PHANTOM = False    # True -> synthetic 500-slice phantom written as a DICOM ZIP (research demo only)
OUT_DIR = os.path.abspath("outputs_case"); os.makedirs(OUT_DIR, exist_ok=True)

if USE_DEMO_PHANTOM and SCAN_PATH is None:
    from ecr_cds.phantom import make_phantom, write_dicom_series
    import tempfile, shutil
    _vol, _seg, _les, _truth = make_phantom(9.0, 150, True)
    _d = tempfile.mkdtemp(); write_dicom_series(_vol, _d)
    SCAN_PATH = os.path.join(OUT_DIR, "phantom_scan.zip")
    with zipfile.ZipFile(SCAN_PATH, "w") as zf:
        for n in sorted(os.listdir(_d)): zf.write(os.path.join(_d, n), f"scan/{n}")
    shutil.rmtree(_d); print("Phantom scan written:", SCAN_PATH, "| truth:", _truth)
elif SCAN_PATH is None:
    try:
        from google.colab import files
        up = files.upload(); SCAN_PATH = next(iter(up))
    except ImportError:
        raise SystemExit("Set SCAN_PATH to a DICOM folder or ZIP")

try:
    vol = load_dicom_series(SCAN_PATH)
except DicomGeometryError as e:
    raise SystemExit(f"Scan rejected: {e}")
m = vol.metadata
qa_rows = [("Image volume z x y x x", "%d x %d x %d voxels" % vol.shape),
           ("Spacing z/y/x (mm)", "%.4g / %.4g / %.4g" % vol.spacing),
           ("Field of view (mm)", "%.2f x %.2f x %.2f" % tuple(m["FieldOfView_mm"])),
           ("Slices / max position residual", f"{m['NumberOfSlices']} / {m['MaxSlicePositionResidual_mm']:.2e} mm"),
           ("Isotropic", m["Isotropic"]), ("Intensity range after rescale", "%.0f to %.0f" % tuple(m["IntensityRange"])),
           ("Rescale slope / intercept", f"{m['RescaleSlope']} / {m['RescaleIntercept']}"),
           ("Padding voxels", m["PaddingVoxels"]), ("Archive housekeeping", m["Housekeeping"]),
           ("Series available", m["SeriesAvailable"]), ("Case id (hashed series)", m["SeriesInstanceUID_hash"])]
for k, v in qa_rows: print(f"{k:34s} {v}")
if max(vol.spacing) > 0.2: print("WARNING: voxel > 0.2 mm - d/p proximity calls are unreliable at this resolution")
print("Identifiers are stored only as salted hashes; images still contain anatomy.")"""))

C.append(nbf.v4.new_code_cell("""# Three-plane preview (display/coordinate check only - not a diagnosis)
import matplotlib.pyplot as plt
def show_planes(volume, zyx=None, masks=None, title=""):
    z, y, x = zyx if zyx is not None else [s // 2 for s in volume.shape]
    arr = volume.array; lo, hi = np.percentile(arr, [1, 99])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    views = [(arr[z], "axial z=%d" % z, lambda mk: mk[z]), (arr[:, y, :], "coronal y=%d" % y, lambda mk: mk[:, y, :]),
             (arr[:, :, x], "sagittal x=%d" % x, lambda mk: mk[:, :, x])]
    for a, (im, t, sl) in zip(ax, views):
        a.imshow(im, cmap="gray", vmin=lo, vmax=hi, aspect="auto"); a.set_title(t)
        for name, (mk, col) in (masks or {}).items():
            if mk is not None and sl(mk).any(): a.contour(sl(mk), colors=col, linewidths=0.8)
    if zyx is not None:
        ax[0].plot(x, y, "r+", ms=12); ax[1].plot(x, z, "r+", ms=12); ax[2].plot(y, z, "r+", ms=12)
    fig.suptitle(title); plt.show()
show_planes(vol, title="Scan preview (centre slices)")"""))

C.append(nbf.v4.new_markdown_cell("""## 3. Target tooth and masks

Three ways to define the target tooth, in order of preference:

1. **DentalSegmentator label map** (`LABELMAP`) exported on this exact volume plus a clinician **seed** `(z, y, x)`; optional expert lesion / canal masks. All masks are checked for shape, spacing, orientation (RAS→LPS) and origin.
2. **Seed only** with `AUTO_SEGMENT = True`: classical intensity/morphology segmentation of teeth and bone (`ecr_cds.auto_segment`), labelled `auto_classical`.
3. **Nothing** with `AUTO_SEGMENT = True`: the system also proposes the seed — the tooth with the largest external radiolucent defect if any exceeds 5 mm³ (metal-flagged teeth excluded), otherwise the tooth nearest the field-of-view centre (small-FOV CBCT is centred on the tooth of interest). The proposal is reported as `seed_source = auto_radiolucency | auto_fov_centre` and **must be confirmed on the overlay**.

A root filling is detected automatically as hyperdense central material and used as the canal (`canal_mode = hyperdense_filling`); otherwise the dark pulp is used. An expert canal mask (`CANAL_MASK`) overrides both. The automatic label map is written to NIfTI so it can be reviewed or replaced in 3D Slicer."""))
C.append(nbf.v4.new_code_cell("""SEED_ZYX   = None      # e.g. (125, 80, 105) - IJK from the Slicer data probe; None = let the system propose it
LABELMAP   = None      # DentalSegmentator label map .nii.gz (1 upper skull, 2 mandible, 3 upper teeth, 4 lower teeth)
TEETH_LABELS, BONE_LABELS = (3, 4), (1, 2)
LESION_MASK = None     # expert lesion mask .nii.gz (optional, review mode)
CANAL_MASK  = None     # expert canal mask .nii.gz (optional)
AUTO_SEGMENT = True    # classical automatic teeth/bone segmentation when no label map is given

if USE_DEMO_PHANTOM and SEED_ZYX is None and LABELMAP is None:
    # phantom: write the matching label map + expert masks on the same grid (research demo only)
    import nibabel as nib
    from ecr_cds.phantom import VOX
    def _save(arr, path):
        aff = np.diag([-VOX, -VOX, VOX, 1.0]); aff[:3, 3] = [-vol.origin[0], -vol.origin[1], vol.origin[2]]
        nib.save(nib.Nifti1Image(np.transpose(arr.astype(np.uint8), (2, 1, 0)), aff), path)
    lab = np.zeros(vol.shape, np.uint8); lab[_seg.bone] = 1; lab[_seg.tooth] = 3
    LABELMAP = os.path.join(OUT_DIR, "phantom_labels.nii.gz"); _save(lab, LABELMAP)
    LESION_MASK = os.path.join(OUT_DIR, "phantom_expert_lesion.nii.gz"); _save(_les, LESION_MASK)
    SEED_ZYX = (125, vol.shape[1] // 2, vol.shape[2] // 2 + 25)

from ecr_cds.adapters import load_labelmap, MaskGeometryError, SlicerLabelmapToothSegmenter
from ecr_cds.auto_segment import (auto_segment, AutoToothSegmenter, score_candidates_radiolucency, choose_seed,
                                  save_labelmap_nifti, RadiolucencyLesionDetector, soft_tissue_level)
masks, auto, seed_source, tooth_scores = {}, None, "clinician", None
if LABELMAP:
    try:
        labels = load_labelmap(LABELMAP, volume=vol)
        masks["teeth"] = (np.isin(labels, TEETH_LABELS), "cyan"); masks["bone"] = (np.isin(labels, BONE_LABELS), "green")
        seg_method = "dentalsegmentator_labelmap"
    except MaskGeometryError as e:
        raise SystemExit(f"Label map rejected: {e}")
elif AUTO_SEGMENT:
    import time; t0 = time.time()
    auto = auto_segment(vol.array, vol.spacing)
    print(f"auto_segment: {len(auto.candidates)} tooth candidates in {time.time() - t0:.0f} s;", auto.notes[0])
    tooth_scores = score_candidates_radiolucency(auto, vol.array, vol.spacing)
    print(f"{'rank':4s} {'seed (z,y,x)':18s} {'FOV dist mm':>11s} {'defect mm3':>10s}  canal mode          note")
    for sc in tooth_scores:
        print(f"{sc['rank_fov']:<4d} {str(tuple(sc['seed_zyx'])):18s} {sc['dist_to_fov_centre_mm']:11.1f} {sc['radiolucent_defect_mm3']:10.1f}  {sc['canal_mode']:19s} {sc['note']}")
    if SEED_ZYX is None:
        SEED_ZYX, seed_source = choose_seed(tooth_scores)
        print("System-proposed seed:", SEED_ZYX, "| basis:", seed_source, "| confirm on the overlay below")
    masks["teeth"] = (auto.labels == 3, "cyan"); masks["bone"] = (auto.labels == 1, "green")
    LABELMAP_AUTO = os.path.join(OUT_DIR, "auto_labels.nii.gz"); save_labelmap_nifti(auto.labels, vol, LABELMAP_AUTO)
    seg_method = "auto_classical"
    print("Automatic label map written for review in Slicer:", LABELMAP_AUTO)
else:
    seg_method = None
if LESION_MASK:
    try: masks["lesion"] = (load_labelmap(LESION_MASK, volume=vol) > 0, "red")
    except MaskGeometryError as e: raise SystemExit(f"Lesion mask rejected: {e}")
if SEED_ZYX is not None:
    show_planes(vol, SEED_ZYX, masks, title=f"Seed (+) [{seed_source}] and {seg_method} masks - inspect alignment before proceeding")
    if "teeth" in masks and not masks["teeth"][0][tuple(SEED_ZYX)]:
        print("WARNING: seed is not inside a teeth label - adjust SEED_ZYX")
else:
    print("No seed and no segmentation: automatic measurement will be skipped (status insufficient_evidence).")"""))

C.append(nbf.v4.new_markdown_cell("""## 4. Automatic measurement

`LESION_SOURCE` selects how the lesion is obtained:

* `"ecr_candidate"` (default): experimental cervical-resorption candidate screen.
  Looks for an external connection near an estimated or reviewed CEJ plane,
  depth into a reconstructed tooth contour, 3D extent and stability across three
  intensity thresholds. No diagnosis, Patel code or treatment is produced.
  Numeric cutoffs are configurable engineering defaults, not validated clinical criteria.
* `"expert_mask"` — clinician-drawn mask (evidence `expert_mask`); automatic when `LESION_MASK` is set.
* `"radiolucency_rule"` — model-free detector: hypodense defect inside the closed tooth hull, outside the canal, reaching the external surface (evidence `rule_based_radiolucency`). Does not distinguish ECR from caries, abrasion or artefact; sub-threshold components are listed for review.
* `"model"` — the texture-SVM in `LESION_MODEL`. The bundled model is **phantom-only** (`research_phantom_model`); on real scans it mislabels dentine wholesale, which the report flags.

Automatic masks are not DentalSegmentator or expert masks. CEJ and apex points
can be reviewed in the viewer and entered in full-volume (z,y,x) coordinates.
Unknown landmarks are flagged, never silently promoted to expert evidence.
Unconfirmed ECR blocks classification and treatment even with another lesion source.

ECR can start near the cervical attachment and extend coronally/apically; caries
and anatomical concavities can mimic it ([Patel et al.](https://onlinelibrary.wiley.com/doi/10.1111/iej.12942);
[CBCT differential study](https://academic.oup.com/dmfr/article/52/2/20220223/7280316?login=false)).
These publications motivate review features, not the numerical settings or performance of this implementation."""))
C.append(nbf.v4.new_code_cell("""from ecr_cds import run_pipeline
from ecr_cds.adapters import PretrainedLesionSegmenter
from ecr_cds.decision_matrix import ClinicalModifiers
from ecr_cds.rl_policy import ConstrainedBanditPolicy
from ecr_cds.pipeline import to_json
from ecr_cds.training import load_model
from ecr_cds.ecr_detection import CervicalResorptionDetector
from ecr_cds.preprocess import crop_to_bbox

LESION_SOURCE = "ecr_candidate"   # ecr_candidate | expert_mask | radiolucency_rule | model
CEJ_ZYX = None                    # reviewed full-volume voxel point, not invented
APEX_ZYX = None                   # supply both reviewed points or neither
ECR_CONFIRMED = False             # only a clinician can supply the ECR diagnosis
LESION_MODEL = "ecr_cds/models/lesion_svm_phantom.joblib"   # only used when LESION_SOURCE == "model"
POLICY_PATH = os.path.join(OUT_DIR, "policy.json")
policy = ConstrainedBanditPolicy.load(POLICY_PATH) if os.path.exists(POLICY_PATH) else ConstrainedBanditPolicy()
CASE_ID = vol.metadata["SeriesInstanceUID_hash"]
auto_report = {"status": "insufficient_evidence", "reason": "seed and/or segmentation not available"}
if LESION_MASK: LESION_SOURCE = "expert_mask"

if SEED_ZYX is not None and (LABELMAP or auto is not None):
    if LABELMAP:
        tooth_seg = SlicerLabelmapToothSegmenter(LABELMAP, tuple(SEED_ZYX), TEETH_LABELS, BONE_LABELS); tooth_seg.canal_mask_path = CANAL_MASK
    else:
        tooth_seg = AutoToothSegmenter(auto, tuple(SEED_ZYX))
        if CANAL_MASK: tooth_seg.canal_mask = load_labelmap(CANAL_MASK, volume=vol) > 0
    card, detector = None, None
    if LESION_SOURCE == "ecr_candidate":
        preview_seg = tooth_seg(vol.array, vol.spacing)
        crop = crop_to_bbox(preview_seg.tooth, margin=int(round(3.0/min(vol.spacing))))
        offset = np.array([s.start for s in crop])
        threshold = auto.t_tooth if auto is not None else float(np.percentile(vol.array[preview_seg.tooth], 20))
        dense = auto.t_dense if auto is not None else float(np.percentile(vol.array[preview_seg.tooth], 95))
        detector = CervicalResorptionDetector(threshold, dense, vol.spacing,
            cej_zyx=None if CEJ_ZYX is None else tuple(np.asarray(CEJ_ZYX)-offset),
            apex_zyx=None if APEX_ZYX is None else tuple(np.asarray(APEX_ZYX)-offset),
            canal_source="expert_mask" if CANAL_MASK else getattr(tooth_seg, "canal_mode", "automatic"))
        lesion_seg, thr = detector, .5
    elif LESION_SOURCE == "expert_mask":
        lesion_seg, thr = PretrainedLesionSegmenter(mask_path=LESION_MASK), 0.5
    elif LESION_SOURCE == "radiolucency_rule":
        t_defect = auto.t_tooth if auto is not None else float(np.percentile(vol.array[vol.array > vol.array.min()], 88))
        t_soft = auto.t_hard if auto is not None else t_defect * 0.75
        detector = RadiolucencyLesionDetector(t_defect, soft_tissue_level(vol.array, t_soft))
        lesion_seg, thr = (lambda norm, seg: detector(norm, seg, vol.spacing)), 0.5
    else:
        cp = LESION_MODEL.replace(".joblib", ".card.json"); card = json.load(open(cp)) if os.path.exists(cp) else None
        try:
            model = load_model(LESION_MODEL)
        except Exception as e:   # scikit-learn pickle mismatch -> rebuild from the recorded phantom recipe
            print("Rebuilding bundled model from its training card:", e)
            from ecr_cds.phantom import make_phantom; from ecr_cds.preprocess import preprocess_volume; from ecr_cds.training import train_from_cases
            V, S, L = [], [], []
            for c in card["cases"]:
                v_, s_, l_, _ = make_phantom(c["height_mm"], c["spread_deg"], c["perforate"]); V.append(preprocess_volume(v_, roi=s_.tooth)); S.append(s_); L.append(l_)
            model = train_from_cases(V, S, L, threshold=card["youden_threshold"])
        lesion_seg, thr = (lambda norm, seg: model(norm, seg)), model.threshold
    auto_report = run_pipeline(None, tooth_seg, lesion_seg, modifiers=ClinicalModifiers(), policy=policy, volume=vol,
                               lesion_threshold=thr, case_id=CASE_ID, model_card=card, lesion_source=LESION_SOURCE,
                               ecr_confirmed=ECR_CONFIRMED)
    auto_report["segmentation"] = {"method": seg_method, "seed_zyx": list(SEED_ZYX), "seed_source": seed_source,
                                   "canal_mode": getattr(tooth_seg, "canal_mode", "expert_mask" if CANAL_MASK else "dark_pulp_otsu"),
                                   "tooth_candidates": tooth_scores}
    if isinstance(detector, RadiolucencyLesionDetector):
        auto_report["radiolucency_components"] = detector.last_components
    if isinstance(detector, CervicalResorptionDetector):
        full_candidates = np.zeros(vol.shape, np.uint16)
        full_candidates[crop] = detector.last_labels
        save_labelmap_nifti(full_candidates, vol, os.path.join(OUT_DIR, "ecr_review_candidates.nii.gz"))
        full_canal = np.zeros(vol.shape, np.uint8)
        full_canal[crop] = preview_seg.canal[crop]
        save_labelmap_nifti(full_canal, vol, os.path.join(OUT_DIR, "automatic_canal_proxy.nii.gz"))
        for c in auto_report["candidates"]:
            print(c["component_id"], c["disposition"], c["volume_mm3"], "mm3 at", c["centroid_volume_zyx"],
                  "| flags:", c["flags"], "| rejection:", c["rejection_reasons"])
    to_json(auto_report, os.path.join(OUT_DIR, "auto_report.json"))

print("STATUS:", auto_report["status"].upper(), "| evidence:", auto_report.get("evidence_level"))
if auto_report["status"] in ("ok", "provisional"):
    p, d = auto_report["patel"], auto_report["decision"]
    print(auto_report["evidence_note"])
    print(f"Patel {p['code']} | apical {p['apical_extent_mm']:.1f} mm | spread {p['spread_deg']:.0f} deg | canal distance {p['min_canal_distance_mm']:.2f} mm | volume {p['lesion_volume_mm3']:.0f} mm3 | borderline={p['borderline']}")
    print("Provisional ranking:", " > ".join(d["ranking"]))
    print("Missing clinical inputs:", auto_report["rule_recommendation"]["missing_inputs"])
    for r in d["rationale"]: print("  -", r)
    if USE_DEMO_PHANTOM: print("Phantom truth:", _truth)
else:
    print(auto_report.get("reason") or auto_report.get("note"))
if auto_report.get("radiolucency_components"):
    off = auto_report.get("roi_bbox_zyx", [[0], [0], [0]])
    print("Raw radiolucent components before post-processing (volume zyx):")
    for c in auto_report["radiolucency_components"]:
        zyx = [c["centroid_crop_zyx"][i] + off[i][0] for i in range(3)]
        print(f"  {c['mm3']:.1f} mm3 at {tuple(zyx)} | surface contact {c['surface_contact']:.2f} | inside tooth mask {c['fraction_inside_tooth_mask']:.2f}")"""))

C.append(nbf.v4.new_code_cell("""# Overlay of the segmentation actually used (handles empty masks)
if auto_report["status"] in ("ok", "provisional", "no_lesion_detected", "ecr_not_confirmed", "candidates_for_review", "no_candidate_not_exclusion") and SEED_ZYX is not None and (LABELMAP or auto is not None):
    from ecr_cds.preprocess import preprocess_volume, crop_to_bbox
    from ecr_cds.segmentation import lesion_mask_from_probability, ToothSegmentation
    seg = tooth_seg(vol.array, vol.spacing)
    sl = crop_to_bbox(seg.tooth, margin=int(round(3.0 / min(vol.spacing))))   # same crop as run_pipeline
    seg_c = ToothSegmentation(seg.tooth[sl], seg.canal[sl], seg.bone[sl], intensity=vol.array[sl].astype(np.float32))
    norm = preprocess_volume(vol.array[sl], roi=seg_c.tooth)
    prob_c = lesion_seg(norm, seg_c)
    if prob_c.shape == vol.shape: prob_c = prob_c[sl]
    lesion = np.zeros(vol.shape, bool)
    lesion[sl] = (prob_c >= .5) if LESION_SOURCE == "ecr_candidate" else lesion_mask_from_probability(prob_c, seg_c, threshold=thr)
    if lesion.any():
        centre = [int(v) for v in np.argwhere(lesion).mean(0)]; t = "Research only: tooth (cyan), canal proxy (yellow), candidates (red)"
    else:
        centre = list(SEED_ZYX); t = "No review candidate selected; this does NOT exclude ECR"
    show_planes(vol, centre, {"tooth": (seg.tooth, "cyan"), "canal": (seg.canal, "yellow"), "lesion": (lesion if lesion.any() else None, "red")}, title=t)"""))

C.append(nbf.v4.new_markdown_cell("""## 4b. Multi-agent case conference (suggested diagnosis and plan)

Eight agents share a blackboard and leave a complete evidence trail:

| agent | reads | writes |
|---|---|---|
| imaging QA | geometry, intensities | fitness for ECR assessment |
| anatomy | masks, seed | tooth/canal/landmark reliability |
| lesion screening | crop | candidate defects + 25 quantitative visual descriptors + tiles |
| radiologist | descriptors (+ tiles with a vision model) | differential diagnosis per candidate |
| skeptic | everything above | adversarial challenges (artefact, landmarks, mimics, size) |
| Patel classifier | accepted candidate | suggested 3D code with margins |
| treatment planner | code + clinical inputs | matrix + constrained RL ranking + contingencies |
| moderator | the debate | consensus, actions required |
| context (rev 2.7) | Diagnocat findings + patient chart | prior shifts, input hints - never findings |
| verifier (rev 2.7) | each candidate, perception toolkit | re-delineation under other processing, multiplanar evidence frame, challenges |
| coverage (rev 2.7) | everything | examined vs complaint teeth, missing prerequisites |
| knowledge (rev 2.7) | answered questions, plan, gaps | ESE / Patel / AAE citations, CDT codes |

Rev 2.7 separates **seeing** (`ecr_cds.perception`: segment, propose, delineate, measure, reformat, re-verify; every call
logged) from **reasoning** (the agents). Context and knowledge interpret established findings; they cannot create one.

The radiologist answers the clinician's **seven diagnostic questions** from the scan (external origin, independence
from the canal, growth pattern, preserved pericanalar dentine, resorption-vs-caries morphology, multiplanar persistence,
extent) and a clinical-correlation agent answers **question 8** from the findings you enter in `CLINICAL_FINDINGS`.
Internal resorption is part of the differential. The radiologist and skeptic debate for up to `CONFERENCE_ROUNDS` rounds. An optional language/vision model
(`LLM_BACKEND`) adds a *second opinion* that is logged next to the rule result and can lower but never raise the
confidence band. Posteriors come from expert-set criterion weights (Patel 2018, ESE 2023, Mavridou 2016, DMFR 2023)
and are **not calibrated**; the output is a suggestion for the treating clinician, not a diagnosis."""))
C.append(nbf.v4.new_markdown_cell("""### 4b-local. Local vision model: NVIDIA Nemotron 3.5 Bolt 74B-A9B (rev 2.8)

Instead of a hosted LLM, the radiologist's **second opinion** can come from a model running on your own GPU, so no
image or chart text leaves the machine. Nemotron 3.5 Bolt (74 B total / 9 B active, 2-bit 24 GB GGUF) runs under
llama.cpp, Ollama or vLLM and is reached through an OpenAI-compatible endpoint (`NEMOTRON_BASE_URL`).

Guards (see `docs/NEMOTRON_LOCAL.md`): one provenance-stamped evidence frame per request; five samples at the
model-card temperature with **discrete-semantic-entropy abstention** (disagreeing samples -> weight 0, rule posterior
kept); a **counterfactual** candidate-filled frame that caps the weight at 0.2 when the model's answer ignores the
image; hard cap 0.5. The model never detects, measures or stages, and it has no medical evaluation in its model card.

The cell below can optionally install llama.cpp and download the 24 GB build on an A100 Colab runtime
(`RUN_NEMOTRON_SERVER = True`; ~25 GB download, several minutes). The default leaves it off and expects an already
running server, or falls back to the rev 2.7 behaviour when none is reachable."""))
C.append(nbf.v4.new_code_cell("""import os, subprocess, time, urllib.request
LLM_BACKEND = "auto"          # "auto" | "none" | "nemotron" | "openai" | "perplexity"   ("auto" picks nemotron when NEMOTRON_BASE_URL is set)
NEMOTRON_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", "http://localhost:8080/v1")
N_SAMPLES = 5                 # draws per question for the entropy estimate (1 = single greedy answer, no abstention)
DSE_THRESHOLD = 0.3           # abstain at/above this discrete semantic entropy (Kim et al. 2026 protocol value)
ENABLE_THINKING = False       # chat-template reasoning mode (slower)
VERIFY_MODE = "smooth"        # "smooth" (frozen rev 2.7 baseline) | "tta" (gain / noise / threshold / smoothing paths)
COUNTERFACTUAL_CHECK = True   # candidate-filled frame re-query (only used when a vision backend is active)
SHARPEN_DISPLAY = False       # unsharp-mask the evidence frames (display only, recorded in provenance)

RUN_NEMOTRON_SERVER = False   # True: install llama.cpp + download the 24 GB GGUF and start llama-server in the background (A100 runtime)
if RUN_NEMOTRON_SERVER:
    if not os.path.exists("llama.cpp/build/bin/llama-server"):
        subprocess.run("git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DGGML_CUDA=ON "
                       "&& cmake --build build --config Release -j -t llama-server", shell=True, check=True)
    subprocess.run("pip -q install -U huggingface_hub && hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF --local-dir bolt",
                   shell=True, check=True)
    import glob
    model = sorted(glob.glob("bolt/*.gguf"), key=os.path.getsize)[-1]
    mmproj = [f for f in glob.glob("bolt/*.gguf") if "mmproj" in f.lower()]
    cmd = ["llama.cpp/build/bin/llama-server", "--model", model, "--jinja", "--flash-attn", "on", "--temp", "1.0", "--top-p", "0.95",
           "-ngl", "999", "-c", "32768", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--port", "8080"] + (["--mmproj", mmproj[0]] if mmproj else [])
    server = subprocess.Popen(cmd, stdout=open("llama_server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8080/v1/models", timeout=2); break
        except Exception:
            time.sleep(5)
    NEMOTRON_BASE_URL = "http://localhost:8080/v1"
os.environ["NEMOTRON_BASE_URL"] = NEMOTRON_BASE_URL

def make_backend():
    from ecr_cds.agents import auto_backend
    from ecr_cds.agents.llm_backend import NemotronLocalBackend
    if LLM_BACKEND in ("nemotron", "auto"):
        try:
            return NemotronLocalBackend(base_url=NEMOTRON_BASE_URL, n_samples=N_SAMPLES, dse_threshold=DSE_THRESHOLD, enable_thinking=ENABLE_THINKING)
        except RuntimeError as e:
            print("Nemotron server not reachable ->", str(e)[:120])
            if LLM_BACKEND == "nemotron":
                raise
    return auto_backend(LLM_BACKEND)
print("backend selection:", LLM_BACKEND, "| server:", NEMOTRON_BASE_URL, "| verify:", VERIFY_MODE, "| counterfactual:", COUNTERFACTUAL_CHECK)"""))
C.append(nbf.v4.new_code_cell("""from ecr_cds.agents import CaseConference, ClinicalFindings, auto_backend, consensus_markdown
from IPython.display import Markdown, Image, display

RUN_CONFERENCE = True
CONFERENCE_ROUNDS = 2
# Question 8 - clinical findings (enter only what was examined; None = not assessed). Nothing here is inferred from the image.
CLINICAL_FINDINGS = ClinicalFindings(pink_spot=None, cervical_defect_hard_irregular=None, bleeding_on_probing_defect=None,
                                     pulp_sensibility_normal=None, symptomatic=None, history_trauma_ortho_bleaching=None,
                                     restoration_at_defect=None)
# rev 2.7 - upstream context for THIS tooth (optional). Diagnocat has no ECR class: its findings shift priors and fill
# clinical inputs; the patient chart adds history / complaint focus. Leave None to use default priors.
from ecr_cds.diagnocat import DiagnocatStudy, DiagnocatTooth, DiagnocatCondition, build_contexts
from ecr_cds.agents.chart import PatientChart, HistoryEvent
TOOTH_FDI = None              # e.g. 26 (FDI / ISO two-digit); None = unknown
DIAGNOCAT_CONDITIONS = []     # e.g. [("Endodontically treated tooth", 0.9, "ai"), ("Artificial crown", 0.95, "ai")]
PATIENT_CHART = None          # e.g. PatientChart(age_years=34, complaint_teeth_iso=[26], symptoms=["pink spot"], history=[HistoryEvent("orthodontics", None, 10)])
TOOTH_CONTEXT = None
if TOOTH_FDI is not None:
    _study = DiagnocatStudy("notebook", [DiagnocatTooth(TOOTH_FDI, [DiagnocatCondition(code="", name=n, probability=p, source=src)
                                                                  for n, p, src in DIAGNOCAT_CONDITIONS])], "mock")
    from ecr_cds.diagnocat.schema import normalise_condition_name
    for _c in _study.teeth[0].conditions: _c.code = normalise_condition_name(_c.name)
    TOOTH_CONTEXT = build_contexts(_study, PATIENT_CHART)[0]
    print("context:", TOOTH_CONTEXT.prior_shifts, TOOTH_CONTEXT.modifier_hints)
VERIFY_CANDIDATES = True      # critic re-inspection + provenance-stamped multiplanar evidence frames
CLAHE_DISPLAY = False         # display-only contrast enhancement in the evidence frames (recorded in the panel metadata)
CONF_DIR = os.path.join(OUT_DIR, "conference")
conference_report = None
if RUN_CONFERENCE and SEED_ZYX is not None and (LABELMAP or auto is not None):
    backend = make_backend()
    print("model backend:", backend.name, "| images:", backend.supports_images)
    conference = CaseConference(backend=backend, max_rounds=CONFERENCE_ROUNDS)
    conference_report = conference.run(vol, tooth_seg, tuple(SEED_ZYX), seed_source=seed_source, auto=auto,
                                       canal_source="expert_mask" if CANAL_MASK else None,
                                       cej_zyx=CEJ_ZYX, apex_zyx=APEX_ZYX, modifiers=ClinicalModifiers(),
                                       policy=policy, case_id=CASE_ID, clinical_findings=CLINICAL_FINDINGS,
                                       context=TOOTH_CONTEXT, verify=VERIFY_CANDIDATES, clahe_display=CLAHE_DISPLAY,
                                       verify_mode=VERIFY_MODE, sharpen_display=SHARPEN_DISPLAY, counterfactual_check=COUNTERFACTUAL_CHECK,
                                       tooth_type=TOOTH_CONTEXT.tooth_type if TOOTH_CONTEXT else "unknown",
                                       examined_teeth=[TOOTH_FDI] if TOOTH_FDI else None,
                                       focus_teeth=PATIENT_CHART.focus_teeth() if PATIENT_CHART else None)
    files = conference.save_outputs(conference_report, CONF_DIR, volume=vol)
    print("STATUS:", conference_report["status"].upper())
    display(Markdown(consensus_markdown(conference_report).split("## Conference transcript")[0]))
    _llm_rows = [r for r in conference_report["candidates"] if r.get("llm_meta")]
    if _llm_rows:
        print("vision-model second opinion per candidate (weight 0 = abstained; grounding 'weak' = answer ignored the image):")
        for r in _llm_rows: print(f"  cand {r['component_id']}: weight={r['llm_weight']:.2f} dse={(r['llm_meta'].get('dse') or {}).get('dse')} grounding={(r['llm_meta'].get('grounding') or {}).get('verdict')}")
    for f in files:
        if f.endswith("_evidence_frame.png"): display(Image(f, width=1000))
        elif f.endswith("_tiles.png") and not VERIFY_CANDIDATES: display(Image(f, width=900))
    print("perception tool calls:", [c["tool"] for c in conference_report["tool_log"]])
    print("Transcript:")
    for t in conference_report["conference"]["transcript"]: print(f"  [{t['agent']} r{t['round']}] {t['text'][:220]}")
else:
    print("Conference skipped (needs a seed and a segmentation).")"""))
C.append(nbf.v4.new_markdown_cell("""### Confirmed diagnosis feedback (improves the radiologist agent over time)

When the diagnosis has been established clinically/histologically, record it. The radiologist agent re-tunes its ECR
criterion weights only after at least 20 independent confirmed cases (ridge-penalised toward the expert prior), and
the report always states which weight version produced a suggestion."""))
C.append(nbf.v4.new_code_cell("""RECORD_DIAGNOSIS = False
CONFIRMED_DIAGNOSIS = None    # one of: ECR, caries, noncarious_cervical_lesion, restoration_margin_or_burnout, anatomical_concavity_or_furcation, imaging_artifact
if RECORD_DIAGNOSIS and conference_report and conference_report.get("suggested_diagnosis"):
    from ecr_cds.agents import record_diagnosis_feedback
    res = record_diagnosis_feedback(os.path.join(OUT_DIR, "diagnosis_feedback.json"), conference_report, CONFIRMED_DIAGNOSIS,
                                    radiologist=conference.radiologist, min_cases=20)
    print(res)
    if res.get("weights", {}).get("updated"):
        np.save(os.path.join(OUT_DIR, "radiologist_weights.npy"), conference.radiologist.W); print("tuned weights saved")
else:
    print("Diagnosis feedback is off. Set RECORD_DIAGNOSIS = True with a CONFIRMED_DIAGNOSIS once established.")"""))

C.append(nbf.v4.new_markdown_cell("""## 4c. Diagnocat -> agentic system (patient-level, rev 2.7)

Feed a **Diagnocat** study (JSON export, report PDF, or the partner API once you have credentials) and a patient chart.
Teeth are triaged (chief complaint, Diagnocat flags, history), each gets its own conference with its context, and the
patient report lists per-tooth diagnosis, Patel code, plan with CDT codes, coverage gaps and cited guideline statements.

Diagnocat **does not detect external cervical resorption** (no such class; an independent validation found no AI
output for 14 external-resorption cases). It provides anatomy, numbering, per-tooth conditions and clinician flags;
resorption reasoning comes from this package. Without a Diagnocat file the pipeline still runs on our own
segmentation, ranking teeth by radiolucent defect volume (no FDI numbering in that fallback)."""))
C.append(nbf.v4.new_code_cell("""RUN_DIAGNOCAT_PIPELINE = False
DIAGNOCAT_JSON = None   # path to a Diagnocat JSON export (or our interchange schema); see docs/DIAGNOCAT_INTEGRATION.md
DIAGNOCAT_PDF = None    # or the Diagnocat CBCT report PDF (best-effort text parse)
MAX_TEETH = 2
if RUN_DIAGNOCAT_PIPELINE and (LABELMAP or auto is not None):
    from ecr_cds.diagnocat import load_diagnocat_json, load_diagnocat_pdf
    from ecr_cds.diagnocat.pipeline import run_diagnocat_pipeline, patient_markdown
    study = load_diagnocat_json(DIAGNOCAT_JSON) if DIAGNOCAT_JSON else load_diagnocat_pdf(DIAGNOCAT_PDF) if DIAGNOCAT_PDF else None
    patient_report = run_diagnocat_pipeline(vol, study, chart=PATIENT_CHART, max_teeth=MAX_TEETH, backend=make_backend(),
                                            max_rounds=CONFERENCE_ROUNDS, clinical_findings=CLINICAL_FINDINGS,
                                            out_dir=os.path.join(OUT_DIR, "diagnocat"), verify=VERIFY_CANDIDATES, verify_mode=VERIFY_MODE,
                                            sharpen_display=SHARPEN_DISPLAY, counterfactual_check=COUNTERFACTUAL_CHECK)
    display(Markdown(patient_markdown(patient_report).split("## Per-tooth conference summaries")[0]))
else:
    print("Diagnocat pipeline off. Set RUN_DIAGNOCAT_PIPELINE = True and point DIAGNOCAT_JSON / DIAGNOCAT_PDF at the export.")"""))

C.append(nbf.v4.new_markdown_cell("""## 5. Clinician-reviewed Patel components and clinical inputs

Enter only what has been reviewed. If any Patel component is `None`, **no code and no recommendation are issued**. Clinical inputs left `None` are reported as missing and keep the recommendation provisional (confirmation required)."""))
C.append(nbf.v4.new_code_cell("""from ecr_cds.decision_matrix import manual_patel_code, recommend_from_matrix, END_STAGE_CODES, end_stage_matrix

# --- reviewed Patel components (None = not established) ---
HEIGHT    = None     # 1 supracrestal | 2 coronal third, subcrestal | 3 middle third | 4 apical third
SPREAD    = None     # "A" <=90 deg | "B" <=180 | "C" <=270 | "D" >270
PROXIMITY = None     # "d" confined to dentine | "p" probable pulpal involvement

# --- clinical inputs (None = unknown) ---
clinical = ClinicalModifiers(symptomatic=None, pulp_vital=None, restorable=None, perio_support_adequate=None,
                             multirooted=None, lesion_on_root=None, surgically_accessible=None, persistent_bleeding=None,
                             growing_patient=None, aesthetic_zone=None, root_fracture_risk_high=None)
TOOTH_TYPE, AGE_YEARS = None, None

code = manual_patel_code(HEIGHT, SPREAD, PROXIMITY)
if code is None and auto_report["status"] in ("ok", "provisional"):
    print("No reviewed components entered; the automatic code", auto_report["patel"]["code"], "is shown above but is NOT adopted here.")
manual_report = {"case_id": CASE_ID, "status": "insufficient_evidence", "patel_code": None}
if code is not None and ECR_CONFIRMED:
    rec = recommend_from_matrix(code, clinical, portal_angle_deg=None)
    manual_report = {"case_id": CASE_ID, "status": "provisional" if rec.missing_inputs else "reviewed", "patel_code": code,
                     "source": "clinician-reviewed components", "recommendation": rec.to_dict(),
                     "end_stage_matrix": end_stage_matrix() if code in END_STAGE_CODES else None,
                     "disclaimer": "Advisory. Irreversible options require explicit clinician confirmation."}
    print(f"Patel {code} -> {rec.primary.value}  (confirmation required: {rec.requires_confirmation})")
    print("Safe set:", [t.value for t in rec.safe_set]); print("Missing inputs:", rec.missing_inputs)
    for r in rec.rationale: print("  -", r)
else:
    print("STATUS: insufficient_evidence - ECR unconfirmed or Patel components incomplete; no recommendation issued.")
json.dump(manual_report, open(os.path.join(OUT_DIR, "manual_report.json"), "w"), indent=2, default=str)"""))

C.append(nbf.v4.new_markdown_cell("""## 6. Training (research; off by default)

Requires independent expert-labelled cases with stable **patient identifiers** (repeated scans of one patient share an id) and should include negative cases. Evaluation is nested: outer leave-one-patient-out, threshold chosen on inner patient folds only. At least four patients are needed to execute; that is a software minimum, not a validation sample size. `INCLUDE_PHANTOMS` adds synthetic cases and marks the model as research-only."""))
C.append(nbf.v4.new_code_cell("""RUN_TRAINING = False
TRAIN_CASES = [  # {"patient_id": "...", "dicom": "...zip or folder", "labelmap": "...nii.gz", "seed": [z,y,x], "lesion_mask": "...nii.gz", "canal_mask": None}
]
INCLUDE_PHANTOMS = False

if RUN_TRAINING:
    from ecr_cds.training import nested_patient_cv, train_from_cases, save_model
    from ecr_cds.preprocess import preprocess_volume
    V, S, L, P, names = [], [], [], [], []
    for c in TRAIN_CASES:
        v_ = load_dicom_series(c["dicom"]); ts = SlicerLabelmapToothSegmenter(c["labelmap"], tuple(c["seed"])); ts.canal_mask_path = c.get("canal_mask")
        s_ = ts(v_.array, v_.spacing); V.append(preprocess_volume(v_.array, roi=s_.tooth)); S.append(s_)
        L.append(load_labelmap(c["lesion_mask"], volume=v_) > 0); P.append(str(c["patient_id"])); names.append(c["dicom"])
    if INCLUDE_PHANTOMS:
        from ecr_cds.phantom import make_phantom
        for k, (h, sdeg, pf) in enumerate([(3.0, 150, True), (5.0, 240, True), (9.0, 150, True), (2.0, 70, False), (9.0, 300, False)]):
            v_, s_, l_, _ = make_phantom(h, sdeg, pf); V.append(preprocess_volume(v_, roi=s_.tooth)); S.append(s_); L.append(l_); P.append(f"phantom{k}"); names.append(f"phantom{k}")
    cv = nested_patient_cv(V, S, L, P)
    for f in cv["folds"]: print(f"patient {f['test_patient']}: Dice {f['dice']:.2f} P {f['precision']:.2f} R {f['recall']:.2f} (inner thr {f['inner_threshold']:.2f})")
    print(f"Dice {cv['dice_mean']:.2f} +/- {cv['dice_sd']:.2f} over {cv['n_patients']} patients;", cv["note"])
    final = train_from_cases(V, S, L); final.threshold = cv["threshold_median"]
    NEW_MODEL = os.path.join(OUT_DIR, "lesion_svm_trained.joblib")
    save_model(final, NEW_MODEL, {"name": "lesion_svm_trained", "data": ("clinical+SYNTHETIC phantom" if INCLUDE_PHANTOMS else "clinical"),
                                  "cases": names, "patients": sorted(set(P)), "nested_cv": cv, "external_validation": False})
    print("saved", NEW_MODEL, "- set LESION_MODEL to this path and re-run Section 4")
else:
    print("Training is off. Provide TRAIN_CASES (>= 4 patients) and set RUN_TRAINING = True.")"""))

C.append(nbf.v4.new_markdown_cell("""## 7. Feedback (off by default — record only real events)

A **decision event** records what the clinician actually chose. An **outcome event** records an observed recall (success/survival at a validated follow-up time). Event IDs are derived from the case id and event type, so re-running a cell cannot double-count. Feedback is applied to the policy only for reports that carry a decision (Section 4)."""))
C.append(nbf.v4.new_code_cell("""RECORD_FEEDBACK = False
CLINICIAN_CHOICE = None     # one of: periodic_review, external_repair, external_repair_vpt, external_repair_rct, internal_repair_rct, intentional_replantation, root_resection_hemisection, decoronation, extraction
OUTCOME = None              # None for a decision event; or {"success": True/False, "survival": True/False, "months": 12.0} for an observed recall

if RECORD_FEEDBACK:
    from ecr_cds.rl_policy import ClinicalContext, Decision, Feedback, build_reward
    from ecr_cds.decision_matrix import Treatment
    from ecr_cds.patel_classifier import PatelParameters
    assert auto_report["status"] in ("ok", "provisional"), "No automatic decision in this report to attach feedback to"
    assert CLINICIAN_CHOICE, "Set CLINICIAN_CHOICE to the treatment actually chosen"
    rep = auto_report; d = rep["decision"]
    patel = PatelParameters(**{k: v for k, v in rep["patel"].items() if k != "code"})
    ctx = ClinicalContext(patel, clinical, TOOTH_TYPE or "unknown", AGE_YEARS)
    dec = Decision(d.get("cell_key") or d["patel_code"], Treatment(d["chosen"]), [Treatment(x) for x in d["ranking"]], d["propensities"],
                   [Treatment(x) for x in d["safe_set"]], d["requires_confirmation"], d["reinspect_requested"], d["rationale"])
    fb = Feedback(Treatment(CLINICIAN_CHOICE), case_id=CASE_ID, **({} if OUTCOME is None else
                  {"outcome_success": OUTCOME.get("success"), "outcome_survival": OUTCOME.get("survival"), "follow_up_months": OUTCOME.get("months")}))
    r, comp = build_reward(dec, fb, patel.borderline, max(d["propensities"].values()), 0)
    applied = policy.update(ctx, dec, fb, r, comp); policy.save(POLICY_PATH)
    print(("applied" if applied else "already recorded (idempotent)"), fb.event_id, "reward", round(r, 3), comp,
          "| events:", len(policy.log), "| independent cases:", policy.n_cases)
    with open(os.path.join(OUT_DIR, "audit_log.jsonl"), "a") as f:
        f.write(json.dumps({"event_id": fb.event_id, "applied": applied, "clinician_choice": CLINICIAN_CHOICE, "outcome": OUTCOME, "reward": r}) + "\\n")
else:
    print("Feedback is off. Set RECORD_FEEDBACK = True with a real CLINICIAN_CHOICE (and OUTCOME only when observed).")"""))

C.append(nbf.v4.new_markdown_cell("""## 8. Policy audit (held-out evaluation; no automatic promotion)

The most recent 30 % of cases are held out; a candidate is refit on the rest and scored on the held-out cases with a doubly-robust estimator and case-level bootstrap. Promotion requires the lower CI bound to beat the deployed policy on at least 50 independent held-out cases, and is a manual step."""))
C.append(nbf.v4.new_code_cell("""from ecr_cds.rl_policy import split_log_by_case, offline_evaluate, promotion_gate
if policy.log:
    tr, ho = split_log_by_case(policy.log, 0.3)
    cand = ConstrainedBanditPolicy()
    for e in tr:
        k = (e["code"], e["chosen"]); r01 = float(np.clip((e["reward"] + 1) / 2.4, 0, 1))
        cand.alpha[k] = cand.alpha.get(k, 1.0) + r01; cand.beta[k] = cand.beta.get(k, 1.0) + (1 - r01)
    target = []
    for e in ho:
        acts = [a for (c, a) in cand.alpha if c == e["code"]] or [e["chosen"]]
        mu = np.array([cand.alpha.get((e["code"], a), 1) / (cand.alpha.get((e["code"], a), 1) + cand.beta.get((e["code"], a), 1)) for a in acts])
        pr = np.exp(8 * mu); pr /= pr.sum(); target.append(dict(zip(acts, pr.tolist())))
    ev = offline_evaluate(ho, target); print(json.dumps(ev, indent=2))
    print("Gate:", "would pass - review manually before promoting" if promotion_gate(ev) else "not met (need >= 50 independent held-out cases and CI lower bound >= deployed value)")
else:
    print("No feedback events logged; nothing to audit.")"""))

C.append(nbf.v4.new_markdown_cell("""## 8b. Fine-tune the second-opinion model (optional, GPU runtime)

The radiologist and skeptic agents can consult a language model. This section fine-tunes one on prompts that are
**exactly** what the agents send at run time, with targets from cases whose truth is known:

* **synthetic phantoms** (ECR, open wedge, NCCL, internal resorption, intact) with *injected* confounders (noise, metal
  streaks, partial volume, grey-level drift, restoration band) — the differential target is the phantom truth and the
  skeptic target lists the confounders that were really injected;
* **clinician-confirmed cases** converted from saved conference reports (`from_confirmed_case`) — the only rows with
  clinical ground truth; they are weighted up and must be kept in their own split.

What this buys: strict JSON, consistent Q1–Q8 reasoning, sensitivity to the confounders, and disagreement with the
rule engine where the rules are wrong on phantoms (e.g. wide shallow NCCL being called ECR). What it cannot buy:
clinical accuracy — synthetic labels are not patients. Evaluate with `ecr_cds.finetune.evaluate` and treat
synthetic scores as reasoning-fidelity metrics only.

Set `RUN_FINETUNE = True` on a GPU runtime (T4 is enough for a 7B model in 4-bit). A pre-generated dataset ships in
`finetune_data/` inside the package; `N_EXTRA_CASES` adds fresh phantoms (about 1 min per case on CPU)."""))
C.append(nbf.v4.new_code_cell("""RUN_FINETUNE = False
N_EXTRA_CASES = 0                      # new synthetic cases to add to the shipped dataset
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" # any chat model with a chat template; 1.5B/3B train faster on T4
EPOCHS = 2
CONFIRMED_CASES = []                   # [(path_to_conference_report.json, "ECR" | "caries" | ...), ...] clinician-confirmed only
FT_DIR = os.path.join(OUT_DIR, "finetune")

if RUN_FINETUNE:
    import subprocess, shutil, glob, random
    from ecr_cds.finetune import dataset as ftds
    os.makedirs(FT_DIR, exist_ok=True)
    raw = os.path.join(FT_DIR, "raw.jsonl")
    shipped = glob.glob(os.path.join(os.path.dirname(ecr_cds.__file__), "..", "finetune_data", "split", "*.jsonl"))
    with open(raw, "w") as f:
        for p in shipped:
            f.write(open(p).read())
    for i in range(N_EXTRA_CASES):
        rng = random.Random(1000 + i)
        exs, meta = ftds.generate_case(ftds.sample_case(rng), rng, f"colab_{i:04d}")
        ftds.write_jsonl(exs, raw); print("generated", meta["case_id"], meta["spec"]["morphology"], meta["status"])
    counts = ftds.split(raw, os.path.join(FT_DIR, "split"), eval_fraction=0.2)
    print("split:", counts)
    if CONFIRMED_CASES:   # real labelled cases: own file, appended to train only
        with open(os.path.join(FT_DIR, "split", "train.jsonl"), "a") as f:
            for path, dx in CONFIRMED_CASES:
                for e in ftds.from_confirmed_case(path, dx, weight=3):
                    f.write(e.to_json() + "\\n")
    # baseline: deterministic rules on the held-out prompts
    from ecr_cds.finetune.evaluate import evaluate, RulesBackend
    rows = [json.loads(l) for l in open(os.path.join(FT_DIR, "split", "eval.jsonl"))]
    print("rules baseline:", json.dumps(evaluate(rows, RulesBackend())["radiologist"], indent=1))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.44", "peft>=0.12", "trl>=0.9", "datasets", "accelerate", "bitsandbytes"], check=False)
    subprocess.run([sys.executable, "-m", "ecr_cds.finetune.train_lora", "--data", os.path.join(FT_DIR, "split", "train.jsonl"),
                    "--eval", os.path.join(FT_DIR, "split", "eval.jsonl"), "--base", BASE_MODEL, "--qlora",
                    "--epochs", str(EPOCHS), "--out", os.path.join(FT_DIR, "adapter")], check=True)
    os.environ["LLM_LOCAL_MODEL"] = os.path.join(FT_DIR, "adapter")
    from ecr_cds.agents import LocalHFBackend
    tuned = LocalHFBackend()
    print("fine-tuned model:", json.dumps(evaluate(rows, tuned, limit=60), indent=1))
    print("Re-run section 4b with LLM_BACKEND='auto' to use the fine-tuned model as the second opinion.")
else:
    print("Fine-tuning skipped (RUN_FINETUNE=False). The conference keeps using the rules plus any configured hosted model.")"""))

C.append(nbf.v4.new_markdown_cell("## 9. Export — only files generated in this run (no raw DICOM, no identifiers in reports)"))
C.append(nbf.v4.new_code_cell("""EXPORT_TO_DRIVE = False
generated = sorted(f for f in os.listdir(OUT_DIR) if not f.startswith("phantom_scan"))
print("Generated in", OUT_DIR); [print("  ", f) for f in generated]
print("Note: image crops / masks (if any) contain patient anatomy even though identifiers are removed.")
if EXPORT_TO_DRIVE:
    from google.colab import drive; drive.mount("/content/drive")
    import shutil, time; dst = f"/content/drive/MyDrive/ecr_cds_outputs/{CASE_ID}_{time.strftime('%Y%m%d_%H%M%S')}"; os.makedirs(dst, exist_ok=True)
    for f in generated: shutil.copy(os.path.join(OUT_DIR, f), dst)
    print("exported to", dst)"""))

nb.cells = C
nbf.write(nb, os.path.join(ROOT, "ECR_CDS_Colab.ipynb"))
print("written, embedded package", len(PKG_B64) // 1024, "KB")
