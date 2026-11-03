"""Synthetic CBCT phantom generator (500 slices, 0.1 mm isotropic) used for
self-tests, demos and the bundled phantom-trained model."""
from __future__ import annotations
import os
import numpy as np
from .segmentation import ToothSegmentation

VOX = 0.1  # mm, isotropic
SHAPE = (500, 160, 160)  # Z (axial slices), Y, X  -> 50 x 16 x 16 mm
TINY_VOX = 0.15  # mm; under the 0.2 mm geometry-QA limit after DICOM round-trip


def make_phantom(height_mm_apical: float, spread_deg: float, perforate: bool,
                 portal_angle_deg: float = 0.0, morphology: str = "wedge"):
    """Return (volume, masks, truth). Tooth axis = Z. Crown at low z.

    morphology: "wedge" (legacy open wedge), "ecr" (narrow cervical portal, undermining interior,
    irregular margins, mottled repair inclusions), "nccl" (shallow smooth supracrestal saucer) or
    "internal" (canal-centred balloon expansion without a surface breach).
    Synthetic geometry only - used to exercise code paths, not to validate detection.
    """
    Z, Y, X = SHAPE
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    cy, cx = Y / 2, X / 2
    r = np.hypot(yy - cy, xx - cx) * VOX                       # radial distance mm
    z_mm = zz * VOX
    cej = 12.0                                                 # CEJ at 12 mm from top
    apex = cej + 21.0                                          # root length 21 mm
    crest = cej + 1.5                                          # bone crest 1.5 mm apical to CEJ
    # tooth: crown radius 4.5 mm above CEJ, root tapers 3.5 -> 0.8 mm
    root_frac = np.clip((z_mm - cej) / (apex - cej), 0, 1)
    tooth_r = np.where(z_mm < cej, 4.5, 3.5 - 2.7 * root_frac)
    tooth = (r <= tooth_r) & (z_mm >= 4.0) & (z_mm <= apex)
    canal_r = np.where(z_mm < cej, 1.6, 1.2 - 0.9 * root_frac)
    canal = (r <= canal_r) & (z_mm >= 6.0) & (z_mm <= apex - 0.5)
    enamel = tooth & (z_mm < cej) & (r > 3.3)
    bone = (~tooth) & (z_mm >= crest) & (z_mm <= apex + 5) & (r <= 7.5) & (r > tooth_r + 0.25)

    # lesion: wedge on the root surface from CEJ-0.5 mm down to cej+height, angular width spread_deg,
    # radially from surface inward to (canal_r + gap)
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) - portal_angle_deg) % 360
    ang = np.minimum(ang, 360 - ang)  # centred at portal
    gap = -0.2 if perforate else 1.0  # mm; negative => invades canal
    inner_r = canal_r + gap
    lesion = tooth & (z_mm >= cej - 0.5) & (z_mm <= cej + height_mm_apical) \
        & (ang <= spread_deg / 2) & (r >= inner_r) & (r <= tooth_r + 0.01)
    lesion &= ~canal if not perforate else np.ones_like(lesion)
    lesion &= tooth
    inclusions = np.zeros(SHAPE, bool)
    if morphology == "ecr":
        from scipy import ndimage as _ndi
        rng0 = np.random.default_rng(1)
        # narrow portal of entry at the CEJ (channel ~0.8 mm wide) ...
        portal = tooth & (np.abs(z_mm - cej) <= 0.4) & (ang <= 10) & (r >= tooth_r - 0.8)
        # ... feeding an interior cavity that stays 0.5 mm under the root surface (undermining)
        interior = lesion & (r <= tooth_r - 0.5) & (z_mm >= cej)
        field = _ndi.gaussian_filter(rng0.normal(size=SHAPE).astype(np.float32), sigma=2.5)
        field /= field.std()
        interior &= field > -0.9           # irregular, ill-defined margins (~0.25 mm texture)
        lesion = (portal | interior) & tooth
        if not perforate:
            lesion &= ~canal
        # mottled fibro-osseous repair tissue: small radiopaque islands inside the cavity
        blobs = _ndi.gaussian_filter(rng0.normal(size=SHAPE).astype(np.float32), sigma=1.5)
        inclusions = lesion & (blobs > 1.2 * blobs.std()) & (r <= tooth_r - 0.6)
    elif morphology == "internal":
        # internal inflammatory resorption: balloon-shaped expansion of the canal, centred on it, no surface breach
        zc = cej + max(2.0, height_mm_apical / 2)
        lesion = tooth & (np.hypot(r, 0) <= canal_r + 1.3) & (np.abs(z_mm - zc) <= max(1.0, height_mm_apical / 2)) \
            & (r <= tooth_r - 0.6)
        lesion |= canal & (np.abs(z_mm - zc) <= max(1.0, height_mm_apical / 2))
    elif morphology == "nccl":
        # shallow smooth saucer on the exposed root just coronal to the crest, open to the surface
        lesion = tooth & (z_mm >= cej - 0.3) & (z_mm <= min(cej + height_mm_apical, crest - 0.2)) \
            & (ang <= spread_deg / 2) & (r >= tooth_r - 0.6)

    vol = np.zeros(SHAPE, np.float32)
    rng = np.random.default_rng(0)
    vol[bone] = 900; vol[tooth] = 1600; vol[enamel] = 2600; vol[canal] = 300
    vol[lesion] = 700
    vol[inclusions] = 1500
    vol += rng.normal(0, 60, SHAPE).astype(np.float32)
    truth = {"height_mm": height_mm_apical, "spread_deg": spread_deg, "perforate": perforate,
             "cej_mm": cej, "apex_mm": apex, "crest_mm": crest, "morphology": morphology}
    canal_out = canal & ~lesion if (not perforate or morphology == "internal") else canal
    seg = ToothSegmentation(tooth=tooth, canal=canal_out, bone=bone)
    return vol, seg, lesion, truth


def make_tiny_cbct(voxel_mm: float = TINY_VOX, seed: int = 0):
    """One tooth, enamel, canal, bone and a cervical surface defect in a small isotropic volume.

    About 16 x 14 x 14 mm.  Used to exercise ``vlm-measure`` without a patient scan.
    Intensities follow the large phantom (dentine 1600, enamel 2600, lesion ~620) so the
    classical segmenter and cervical screen have something to find.  Not a clinical phantom.
    """
    sp = float(voxel_mm)
    Z, Y, X = int(round(16 / sp)), int(round(14 / sp)), int(round(14 / sp))
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    cy, cx = (Y - 1) / 2.0, (X - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx) * sp
    z_mm = zz * sp
    cej, apex, crest = 5.0, 13.5, 6.5
    root_frac = np.clip((z_mm - cej) / (apex - cej), 0, 1)
    tooth_r = np.where(z_mm < cej, 3.4, 2.8 - 1.8 * root_frac)
    tooth = (r <= tooth_r) & (z_mm >= 2.0) & (z_mm <= apex)
    canal_r = np.where(z_mm < cej, 0.7, 0.55 - 0.25 * root_frac)
    canal = (r <= np.maximum(canal_r, 0.25)) & (z_mm >= 4.0) & (z_mm <= apex - 0.4)
    enamel = tooth & (z_mm < cej - 0.4) & (r > tooth_r - 0.7)
    bone = (~tooth) & (z_mm >= crest) & (z_mm <= apex + 1.2) & (r <= 6.0) & (r >= tooth_r + 0.35)
    ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
    ang = np.minimum(ang, 360.0 - ang)
    lesion = tooth & (z_mm >= cej - 0.3) & (z_mm <= cej + 2.4) & (ang <= 40) & (r >= tooth_r - 1.4)
    lesion &= ~canal
    vol = np.zeros((Z, Y, X), np.float32)
    rng = np.random.default_rng(seed)
    vol[bone] = 900
    vol[tooth] = 1600
    vol[enamel] = 2600
    vol[canal] = 280
    vol[lesion] = 620
    vol += rng.normal(0, 20, vol.shape).astype(np.float32)
    return vol


def write_dicom_series(vol: np.ndarray, out_dir: str, voxel_mm: float = None):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    study, series = generate_uid(), generate_uid()
    vox = VOX if voxel_mm is None else float(voxel_mm)
    v16 = np.clip(vol, 0, 4095).astype(np.uint16)
    for i in range(vol.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.Modality = "CT"; ds.Manufacturer = "Phantom"; ds.KVP = 90
        ds.PatientID = "PHANTOM"; ds.SeriesDescription = "CBCT phantom"
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[2]
        ds.PixelSpacing = [vox, vox]; ds.SliceThickness = vox
        ds.ImagePositionPatient = [0.0, 0.0, i * vox]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.InstanceNumber = i + 1
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        ds.RescaleIntercept = 0; ds.RescaleSlope = 1
        ds.PixelData = v16[i].tobytes()
        ds.is_little_endian = True; ds.is_implicit_VR = False
        ds.save_as(os.path.join(out_dir, f"slice_{i:04d}.dcm"), enforce_file_format=True)


