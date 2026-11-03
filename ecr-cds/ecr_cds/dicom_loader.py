"""DICOM series ingestion for CBCT volumes (folder or ZIP archive).

Design notes (from Spin-Neto et al., J Digit Imaging 2011):
* CBCT grey values are NOT calibrated Hounsfield Units. We therefore never
  threshold on absolute HU; all downstream logic uses percentile-normalised
  intensities and texture statistics.
* Plane definition, slice thickness and windowing change measurements. We
  (a) keep the native voxel grid, (b) record acquisition metadata for
  provenance, and (c) re-slice along the tooth long axis only inside the
  classifier using the recorded geometry.

Robustness rules (review revision 2.0.0):
* physical sorting along the slice normal (InstanceNumber is ignored)
* reject: gaps / non-uniform spacing, duplicate positions, duplicate SOP
  instance UIDs, changing orientation / spacing / dimensions, in-plane
  position drift (gantry tilt), multi-frame objects, invalid rescale tags
* per-slice RescaleSlope/Intercept; PixelPaddingValue voxels are reported
  separately and set to the minimum valid intensity, never used as tissue
* ZIP input: unsafe member paths rejected, macOS metadata (__MACOSX, ._*,
  .DS_Store) and non-DICOM vendor files skipped
* de-identified provenance: PatientID / names / UIDs are replaced by a short
  salted hash so cases can be linked without exposing identifiers
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset
except Exception:  # pragma: no cover
    pydicom = None
    FileDataset = object  # type: ignore


class DicomGeometryError(ValueError):
    """Raised when the series cannot be assembled into a trustworthy volume."""


@dataclass
class CBCTVolume:
    """A CBCT volume in LPS patient coordinates.

    array      : (Z, Y, X) float32 grey values after rescale slope/intercept
    spacing    : (dz, dy, dx) in mm
    origin     : ImagePositionPatient of the first (most negative along normal) slice (mm)
    direction  : 3x3 direction cosines (rows = z, y, x axes) in LPS
    metadata   : de-identified acquisition provenance
    padding    : bool mask of PixelPaddingValue voxels (None if absent)
    """

    array: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)
    padding: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def is_isotropic(self) -> bool:
        return bool(np.allclose(self.spacing, self.spacing[0], rtol=0.02))

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """Map (z, y, x) voxel indices to LPS mm coordinates."""
        ijk = np.asarray(ijk, dtype=float)
        return np.asarray(self.origin) + (ijk * np.asarray(self.spacing)) @ self.direction

    def affine_lps(self) -> np.ndarray:
        """4x4 affine mapping (x_idx, y_idx, z_idx, 1) -> LPS mm (nibabel-style column order)."""
        A = np.eye(4)
        # columns correspond to x, y, z index axes
        A[:3, 0] = self.direction[2] * self.spacing[2]
        A[:3, 1] = self.direction[1] * self.spacing[1]
        A[:3, 2] = self.direction[0] * self.spacing[0]
        A[:3, 3] = self.origin
        return A


# --------------------------------------------------------------------------- #
_SKIP_PREFIXES = ("__MACOSX/",)
_SKIP_BASENAMES = {".DS_Store", "Thumbs.db", "DICOMDIR"}


def _safe_member(name: str) -> bool:
    if name.endswith("/"):
        return False
    norm = os.path.normpath(name)
    if norm.startswith("..") or os.path.isabs(norm) or "\\" in name:
        return False
    base = os.path.basename(name)
    if name.startswith(_SKIP_PREFIXES) or base.startswith("._") or base in _SKIP_BASENAMES:
        return False
    return True


def extract_dicom_zip(zip_path: str, out_dir: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
    """Extract only safe, non-metadata members of a scan ZIP. Returns (dir, housekeeping)."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="ecr_dicom_")
    skipped_meta = skipped_unsafe = kept = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            norm = os.path.normpath(name)
            if norm.startswith("..") or os.path.isabs(norm) or "\\" in name:
                skipped_unsafe += 1
                continue
            if not _safe_member(name):
                skipped_meta += 1
                continue
            target = os.path.join(out_dir, norm)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            kept += 1
    return out_dir, {"members_kept": kept, "metadata_skipped": skipped_meta, "unsafe_skipped": skipped_unsafe}


def _hash_id(value, salt: str = "ecr-cds") -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:12]


_META_TAGS = ["Manufacturer", "ManufacturerModelName", "KVP", "XRayTubeCurrent", "ExposureTime",
              "SliceThickness", "PixelSpacing", "Rows", "Columns", "BitsStored", "PixelRepresentation",
              "RescaleIntercept", "RescaleSlope", "PixelPaddingValue", "Modality", "SeriesDescription"]


def _read_headers(path: str) -> Tuple[List["FileDataset"], int]:
    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")
    files, skipped = [], 0
    for root, _, names in os.walk(path):
        for n in sorted(names):
            if n.startswith("._") or n in _SKIP_BASENAMES:
                skipped += 1
                continue
            fp = os.path.join(root, n)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:
                skipped += 1
                continue
            if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "PixelSpacing"):
                skipped += 1
                continue
            ds._fp = fp  # type: ignore[attr-defined]
            files.append(ds)
    if not files:
        raise FileNotFoundError(f"No DICOM slices with geometry found in {path}")
    return files, skipped


def load_dicom_series(path: str, series_uid: Optional[str] = None,
                      expected_slices: Optional[int] = None,
                      position_tol_mm: float = 0.05) -> CBCTVolume:
    """Load a single-frame CBCT series from a folder or ZIP.

    Validates geometry strictly; raises DicomGeometryError on any inconsistency.
    """
    housekeeping: Dict[str, int] = {}
    if os.path.isfile(path) and zipfile.is_zipfile(path):
        path, housekeeping = extract_dicom_zip(path)
    headers, skipped_files = _read_headers(path)
    housekeeping["non_dicom_skipped"] = skipped_files

    by_series: Dict[str, List[FileDataset]] = {}
    for ds in headers:
        by_series.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
    if series_uid is None:
        series_uid = max(by_series, key=lambda k: len(by_series[k]))
    if series_uid not in by_series:
        raise DicomGeometryError(f"Series {series_uid} not found; available: {list(by_series)}")
    slices = by_series[series_uid]

    # ---- consistency checks ----
    for ds in slices:
        if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
            raise DicomGeometryError("Multi-frame DICOM objects are not supported; export single-frame slices")
    ref = slices[0]
    iop_ref = np.asarray(ref.ImageOrientationPatient, dtype=float)
    ps_ref = np.asarray(ref.PixelSpacing, dtype=float)
    dims_ref = (int(ref.Rows), int(ref.Columns))
    sops = set()
    for ds in slices:
        if not np.allclose(np.asarray(ds.ImageOrientationPatient, float), iop_ref, atol=1e-4):
            raise DicomGeometryError("ImageOrientationPatient changes within the series")
        if not np.allclose(np.asarray(ds.PixelSpacing, float), ps_ref, rtol=1e-4):
            raise DicomGeometryError("PixelSpacing changes within the series")
        if (int(ds.Rows), int(ds.Columns)) != dims_ref:
            raise DicomGeometryError("Rows/Columns change within the series")
        sop = str(getattr(ds, "SOPInstanceUID", ""))
        if sop:
            if sop in sops:
                raise DicomGeometryError(f"Duplicate SOPInstanceUID {sop}")
            sops.add(sop)
        for tag in ("RescaleSlope", "RescaleIntercept"):
            v = getattr(ds, tag, None)
            if v is not None:
                try:
                    fv = float(v)
                except Exception as e:
                    raise DicomGeometryError(f"Invalid {tag}={v!r}") from e
                if tag == "RescaleSlope" and (fv == 0 or not np.isfinite(fv)):
                    raise DicomGeometryError(f"Invalid RescaleSlope={fv}")
                if not np.isfinite(fv):
                    raise DicomGeometryError(f"Invalid {tag}={fv}")

    row_cos, col_cos = iop_ref[:3], iop_ref[3:]
    normal = np.cross(row_cos, col_cos)
    normal /= np.linalg.norm(normal)

    positions = np.array([np.asarray(s.ImagePositionPatient, float) for s in slices])
    along = positions @ normal
    order = np.argsort(along)
    slices = [slices[i] for i in order]
    positions, along = positions[order], along[order]

    # in-plane drift (gantry tilt / table shift) -> positions must differ only along the normal
    inplane = positions - np.outer(along, normal)
    if np.any(np.linalg.norm(inplane - inplane[0], axis=1) > position_tol_mm):
        raise DicomGeometryError("Slice positions drift in-plane (gantry tilt or table motion)")
    dz_all = np.diff(along)
    if len(dz_all) == 0:
        raise DicomGeometryError("A single slice is not a volume")
    if np.any(np.abs(dz_all) < position_tol_mm):
        raise DicomGeometryError("Duplicate slice locations detected")
    dz = float(np.median(dz_all))
    if np.any(np.abs(dz_all - dz) > max(0.1 * dz, position_tol_mm)):
        raise DicomGeometryError("Non-uniform slice spacing / missing slices detected")
    if expected_slices is not None and len(slices) != expected_slices:
        raise DicomGeometryError(f"Expected {expected_slices} slices, found {len(slices)}")

    dy, dx = float(ps_ref[0]), float(ps_ref[1])

    # ---- pixel data (second pass; per-slice rescale; padding handled separately) ----
    arrays, pad_masks = [], []
    pad_any = False
    for ds in slices:
        full = pydicom.dcmread(ds._fp, force=True)  # type: ignore[attr-defined]
        raw = full.pixel_array
        pad_val = getattr(full, "PixelPaddingValue", None)
        pad = np.zeros(raw.shape, bool)
        if pad_val is not None:
            try:
                pad = raw == int(pad_val)
                pad_any |= bool(pad.any())
            except Exception:
                pass
        slope = float(getattr(full, "RescaleSlope", 1.0))
        intercept = float(getattr(full, "RescaleIntercept", 0.0))
        arrays.append(raw.astype(np.float32) * slope + intercept)
        pad_masks.append(pad)
    volume = np.stack(arrays, axis=0)
    padding = np.stack(pad_masks, axis=0) if pad_any else None
    if padding is not None:
        valid_min = float(volume[~padding].min())
        volume[padding] = valid_min

    meta: Dict[str, object] = {t: _safe(getattr(ref, t, None)) for t in _META_TAGS}
    meta.update({
        "PatientID_hash": _hash_id(getattr(ref, "PatientID", None)),
        "StudyInstanceUID_hash": _hash_id(getattr(ref, "StudyInstanceUID", None)),
        "SeriesInstanceUID_hash": _hash_id(series_uid),
        "NumberOfSlices": len(slices),
        "VoxelSize_mm": (abs(dz), dy, dx),
        "Isotropic": bool(np.allclose([abs(dz), dy], dx, rtol=0.02)),
        "FieldOfView_mm": (abs(dz) * len(slices), dy * dims_ref[0], dx * dims_ref[1]),
        "IntensityRange": (float(volume.min()), float(volume.max())),
        "PaddingVoxels": int(padding.sum()) if padding is not None else 0,
        "MaxSlicePositionResidual_mm": float(np.max(np.abs(dz_all - dz))),
        "Housekeeping": housekeeping,
        "SeriesAvailable": len(by_series),
    })

    direction = np.vstack([normal, col_cos, row_cos])
    return CBCTVolume(array=volume, spacing=(abs(dz), dy, dx),
                      origin=tuple(float(v) for v in positions[0]),
                      direction=direction, metadata=meta, padding=padding)


def _safe(v):
    if v is None:
        return None
    try:
        if hasattr(v, "__iter__") and not isinstance(v, str):
            return [float(x) if _isnum(x) else str(x) for x in v]
        return float(v) if _isnum(v) else str(v)
    except Exception:
        return str(v)


def _isnum(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False
