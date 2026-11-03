"""Loaders: Partner API client (configurable), exported JSON, report PDF text, STL -> voxel mask."""
from __future__ import annotations

import json
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .schema import DiagnocatCondition, DiagnocatStudy, DiagnocatTooth, normalise_condition_name, _SEVERITY_WORDS

# --------------------------------------------------------------------------- #
# JSON (our interchange schema, or a Partner-API-like payload mapped through FIELD_MAP)
FIELD_MAP_DEFAULT = {
    "teeth": ["teeth", "tooth_list", "dental_chart", "items"],
    "iso_number": ["iso_number", "ISONumber", "iso", "fdi", "tooth_number", "number", "toothNumber"],
    "conditions": ["conditions", "findings", "pathologies", "tags"],
    "cond_name": ["name", "code", "condition", "title", "type"],
    "cond_prob": ["probability", "confidence", "score", "p"],
    "cond_source": ["source", "origin", "author"],
    "cond_severity": ["severity", "grade", "degree"],
    "seed_zyx": ["seed_zyx", "seed", "voxel_zyx"],
    "stl_path": ["stl_path", "stl", "mesh"],
    "canal_stl_path": ["canal_stl_path", "pulp_stl", "canal_stl"],
    "slices": ["slices", "images", "slice_paths"],
    "comment": ["comment", "comments", "note"],
}


def _get(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_study_dict(data: Dict[str, Any], source: str = "json_export", field_map: Optional[Dict] = None) -> DiagnocatStudy:
    fm = {**FIELD_MAP_DEFAULT, **(field_map or {})}
    teeth_raw = _get(data, fm["teeth"], [])
    teeth: List[DiagnocatTooth] = []
    for t in teeth_raw:
        iso = _get(t, fm["iso_number"])
        if iso is None:
            continue
        conds: List[DiagnocatCondition] = []
        for c in _get(t, fm["conditions"], []) or []:
            if isinstance(c, str):
                c = {"name": c}
            name = str(_get(c, fm["cond_name"], "unknown"))
            p = _get(c, fm["cond_prob"])
            try:
                p = None if p is None else float(p)
                if p is not None and p > 1.0:  # percent
                    p = p / 100.0
            except (TypeError, ValueError):
                p = None
            src = str(_get(c, fm["cond_source"], "ai")).lower()
            src = "clinician" if src in ("clinician", "manual", "user", "doctor", "approved") else "ai"
            sev = _get(c, fm["cond_severity"])
            conds.append(DiagnocatCondition(code=normalise_condition_name(name), name=name, probability=p, source=src,
                                            severity=None if sev is None else str(sev).lower(),
                                            detail={k: v for k, v in c.items() if k not in sum(fm.values(), [])}))
        teeth.append(DiagnocatTooth(iso_number=int(iso), conditions=conds, seed_zyx=_get(t, fm["seed_zyx"]),
                                    stl_path=_get(t, fm["stl_path"]), canal_stl_path=_get(t, fm["canal_stl_path"]),
                                    slices=list(_get(t, fm["slices"], []) or []), comment=_get(t, fm["comment"]),
                                    raw={k: v for k, v in t.items() if k not in sum(fm.values(), [])}))
    return DiagnocatStudy(study_id=str(_get(data, ["study_id", "id", "study_uid", "studyId"], "diagnocat_study")),
                          teeth=teeth, source=source, diagnocat_version=_get(data, ["diagnocat_version", "version", "app_version"]),
                          report_pdf=_get(data, ["report_pdf"]), label_map_path=_get(data, ["label_map_path"]),
                          notes=list(_get(data, ["notes"], []) or []), raw=data)


def load_diagnocat_json(path: str, field_map: Optional[Dict] = None) -> DiagnocatStudy:
    return parse_study_dict(json.load(open(path)), source="json_export", field_map=field_map)


# --------------------------------------------------------------------------- #
# Report PDF (best effort). Diagnocat's CBCT radiological report lists one block per tooth with the tooth
# number and its conditions (optionally with probabilities when "Conditions Details" is on).
_TOOTH_HDR = re.compile(r"\b(?:tooth|teeth)\s*#?\s*(\d{2})\b", re.I)
_PROB = re.compile(r"(\d{1,3})\s*%|\b(0\.\d{1,2})\b")


def parse_report_text(text: str, study_id: str = "diagnocat_report") -> DiagnocatStudy:
    from .schema import CONDITION_ALIASES
    names = sorted(CONDITION_ALIASES.keys(), key=len, reverse=True)
    teeth: Dict[int, DiagnocatTooth] = {}
    notes = []
    # split into tooth blocks
    idx = [(m.start(), int(m.group(1))) for m in _TOOTH_HDR.finditer(text)]
    if not idx:
        notes.append("no 'Tooth NN' headers found in the report text; nothing parsed")
    for k, (start, iso) in enumerate(idx):
        end = idx[k + 1][0] if k + 1 < len(idx) else len(text)
        block = text[start:end]
        low = block.lower()
        tooth = teeth.setdefault(iso, DiagnocatTooth(iso_number=iso))
        for name in names:
            for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
                tail = low[m.end(): m.end() + 40]
                pm = _PROB.search(tail)
                p = None
                if pm:
                    p = float(pm.group(1)) / 100 if pm.group(1) else float(pm.group(2))
                sev = next((w for w in _SEVERITY_WORDS if w in tail[:25]), None)
                src = "clinician" if re.search(r"added by|manual|approved", tail) else "ai"
                code = normalise_condition_name(name)
                if not any(c.code == code for c in tooth.conditions):
                    tooth.conditions.append(DiagnocatCondition(code=code, name=name, probability=p, source=src, severity=sev))
    version = re.search(r"diagnocat\s+(?:ai\s+)?v?(\d+(?:\.\d+)+)", text, re.I)
    return DiagnocatStudy(study_id=study_id, teeth=sorted(teeth.values(), key=lambda t: t.iso_number), source="report_pdf",
                          diagnocat_version=version.group(1) if version else None, notes=notes)


def load_diagnocat_pdf(path: str) -> DiagnocatStudy:
    text = ""
    try:
        import pypdf
        text = "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(path).pages)
    except Exception as e:  # pragma: no cover
        try:
            import fitz
            text = "\n".join(p.get_text() for p in fitz.open(path))
        except Exception as e2:
            raise RuntimeError(f"could not read PDF text: {e}; {e2}")
    st = parse_report_text(text, study_id=os.path.splitext(os.path.basename(path))[0])
    st.report_pdf = path
    st.notes.append("parsed from report PDF text (best effort); verify against the Diagnocat UI")
    return st


# --------------------------------------------------------------------------- #
# STL -> voxel mask in our scan grid (pure numpy; binary or ASCII STL)
def read_stl(path: str) -> np.ndarray:
    """Return triangles as (n, 3, 3) float array in STL units (Diagnocat exports mm)."""
    with open(path, "rb") as f:
        head = f.read(80)
        rest = f.read()
    if head[:5].lower().strip() == b"solid" and b"facet" in rest[:2000]:
        txt = (head + rest).decode("ascii", "ignore")
        v = re.findall(r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", txt)
        arr = np.array(v, dtype=float)
        return arr.reshape(-1, 3, 3)
    n = struct.unpack("<I", rest[:4])[0]
    data = np.frombuffer(rest[4:4 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return data["v"].astype(float)


def stl_to_mask(path: str, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], origin_xyz=(0.0, 0.0, 0.0),
                axis_order: str = "xyz", flip: Tuple[bool, bool, bool] = (False, False, False), samples_per_mm: float = 12.0) -> np.ndarray:
    """Rasterise a closed surface mesh into a filled boolean volume indexed (z, y, x).

    Mesh coordinates are mm; `origin_xyz` is the scan origin, `axis_order` maps mesh axes to scan (x, y, z)
    and `flip` mirrors axes.  Surface points are sampled densely, marked, then each axial slice is
    hole-filled - adequate for tooth / pulp shells.  Verify alignment visually: STL exports and DICOM
    patient coordinates do not always share handedness."""
    tri = read_stl(path)
    if tri.size == 0:
        return np.zeros(shape, bool)
    order = ["xyz".index(c) for c in axis_order]
    tri = tri[:, :, order]
    tri = tri - np.asarray(origin_xyz, float)
    for ax, fl in enumerate(flip):
        if fl:
            tri[:, :, ax] = -tri[:, :, ax]
    sp = np.array([spacing[2], spacing[1], spacing[0]])  # (x, y, z) mm per voxel
    mask = np.zeros(shape, bool)
    # barycentric sampling density per triangle from its longest edge
    e = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1); e = np.maximum(e, np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1))
    for t, L in zip(tri, e):
        n = max(2, int(np.ceil(L * samples_per_mm)))
        u = np.linspace(0, 1, n)
        uu, vv = np.meshgrid(u, u, indexing="ij")
        sel = uu + vv <= 1.0
        pts = t[0] + uu[sel, None] * (t[1] - t[0]) + vv[sel, None] * (t[2] - t[0])
        ijk = np.round(pts / sp).astype(int)              # (x, y, z) voxel indices
        ok = np.all((ijk >= 0) & (ijk < np.array(shape)[::-1]), axis=1)
        ijk = ijk[ok]
        mask[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True
    for z in range(shape[0]):
        if mask[z].any():
            mask[z] = ndi.binary_fill_holes(mask[z])
    return mask


# --------------------------------------------------------------------------- #
class DiagnocatClient:
    """Partner API client skeleton.

    Diagnocat publishes its Partner API as a partner-gated Postman collection; endpoint paths and payloads are
    therefore *configurable* here (environment or constructor) rather than hard-coded.  Set
    DIAGNOCAT_BASE_URL, DIAGNOCAT_API_KEY and, from the collection you receive, DIAGNOCAT_EP_UPLOAD,
    DIAGNOCAT_EP_STATUS, DIAGNOCAT_EP_REPORT ("{study_id}" placeholder).  Without them `available` is False
    and the pipeline uses exported files.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, endpoints: Optional[Dict[str, str]] = None,
                 field_map: Optional[Dict] = None, timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("DIAGNOCAT_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("DIAGNOCAT_API_KEY")
        self.endpoints = {"upload": os.environ.get("DIAGNOCAT_EP_UPLOAD", "/studies"),
                          "status": os.environ.get("DIAGNOCAT_EP_STATUS", "/studies/{study_id}"),
                          "report": os.environ.get("DIAGNOCAT_EP_REPORT", "/studies/{study_id}/report"),
                          **(endpoints or {})}
        self.field_map = field_map
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def upload_study(self, dicom_zip_path: str, patient_ref: str = "anonymous") -> str:
        import requests
        with open(dicom_zip_path, "rb") as f:
            r = requests.post(self.base_url + self.endpoints["upload"], headers=self._headers(), timeout=self.timeout,
                              files={"file": (os.path.basename(dicom_zip_path), f, "application/zip")}, data={"patient": patient_ref})
        r.raise_for_status()
        js = r.json()
        return str(_get(js, ["study_id", "id", "studyId", "uid"]))

    def wait_for_report(self, study_id: str, poll_s: float = 15.0, max_wait_s: float = 1800.0) -> Dict[str, Any]:
        import time
        import requests
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            r = requests.get(self.base_url + self.endpoints["status"].format(study_id=study_id), headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            st = str(_get(r.json(), ["status", "state"], "")).lower()
            if st in ("done", "ready", "completed", "finished", "success"):
                break
            if st in ("failed", "error"):
                raise RuntimeError(f"Diagnocat processing failed for {study_id}: {r.text[:300]}")
            time.sleep(poll_s)
        r = requests.get(self.base_url + self.endpoints["report"].format(study_id=study_id), headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def fetch_study(self, study_id: str) -> DiagnocatStudy:
        st = parse_study_dict(self.wait_for_report(study_id), source="partner_api", field_map=self.field_map)
        st.study_id = study_id
        st.notes.append("fields mapped through FIELD_MAP; confirm against the Partner API collection")
        return st

    def process(self, dicom_zip_path: str, patient_ref: str = "anonymous") -> DiagnocatStudy:
        if not self.available:
            raise RuntimeError("Diagnocat Partner API not configured (DIAGNOCAT_BASE_URL / DIAGNOCAT_API_KEY); "
                               "export the report JSON/PDF and STL from the Diagnocat UI instead")
        return self.fetch_study(self.upload_study(dicom_zip_path, patient_ref))
