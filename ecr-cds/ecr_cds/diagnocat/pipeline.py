"""Patient-level pipeline: DICOM + Diagnocat study (+ chart) -> prioritised per-tooth conferences -> report.

    volume = load_dicom(...)                                  # unmodified voxels, spacing, orientation kept
    study  = load_diagnocat_json(...) | load_diagnocat_pdf(...) | DiagnocatClient().process(zip)
    report = run_diagnocat_pipeline(volume, study, chart=PatientChart(...), max_teeth=3)

Teeth are ordered by triage priority (chief complaint focus, Diagnocat flags, chart history).  For each tooth
the perception toolkit segments the tooth (Diagnocat STL mask if supplied, else our automatic segmentation seeded
at the tooth), and the conference runs with the tooth's context.  Diagnocat has no ECR class, so resorption
reasoning comes from our screen + conference; Diagnocat supplies anatomy, numbering, per-tooth conditions and
clinician-entered flags.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..agents import CaseConference, consensus_markdown
from ..agents.chart import PatientChart
from ..agents.clinical import ClinicalFindings
from ..auto_segment import AutoToothSegmenter
from ..decision_matrix import ClinicalModifiers
from ..knowledge import KnowledgeBase
from ..perception.tools import PerceptionToolkit
from .adapter import stl_to_mask
from .context import ToothContext, build_contexts
from .schema import DiagnocatStudy


class MaskToothSegmenter:
    """ToothSegmenter that uses an externally supplied tooth mask (e.g. voxelised Diagnocat STL)."""

    def __init__(self, auto, tooth_mask: np.ndarray, canal_mask: Optional[np.ndarray] = None):
        self.auto, self.tooth_mask, self.canal_mask = auto, tooth_mask.astype(bool), canal_mask
        self.canal_mode: Optional[str] = None

    def __call__(self, volume: np.ndarray, spacing):
        from ..auto_segment import LABEL_BONE, extract_canal_auto
        from ..segmentation import ToothSegmentation
        tooth = self.tooth_mask
        if self.canal_mask is not None:
            canal, self.canal_mode = self.canal_mask.astype(bool) & tooth, "expert_mask"
        else:
            canal, self.canal_mode = extract_canal_auto(volume, tooth, self.auto.t_dense)
        return ToothSegmentation(tooth=tooth, canal=canal, bone=(self.auto.labels == LABEL_BONE))


def _seed_for_tooth(ctx: ToothContext, study: Optional[DiagnocatStudy], volume, propose: Dict, used_labels: set,
                    stl_kwargs: Optional[Dict] = None):
    """Seed voxel and optional tooth mask for one tooth: Diagnocat STL > Diagnocat seed > next unused auto tooth
    ranked by radiolucent defect volume (our segmentation carries no FDI numbering, so this is by rank)."""
    t = study.tooth(ctx.iso_number) if study is not None else None
    if t is not None and t.stl_path and os.path.exists(t.stl_path):
        mask = stl_to_mask(t.stl_path, volume.shape, volume.spacing, **(stl_kwargs or {}))
        if mask.any():
            return tuple(np.argwhere(mask).mean(0).astype(int)), "diagnocat_stl", mask
    if t is not None and t.seed_zyx:
        return tuple(int(x) for x in t.seed_zyx), "diagnocat_seed", None
    scores = [s for s in propose.get("scores", []) if s["label"] not in used_labels and "metal" not in s.get("note", "")]
    if scores:
        s = max(scores, key=lambda r: r.get("radiolucent_defect_mm3", 0))
        used_labels.add(s["label"])
        return tuple(int(x) for x in s["seed_zyx"]), "auto_radiolucency_rank", None
    return None, "none", None


def run_diagnocat_pipeline(volume, study: Optional[DiagnocatStudy], chart: Optional[PatientChart] = None,
                           teeth_iso: Optional[List[int]] = None, max_teeth: int = 3, backend=None, max_rounds: int = 2,
                           modifiers: Optional[ClinicalModifiers] = None, clinical_findings: Optional[ClinicalFindings] = None,
                           out_dir: Optional[str] = None, render: bool = True, verify: bool = True,
                           seeds_override: Optional[Dict[int, tuple]] = None, canal_masks: Optional[Dict[int, np.ndarray]] = None,
                           stl_kwargs: Optional[Dict] = None, verify_mode: str = "smooth", sharpen_display: bool = False,
                           counterfactual_check: bool = False) -> Dict[str, Any]:
    """stl_kwargs: origin_xyz / axis_order / flip passed to stl_to_mask (STL-to-DICOM alignment must be verified visually).
    verify_mode / sharpen_display / counterfactual_check: rev 2.8 options forwarded to CaseConference.run (see docs/NEMOTRON_LOCAL.md)."""
    t0 = time.time()
    used_labels: set = set()
    toolkit = PerceptionToolkit()
    kb = KnowledgeBase()
    contexts = build_contexts(study, chart, teeth_iso)
    if not contexts:   # no context at all: single tooth chosen by our own screen
        contexts = [ToothContext(iso_number=0, tooth_type="unknown", notes=["no Diagnocat / chart context; tooth chosen by radiolucency screen"])]
    seg_out = toolkit.call("segment_teeth", array=volume.array, spacing=volume.spacing)
    auto = seg_out["auto"]
    propose = toolkit.call("propose_regions", auto=auto, array=volume.array, spacing=volume.spacing)
    focus = chart.focus_teeth() if chart is not None else []
    per_tooth: List[Dict[str, Any]] = []
    examined: List[int] = []
    conf = CaseConference(backend=backend, max_rounds=max_rounds)
    for ctx in contexts[:max_teeth]:
        if seeds_override and ctx.iso_number in seeds_override:
            seed, seed_src, mask = tuple(seeds_override[ctx.iso_number]), "clinician", None
        else:
            seed, seed_src, mask = _seed_for_tooth(ctx, study, volume, propose, used_labels, stl_kwargs)
        if seed is None:
            per_tooth.append({"iso_number": ctx.iso_number, "status": "not_located", "context": ctx.to_dict(),
                              "note": "no seed from Diagnocat or the radiolucency screen"})
            continue
        cm = canal_masks.get(ctx.iso_number) if canal_masks else None
        if mask is not None:
            segm = MaskToothSegmenter(auto, mask, cm)
        else:
            try:
                segm = AutoToothSegmenter(auto, seed=seed)
                segm.canal_mask = cm
                _ = int(auto.tooth_ids[segm.seed])
                if _ == 0:
                    raise ValueError("seed outside automatic tooth")
            except Exception as e:
                per_tooth.append({"iso_number": ctx.iso_number, "status": "not_located", "context": ctx.to_dict(), "note": str(e)})
                continue
        examined.append(ctx.iso_number)
        cons = conf.run(volume, segm, seed, seed_src, auto=auto, modifiers=modifiers, clinical_findings=clinical_findings,
                        tooth_type=ctx.tooth_type, age_years=chart.age_years if chart else None,
                        case_id=f"{volume.metadata.get('SeriesInstanceUID_hash', 'case')}_t{ctx.iso_number}",
                        render_tiles=render, context=ctx, toolkit=toolkit, verify=verify, verify_mode=verify_mode,
                        sharpen_display=sharpen_display, counterfactual_check=counterfactual_check,
                        examined_teeth=list(examined), focus_teeth=focus)
        if out_dir:
            conf.save_outputs(cons, os.path.join(out_dir, f"tooth_{ctx.iso_number}"), volume=volume)
        per_tooth.append({"iso_number": ctx.iso_number, "seed_zyx": list(map(int, seed)), "seed_source": seed_src,
                          "status": cons["status"], "suggested_diagnosis": cons.get("suggested_diagnosis"),
                          "suggested_patel": cons.get("suggested_patel"), "suggested_plan": cons.get("suggested_plan"),
                          "coverage": cons.get("coverage"), "verification": cons.get("verification"),
                          "context": ctx.to_dict(), "report": cons})
    unexamined = [c.iso_number for c in contexts[max_teeth:]]
    report = {
        "patient_report_version": "2.7",
        "source_volume": volume.metadata, "spacing_zyx_mm": list(map(float, volume.spacing)),
        "diagnocat_study": study.to_dict() if study is not None else None,
        "chart": chart.to_dict() if chart is not None else None,
        "triage_order": [{"iso_number": c.iso_number, "priority": c.priority, "tooth_type": c.tooth_type} for c in contexts],
        "teeth_examined": examined, "teeth_not_examined": unexamined,
        "per_tooth": per_tooth,
        "tool_log": toolkit.log_dicts(),
        "knowledge_scope": [{"id": s["id"], "source": s["source"], "url": s.get("url"), "text": s["text"]} for s in kb.for_tags(["diagnocat"])],
        "seconds": round(time.time() - t0, 1),
        "requires_confirmation": True, "clinical_validation": False,
        "disclaimer": ("Diagnocat does not detect external cervical resorption; its findings are used for anatomy, numbering, "
                       "per-tooth context and triage. Resorption reasoning comes from this package's screen and conference, "
                       "which are uncalibrated suggestions requiring clinician confirmation. A negative screen does not exclude ECR."),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "patient_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        with open(os.path.join(out_dir, "patient_summary.md"), "w") as f:
            f.write(patient_markdown(report))
    return report


def patient_markdown(r: Dict[str, Any]) -> str:
    L = [f"# Patient-level ECR review (rev {r['patient_report_version']})", "", "> " + r["disclaimer"], "",
         f"Teeth examined: {r['teeth_examined'] or 'none'}; not examined (below priority cut): {r['teeth_not_examined'] or 'none'}", "",
         "## Triage order", "", "| tooth (FDI) | type | priority |", "|---|---|---|"]
    L += [f"| {t['iso_number']} | {t['tooth_type']} | {t['priority']} |" for t in r["triage_order"]]
    L += ["", "## Per-tooth outcome", "", "| tooth | seed source | status | leading | P(ECR) | Patel | plan primary |", "|---|---|---|---|---|---|---|"]
    for t in r["per_tooth"]:
        dx = t.get("suggested_diagnosis") or {}
        L.append(f"| {t['iso_number']} | {t.get('seed_source', '-')} | {t['status']} | {dx.get('leading', '-')} | "
                 f"{dx.get('p_ecr_uncalibrated', '-')} | {(t.get('suggested_patel') or {}).get('code', '-')} | "
                 f"{((t.get('suggested_plan') or {}).get('rule_recommendation') or {}).get('primary', '-')} |")
    L += ["", "## Diagnocat scope (cited)", ""]
    L += [f"- `{s['id']}` ({s['source']}): {s['text']} [{s['url']}]" for s in r["knowledge_scope"]]
    L += ["", "## Per-tooth conference summaries", ""]
    for t in r["per_tooth"]:
        if "report" in t:
            L += [f"### Tooth {t['iso_number']}", "", consensus_markdown(t["report"]).replace("# Case conference", "#### Case conference"), ""]
    return "\n".join(L) + "\n"
