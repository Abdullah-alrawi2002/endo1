"""Case conference orchestrator: runs the agents, moderates the debate, writes the consensus.

Output contract
---------------
`CaseConference.run()` returns a JSON-serialisable dict with

  status                  suggested_ecr_with_plan | ecr_possible_conditional_plan | no_ecr_suggested |
                          no_candidates | insufficient_evidence
  suggested_diagnosis     leading hypothesis, uncalibrated posterior, band, differential, evidence for/against
  suggested_extent        rev 2.9 (default): descriptive, measurement-based extent / severity summary of the accepted
                          candidate (None unless ECR suggested/possible).  No staging code.
  suggested_patel         LEGACY, only when CaseConference(staging_scheme="patel_3d"): Patel 3D code
  suggested_plan          safe set + modifiers + constrained policy, contingencies for missing inputs
  feature_bundles         per-candidate VLM visual features + voxel-geometry measurements with provenance
  clinician_actions_required
  conference              full blackboard (evidence trail + transcript)

Everything is a *suggestion*.  `requires_confirmation` is always True.  The
treating clinician confirms the diagnosis, reviews the measured extent and
supplies the clinical inputs before any plan is acted on.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .blackboard import Blackboard, Evidence
from .criteria import CRITERIA, HYPOTHESES, confidence_band, evaluate_criteria
from .llm_backend import LLMBackend, NullBackend
from .roles import (ContextAgent, CoverageAgent, KnowledgeAgent, VerifierAgent)
from .roles import (AnatomyAgent, ClassifierAgent, ClinicalCorrelationAgent, ImagingQAAgent, LesionScreeningAgent,
                    RadiologistAgent, SkepticAgent, TreatmentPlannerAgent)
from .feature_measurement import ExtentSummaryAgent, FeatureMeasurementAgent
from .clinical import ClinicalFindings
from ..decision_matrix import ClinicalModifiers
from ..pipeline import _quality_checks
from ..rl_policy import ConstrainedBanditPolicy
from ..perception.tools import PerceptionToolkit
from dataclasses import asdict

EVIDENCE_LEVEL = "multi_agent_suggestion_uncalibrated"
STAGING_MEASUREMENT = "measurement"   # rev 2.9 default: descriptive extent summary, no staging code
STAGING_PATEL_LEGACY = "patel_3d"     # legacy opt-in for reproducing rev <= 2.8 outputs

# When the conference does not favour ECR it still suggests how to resolve the leading alternative.
# These are generic clinical work-up steps, not treatment prescriptions.
NON_ECR_NEXT_STEPS = {
    "internal_resorption": ["Confirm the canal-centred, ballooning pattern on consecutive axial slices and check for a perforation",
                            "Internal inflammatory resorption is managed by root canal treatment to remove the resorbing tissue (ESE 2023 / AAE); perforating lesions may need repair",
                            "Clinical pulp testing and periapical status guide urgency"],
    "caries": ["Clinical and tactile examination of the cervical area; bitewing/periapical radiograph",
               "If caries is confirmed, manage per caries risk and restorability; re-screen for ECR if the defect is subgingival or undermining"],
    "noncarious_cervical_lesion": ["Clinical inspection for a smooth, hard, saucer/wedge-shaped defect on exposed root surface",
                                   "Monitor or restore according to sensitivity, aesthetics and progression; ECR unlikely if the defect is open and non-undermining"],
    "restoration_margin_or_burnout": ["Inspect the restoration margin clinically (explorer, floss, gingival status)",
                                      "Compare with a periapical radiograph at a different angle: cervical burnout changes with projection, a true defect does not",
                                      "Re-screen with reviewed CEJ/apex points if a defect persists after excluding the margin"],
    "anatomical_concavity_or_furcation": ["Correlate with root anatomy (developmental groove, furcation entrance) on the multiplanar views",
                                          "No lesion-directed treatment is suggested for a normal anatomical concavity"],
    "imaging_artifact": ["Review the raw slices for streak/beam-hardening artefacts from the nearby dense material",
                         "If the finding is clinically suspected, consider a repeat small-FOV scan with artefact reduction or a different projection"],
}


class CaseConference:
    def __init__(self, backend: Optional[LLMBackend] = None, max_rounds: int = 2,
                 suggest_threshold: float = 0.5, possible_threshold: float = 0.35,
                 radiologist_weights: Optional[np.ndarray] = None, radiologist_weights_version: str = "expert_prior_v1",
                 staging_scheme: str = STAGING_MEASUREMENT, vlm_features: bool = True,
                 feature_backend: Optional[LLMBackend] = None):
        """
        staging_scheme:  "measurement" (default, rev 2.9) - descriptive extent summary + measurement-indexed safe set;
                         "patel_3d" - legacy Patel classifier + 32-cell matrix (reproduces rev <= 2.8 reports only).
        vlm_features:    query the vision model for visual features (geometry measurements are always computed).
        feature_backend: backend for the feature-measurement agent (defaults to `backend`).
        """
        if staging_scheme not in (STAGING_MEASUREMENT, STAGING_PATEL_LEGACY):
            raise ValueError(f"staging_scheme must be '{STAGING_MEASUREMENT}' or '{STAGING_PATEL_LEGACY}'")
        self.backend = backend or NullBackend()
        self.max_rounds = max(1, int(max_rounds))
        self.suggest_threshold = suggest_threshold
        self.possible_threshold = possible_threshold
        self.staging_scheme = staging_scheme
        self.qa = ImagingQAAgent(self.backend)
        self.anatomy = AnatomyAgent(self.backend)
        self.screen = LesionScreeningAgent(self.backend)
        self.radiologist = RadiologistAgent(self.backend, radiologist_weights, radiologist_weights_version)
        self.skeptic = SkepticAgent(self.backend)
        self.classifier = ClassifierAgent(self.backend)
        self.planner = TreatmentPlannerAgent(self.backend)
        self.clinical = ClinicalCorrelationAgent(self.backend)
        # rev 2.9: VLM feature measurement (geometry always; model only when vlm_features) + extent summary
        self.features = FeatureMeasurementAgent((feature_backend or self.backend) if vlm_features else NullBackend())
        self.extent = ExtentSummaryAgent()
        self.vlm_features = bool(vlm_features)
        # rev 2.7 agent-layer roles: context intake, critic re-inspection, coverage tracking, knowledge retrieval
        self.context_agent = ContextAgent(self.backend)
        self.verifier = VerifierAgent(self.backend)
        self.coverage = CoverageAgent(self.backend)
        self.knowledge = KnowledgeAgent(self.backend)
        self.last_blackboard: Optional[Blackboard] = None

    # ------------------------------------------------------------------ #
    def run(self, volume, tooth_segmenter, seed_zyx, seed_source: str = "clinician", auto=None,
            canal_source: Optional[str] = None, cej_zyx=None, apex_zyx=None,
            modifiers: Optional[ClinicalModifiers] = None, policy: Optional[ConstrainedBanditPolicy] = None,
            tooth_type: str = "unknown", age_years: Optional[float] = None, case_id: Optional[str] = None,
            render_tiles: bool = True, clinical_findings: Optional[ClinicalFindings] = None,
            context=None, toolkit: Optional[PerceptionToolkit] = None, verify: bool = True,
            clahe_display: bool = False, verify_mode: str = "smooth", sharpen_display: bool = False,
            counterfactual_check: bool = False, examined_teeth: Optional[List[int]] = None,
            focus_teeth: Optional[List[int]] = None, buccal_direction=None) -> Dict:
        """Run the conference on one tooth.

        buccal_direction: optional patient buccal direction (z,y,x in voxel axes) so the feature bundle can label its
                 in-plane extents mesio-distal / bucco-lingual; without it the labels are a radial proxy and flagged.

        context: optional ToothContext (Diagnocat findings + chart) -> prior shifts, modifier / finding hints.
        toolkit: PerceptionToolkit whose call log is attached to the report (one per patient run).
        verify:  critic re-inspection of every candidate (reproducibility + multiplanar evidence frame).
        verify_mode: "smooth" (rev 2.7 frozen baseline) or "tta" (gain / noise / threshold-jitter / smoothing paths).
        sharpen_display: unsharp-mask the evidence frames shown to humans / the VLM (display only, recorded).
        counterfactual_check: also render a candidate-filled frame and re-query the vision model with it; an
                 opinion that does not change is down-weighted (perturbation-based grounding test).
        """
        case_id = case_id or volume.metadata.get("SeriesInstanceUID_hash", "case")
        bb = Blackboard(case_id)
        self.last_blackboard = bb   # arrays (candidate labels, crop) stay available to callers such as the dataset builder
        toolkit = toolkit or PerceptionToolkit()
        mods = modifiers or ClinicalModifiers()
        policy = policy or ConstrainedBanditPolicy()
        prior = None
        if context is not None:
            from ..diagnocat.context import prior_from_shifts
            c = context.to_dict() if hasattr(context, "to_dict") else dict(context)
            if c.get("prior_shifts"):
                prior = prior_from_shifts(c["prior_shifts"])
            # context fills only inputs the clinician left unset
            md = asdict(mods)
            for k, v in (c.get("modifier_hints") or {}).items():
                if k in md and md[k] is None:
                    md[k] = v
            mods = ClinicalModifiers(**md)
            if c.get("clinical_findings_hints"):
                fd = (clinical_findings or ClinicalFindings()).__dict__.copy()
                for k, v in c["clinical_findings_hints"].items():
                    if k in fd and fd[k] is None and v is not None:
                        fd[k] = v
                clinical_findings = ClinicalFindings(**fd)
            if tooth_type == "unknown" and c.get("tooth_type"):
                tooth_type = c["tooth_type"]
            if (canal_source is None or canal_source == "automatic") and c.get("canal_hint") == "hyperdense_filling":
                canal_source = "automatic"   # segmenter decides; the hint is recorded as evidence
        qa = _quality_checks(volume)
        self.qa.run(bb, volume, qa)
        self.context_agent.run(bb, context)
        self.anatomy.run(bb, volume, tooth_segmenter, seed_zyx, seed_source, auto=auto, canal_source=canal_source or "automatic")
        if not canal_source or canal_source == "automatic":   # segmenter learns its canal mode when first called
            canal_source = str(getattr(tooth_segmenter, "canal_mode", None) or "automatic")
        crop = bb.arrays["crop"]
        off = np.array([s.start for s in crop])
        cands = self.screen.run(bb, volume.spacing,
                                cej_zyx=None if cej_zyx is None else tuple(np.asarray(cej_zyx) - off),
                                apex_zyx=None if apex_zyx is None else tuple(np.asarray(apex_zyx) - off),
                                canal_source=canal_source, render=render_tiles)
        descs = bb.arrays["descriptors"]
        # ---- critic re-inspection (perception tools, alternative processing path, MPR evidence) ----
        if verify:
            full_slice = None
            for cid, d in descs.items():
                try:
                    cz = int(np.argwhere(bb.arrays["candidate_labels"] == cid)[:, 0].mean() + off[0])
                    full_slice = volume.array[cz]
                except Exception:
                    full_slice = None
                self.verifier.run(bb, toolkit, d, volume.spacing, source_id=case_id, full_slice=full_slice, clahe=clahe_display,
                                  mode=verify_mode, sharpen=sharpen_display,
                                  counterfactual=counterfactual_check and getattr(self.backend, "supports_images", False))
        # ---- rev 2.9: feature measurement (voxel geometry + VLM visual reading) per candidate -------------------
        self.features.buccal_direction = buccal_direction
        for cid, d in descs.items():
            try:
                self.features.run(bb, d, volume.spacing, source_id=case_id)
            except Exception as e:  # measurement must never break the conference
                bb.say(self.features.name, f"candidate {cid}: feature measurement failed ({type(e).__name__}: {str(e)[:120]})")
        self.screen.reconcile(bb)
        # ---- debate rounds per candidate ---------------------------------
        finals: Dict[int, Evidence] = {}
        for cid, d in descs.items():
            ver = bb.latest("verification", refers_to=cid)
            challenges: List[Dict] = list(ver.content["challenges"]) if ver else []
            diff = None
            for r in range(self.max_rounds):
                diff = self.radiologist.assess(bb, d, rnd=r, challenges=challenges, prior=prior)
                ch = self.skeptic.challenge(bb, d, diff, rnd=r)
                new = [c for c in ch.content["challenges"] if c not in challenges]
                challenges = ch.content["challenges"]
                if not new:          # skeptic has nothing to add: the current assessment already answers all challenges
                    break
            else:                    # rounds exhausted with fresh challenges outstanding: one closing reply
                diff = self.radiologist.assess(bb, d, rnd=self.max_rounds, challenges=challenges, prior=prior)
            finals[cid] = diff
        # ---- moderation -----------------------------------------------------
        # ---- Q8: clinical correlation on every candidate --------------------
        clin: Dict[int, Evidence] = {}
        for cid, ev in finals.items():
            clin[cid] = self.clinical.run(bb, cid, ev.content["posterior"], clinical_findings)
        consensus = self._moderate(bb, finals, descs, volume.spacing, mods, policy, tooth_type, age_years, clin)
        consensus["clinical_findings"] = (clinical_findings or ClinicalFindings()).to_dict()
        # ---- coverage / completeness and knowledge retrieval (interpretation only) ----
        cov = self.coverage.run(bb, context, clinical_findings, examined_teeth=examined_teeth, focus_teeth=focus_teeth,
                                tool_log=toolkit.log_dicts())
        kn = self.knowledge.run(bb, consensus, tooth_type)
        consensus["coverage"] = cov.content
        consensus["knowledge"] = kn.content
        consensus["context"] = (context.to_dict() if hasattr(context, "to_dict") else context) if context is not None else None
        consensus["prior_used"] = prior or "default"
        consensus["modifiers_used"] = asdict(mods)
        consensus["verification"] = {cid: {k: v for k, v in e.content.items() if k != "evidence_frame"}
                                     for cid, e in ((e.refers_to, e) for e in bb.query(kind="verification"))}
        consensus["evidence_frames"] = {e.refers_to: e.content.get("evidence_frame") for e in bb.query(kind="verification")}
        consensus["feature_bundles"] = {e.refers_to: e.content for e in bb.query(kind="feature_bundle")}
        consensus["screen_reconciliation"] = (bb.latest("screen_reconciliation").content if bb.latest("screen_reconciliation") else None)
        consensus["tool_log"] = toolkit.log_dicts()
        consensus["architecture"] = {"perception_layer": "ecr_cds.perception (segment, propose, delineate, measure, reformat, re-verify, "
                                                         "VLM feature measurement grounded in voxel geometry)",
                                     "agent_layer": "ecr_cds.agents (context, QA, anatomy, screening, verifier, feature measurement, radiologist, "
                                                    "skeptic, clinical correlation, "
                                                    + ("extent summary" if self.staging_scheme == STAGING_MEASUREMENT else "Patel classifier [legacy]")
                                                    + ", planner, coverage, knowledge, moderator)",
                                     "rule": "knowledge and context interpret established findings; they never create findings; "
                                             "model visual readings never override voxel measurements",
                                     "settings": {"verify": bool(verify), "verify_mode": verify_mode, "clahe_display": bool(clahe_display),
                                                  "sharpen_display": bool(sharpen_display), "counterfactual_check": bool(counterfactual_check),
                                                  "backend": getattr(self.backend, "name", "none"),
                                                  "staging_scheme": self.staging_scheme, "vlm_features": self.vlm_features,
                                                  "feature_backend": getattr(self.features.backend, "name", "none")}}
        consensus.update(case_id=case_id, qa=qa, acquisition=volume.metadata,
                         roi_bbox_zyx=[[int(s.start), int(s.stop)] for s in crop],
                         evidence_level=EVIDENCE_LEVEL, requires_confirmation=True, clinical_validation=False,
                         backend=self.backend.name, rounds=self.max_rounds,
                         backend_status=("none" if isinstance(self.backend, NullBackend) else
                                         "used" if bb.arrays.get("backend_failures", 0) == 0 else
                                         "partially_failed_fell_back_to_rules" if bb.arrays.get("backend_calls_ok", 0) else
                                         "failed_fell_back_to_rules"),
                         backend_calls={"ok": bb.arrays.get("backend_calls_ok", 0), "failed": bb.arrays.get("backend_failures", 0)},
                         conference=bb.to_dict(),
                         disclaimer=("Multi-agent suggestion from automatic segmentation and expert-set criteria. "
                                     "Posteriors are not calibrated on clinical data. Not a diagnosis; not a "
                                     "treatment decision. A negative result does not exclude ECR."))
        # volume-frame coordinates for every candidate
        for c in consensus.get("candidates", []):
            d = descs[c["component_id"]]
            cen = np.argwhere(bb.arrays["candidate_labels"] == c["component_id"]).mean(0) + off
            c["centroid_volume_zyx"] = cen.round(1).tolist()
        return consensus

    # ------------------------------------------------------------------ #
    def _moderate(self, bb, finals, descs, spacing, mods, policy, tooth_type, age_years, clin=None) -> Dict:
        clin = clin or {}
        legacy = self.staging_scheme == STAGING_PATEL_LEGACY
        actions = ["Review the automatic tooth / canal masks and the candidate tiles.",
                   "Mark the CEJ and apex (CEJ_ZYX / APEX_ZYX) so the cervical band and the CEJ-referenced extents are measured, not assumed.",
                   "Confirm or reject the suggested diagnosis clinically (probing, bleeding, pulp tests, history).",
                   ("Enter reviewed Patel components and clinical inputs in Section 5 before using any plan." if legacy else
                    "Review the measured extent summary (and any model values flagged as unverified) and enter the clinical inputs before using any plan.")]
        stage_key = "suggested_patel" if legacy else "suggested_extent"
        cand_rows = []
        for cid, ev in finals.items():
            d = descs[cid]
            ch = bb.latest("challenge", refers_to=cid)
            post = clin[cid].content["adjusted_posterior"] if cid in clin else ev.content["posterior"]
            leading_c = max(post, key=post.get)
            capped = ev.content["band_capped_by_unverified_anatomy"]
            cand_rows.append({"component_id": cid, "volume_mm3": d.volume_mm3, "leading": leading_c,
                              "p_ecr": post["ECR"], "band": confidence_band(post["ECR"], capped),
                              "posterior": post, "imaging_posterior": ev.content["posterior"],
                              "diagnostic_questions": ev.content["diagnostic_questions"],
                              "clinical_correlation": clin[cid].content["verdict"] if cid in clin else "not_assessed",
                              "criteria_for_ecr": ev.content["criteria_for_ecr"],
                              "criteria_against_ecr": ev.content["criteria_against_ecr"],
                              "challenges": ch.content["challenges"] if ch else [],
                              "llm_notes": ev.content.get("llm_notes"), "llm_posterior": ev.content.get("llm_posterior"),
                              "llm_weight": ev.content.get("llm_weight", 0.0), "llm_meta": ev.content.get("llm_meta")})
        if not cand_rows:
            bb.say("moderator", "No candidate reached the conference; this is not an exclusion of ECR.")
            return {"status": "no_candidates", "candidates": [], "suggested_diagnosis": None,
                    stage_key: None, "suggested_plan": None, "clinician_actions_required": actions[:3]}
        best = max(cand_rows, key=lambda r: r["p_ecr"])
        p = best["p_ecr"]
        leading = best["leading"]
        dx = {"candidate_id": best["component_id"], "leading": leading, "p_ecr_uncalibrated": p,
              "band": best["band"], "differential": sorted(best["posterior"].items(), key=lambda kv: -kv[1]),
              "diagnostic_questions": best["diagnostic_questions"],
              "clinical_correlation": best["clinical_correlation"],
              "supporting_criteria": best["criteria_for_ecr"], "opposing_criteria": best["criteria_against_ecr"],
              "unresolved_challenges": best["challenges"], "weights_version": self.radiologist.weights_version,
              "basis": finals[best["component_id"]].basis}
        suggested_stage, plan = None, None
        if leading == "ECR" and p >= self.suggest_threshold:
            status = "suggested_ecr_with_plan"
            dx["statement"] = (f"Appearance most consistent with external cervical resorption (uncalibrated P={p:.2f}, "
                               f"{best['band']} confidence); differential: " +
                               ", ".join(f"{h} {v:.2f}" for h, v in dx["differential"][1:3]))
        elif p >= self.possible_threshold:
            status = "ecr_possible_conditional_plan"
            dx["statement"] = (f"ECR possible but not favoured over {leading} (P(ECR)={p:.2f}); plan below applies "
                               f"only if ECR is confirmed clinically.")
        else:
            status = "no_ecr_suggested"
            dx["statement"] = (f"Leading hypothesis {leading} (P={best['posterior'][leading]:.2f}); ECR not suggested "
                               f"for any candidate (max P(ECR)={p:.2f}). This does not exclude ECR.")
            dx["suggested_next_steps"] = NON_ECR_NEXT_STEPS.get(leading, [])
        if status in ("suggested_ecr_with_plan", "ecr_possible_conditional_plan"):
            if legacy:
                pe = self.classifier.run(bb, best["component_id"], spacing)
                reviewed = "clinician-reviewed Patel components"
            else:
                if best["component_id"] not in bb.arrays.get("feature_bundles", {}):
                    # feature measurement failed for this candidate: retry geometry-only so the summary still exists
                    try:
                        backend, self.features.backend = self.features.backend, None
                        self.features.run(bb, descs[best["component_id"]], spacing, source_id=bb.case_id)
                    except Exception as e:
                        bb.say("moderator", f"candidate {best['component_id']}: extent measurement unavailable ({type(e).__name__}); "
                                            "no plan can be keyed without measurements.")
                    finally:
                        self.features.backend = backend
                pe = (self.extent.run(bb, best["component_id"])
                      if best["component_id"] in bb.arrays.get("feature_bundles", {}) else None)
                reviewed = "clinician-reviewed extent measurements, clinical confirmation of ECR"
            if pe is None:
                status = "ecr_suggested_measurements_unavailable"
                actions.append("Measurements could not be computed for the leading candidate; review the masks and re-run before any plan.")
            else:
                suggested_stage = pe.content
                pl = self.planner.run(bb, mods, policy, tooth_type, age_years)
                plan = pl.content
                plan["conditional_on"] = (f"clinical confirmation of ECR and {reviewed}"
                                          if status == "ecr_possible_conditional_plan" else
                                          f"{reviewed} and clinical inputs")
                if plan["irreversible_options_in_safe_set"]:
                    actions.append("Irreversible options are in the safe set: explicit confirmation required before any is chosen.")
        bb.say("moderator", f"{status}: {dx['statement']}")
        return {"status": status, "candidates": cand_rows, "suggested_diagnosis": dx,
                stage_key: suggested_stage, "suggested_plan": plan, "clinician_actions_required": actions}

    # ------------------------------------------------------------------ #
    def save_outputs(self, consensus: Dict, out_dir: str, volume=None) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        p = os.path.join(out_dir, "conference_report.json")
        with open(p, "w") as f:
            json.dump(consensus, f, indent=2, default=str)
        paths.append(p)
        md = os.path.join(out_dir, "conference_summary.md")
        with open(md, "w") as f:
            f.write(consensus_markdown(consensus))
        paths.append(md)
        bb = self.last_blackboard
        if bb is not None:
            for cid, png in bb.arrays.get("tiles", {}).items():
                tp = os.path.join(out_dir, f"candidate_{cid}_tiles.png")
                with open(tp, "wb") as f:
                    f.write(png)
                paths.append(tp)
            for cid, png in bb.arrays.get("frames_png", {}).items():
                tp = os.path.join(out_dir, f"candidate_{cid}_evidence_frame.png")
                with open(tp, "wb") as f:
                    f.write(png)
                paths.append(tp)
            if volume is not None and "candidate_labels" in bb.arrays:
                from ..auto_segment import save_labelmap_nifti
                full = np.zeros(volume.shape, np.uint16)
                full[bb.arrays["crop"]] = bb.arrays["candidate_labels"]
                lp = os.path.join(out_dir, "conference_candidates.nii.gz")
                save_labelmap_nifti(full, volume, lp)
                paths.append(lp)
        return paths


# --------------------------------------------------------------------------- #
def consensus_markdown(c: Dict) -> str:
    L = [f"# Case conference - {c.get('case_id')}", "",
         f"**Status:** `{c['status']}`  |  evidence level: `{c.get('evidence_level')}`  |  backend: `{c.get('backend')}` ({c.get('backend_status')})", "",
         "> " + c.get("disclaimer", ""), ""]
    dx = c.get("suggested_diagnosis")
    if dx:
        L += ["## Suggested diagnosis", "", dx["statement"], "",
              "| hypothesis | uncalibrated posterior |", "|---|---|"]
        L += [f"| {h} | {v:.2f} |" for h, v in dx["differential"]]
        L += ["", f"Supporting criteria: {', '.join(dx['supporting_criteria']) or 'none'}  ",
              f"Opposing criteria: {', '.join(dx['opposing_criteria']) or 'none'}  ",
              f"Criteria weights: `{dx['weights_version']}` (basis: {dx['basis']})", ""]
        if dx.get("diagnostic_questions"):
            L += ["### Diagnostic questions (answered from the scan)", "", "| # | question | answer | evidence |", "|---|---|---|---|"]
            L += [f"| {q['id']} | {q['question']} | **{q['answer']}** | {q['evidence']} |" for q in dx["diagnostic_questions"]]
            L += [f"| Q8 | Does the clinical picture fit? | **{dx.get('clinical_correlation', 'not_assessed')}** | "
                  f"{', '.join(f'{k}={v}' for k, v in c.get('clinical_findings', {}).items()) or 'no findings entered'} |", ""]
        if dx["unresolved_challenges"]:
            L += ["Skeptic challenges:"] + [f"- ({ch['severity']}) {ch['issue']}" for ch in dx["unresolved_challenges"]] + [""]
        if dx.get("suggested_next_steps"):
            L += ["Suggested work-up for the leading alternative:"] + [f"- {x}" for x in dx["suggested_next_steps"]] + [""]
    se = c.get("suggested_extent")
    if se:
        L += ["## Measured extent summary (descriptive, not a staging code)", "", se.get("narrative", ""), "",
              f"- Extent descriptor: **{se['extent_descriptor']}** (score {se.get('extent_score')}/8)",
              f"- Pulp relationship: **{se['pulp_relationship']}**",
              f"- Access descriptor: **{se['access_descriptor']}**",
              f"- Measurement confidence: {se.get('measurement_confidence', 'n/a')}; descriptors near a band edge: "
              f"{', '.join(se.get('near_boundary') or []) or 'none'}; unavailable measurements: {', '.join(se.get('unavailable') or []) or 'none'}", ""]
    fbs = c.get("feature_bundles") or {}
    if fbs:
        L += ["## Feature measurements (VLM visual features + voxel geometry)", ""]
        for cid, b in fbs.items():
            vlm = b.get("vlm") or {}
            vis = b.get("visual") or {}
            L += [f"### Candidate {cid}", "",
                  f"VLM status: `{vlm.get('status', 'n/a')}` ({vlm.get('backend', 'none')}); visual features assessed: {vis.get('assessed', False)}", ""]
            if vis.get("assessed"):
                L += ["| visual feature | value | VLM confidence |", "|---|---|---|"]
                L += [f"| {k} | {vis.get(k)} | {(vis.get('confidence') or {}).get(k, 'n/a')} |"
                      for k in vis if k not in ("assessed", "confidence", "free_text")]
                if vis.get("free_text"):
                    L += ["", f"Model description: {vis['free_text']}"]
                L += [""]
            ms = b.get("measurements") or {}
            if ms:
                L += ["| measurement | value | unit | source | verified | confidence | VLM value | agreement |", "|---|---|---|---|---|---|---|---|"]
                for k, m in ms.items():
                    v = m.get("value")
                    vs = f"{v:.2f}" if isinstance(v, float) else str(v)
                    L.append(f"| {k} | {vs} | {m.get('unit')} | {m.get('source')} | {m.get('verified')} | "
                             f"{m.get('confidence', 0):.2f} | {m.get('vlm_value', '-')} | {m.get('vlm_agreement', '-')} |")
                L += [""]
            if b.get("unverified_vlm_values"):
                L += ["Unverifiable VLM values (not geometrically checkable, **not used** for planning): " + ", ".join(b["unverified_vlm_values"]), ""]
            if b.get("vlm_geometry_disagreements"):
                L += ["VLM / geometry disagreements (geometry wins):"] + [f"- {d}" for d in b["vlm_geometry_disagreements"]] + [""]
        L += ["The model never sees the measured numbers; every value it reports is checked against the voxel geometry, and values "
              "that cannot be checked are flagged and excluded from the extent summary and plan.", ""]
    sp = c.get("suggested_patel")
    if sp:
        L += ["## Suggested Patel 3D classification (LEGACY scheme, opt-in)", "",
              f"**{sp['code']}** - apical extent {sp['apical_extent_mm']:.1f} mm from CEJ, spread {sp['spread_deg']:.0f} deg, "
              f"canal distance {sp['min_canal_distance_mm']:.2f} mm, volume {sp['lesion_volume_mm3']:.1f} mm3, "
              f"borderline={sp['borderline']} (height margin {sp['height_margin_mm']:.2f} mm, spread margin "
              f"{sp['spread_margin_deg']:.0f} deg, proximity margin {sp['proximity_margin_mm']:.2f} mm)", ""]
    pl = c.get("suggested_plan")
    if pl:
        rr, pd = pl["rule_recommendation"], pl["policy_decision"]
        L += ["## Suggested treatment plan", "", f"Conditional on: {pl['conditional_on']}", "",
              f"- Matrix primary: **{rr['primary']}** (evidence {rr['evidence']}, prognosis prior {rr['prognosis_prior']})",
              f"- Alternatives: {', '.join(rr['alternatives'])}",
              f"- Policy ranking ({pl['policy_feedback_cases']} feedback cases): {' > '.join(pd['ranking'])}",
              f"- Missing clinical inputs: {', '.join(rr['missing_inputs']) or 'none'}"]
        if pl["contingencies"]:
            L += ["", "Contingencies:"] + [f"- if `{k['input']}` is true -> {k['if_true']}; if false -> {k['if_false']}"
                                           for k in pl["contingencies"]]
        if pl.get("end_stage"):
            L += ["", "End-stage code: see `end_stage_matrix` in the JSON report."]
        if pl.get("escalation_considerations"):
            L += ["", "Escalation considerations (measurement-indexed):"] + [f"- {x}" for x in pl["escalation_considerations"]]
        if rr.get("requires_confirmation"):
            L += ["", "**Irreversible options are present in the safe set: they require explicit clinician confirmation and are never "
                      "selected by the software.**"]
        L += ["", f"Follow-up: {pl['follow_up']}", ""]
        L += ["Rationale:"] + [f"- {r}" for r in rr["rationale"]] + [""]
        if pl.get("cdt_codes"):
            L += ["CDT 2024 codes (AAE guide; confirm with payer):"]
            for opt, codes in pl["cdt_codes"].items():
                L += [f"- {opt}: " + "; ".join(f"{x['code']} {x['descriptor']}" for x in codes)]
            L += [""]
    ctx = c.get("context")
    if ctx:
        L += ["## Upstream context (Diagnocat + chart)", "",
              f"Tooth {ctx.get('iso_number')} ({ctx.get('tooth_type')}), triage priority {ctx.get('priority')}. "
              f"Prior shifts: {ctx.get('prior_shifts') or 'none'}. Modifier hints: {ctx.get('modifier_hints') or 'none'}.",
              "Diagnocat conditions: " + (", ".join(f"{d['name']} (p={d['probability']}, {d['source']})" for d in ctx.get("diagnocat_conditions", [])) or "none"),
              "Predisposing factors: " + (", ".join(ctx.get("predisposing_factors", [])) or "none"), ""]
        if ctx.get("notes"):
            L += [f"- {n}" for n in ctx["notes"]] + [""]
    ver = c.get("verification")
    if ver:
        L += ["## Critic re-inspection (perception layer)", "", "| candidate | verdict | reproducibility (Dice) | axis extent mm |", "|---|---|---|---|"]
        L += [f"| {cid} | {v['verdict']} | {v['reproducibility']:.2f} | {(v.get('axis_persistence') or {}).get('long_axis_extent_mm', 'n/a')} |"
              for cid, v in ver.items()]
        L += ["", "Evidence frames (linked axial neighbours, coronal, sagittal, tooth-axis reformats, full-slice context) are saved as "
                  "`candidate_<id>_evidence_frame.png` with voxel provenance in the JSON report.", ""]
    rows = [r for r in c.get("candidates", []) if r.get("llm_meta")]
    if rows:
        L += ["## Vision-language second opinion (local model)", "",
              "| candidate | backend | weight in fusion | semantic entropy (clusters/samples) | abstained | grounding test | model P(ECR) |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            m = r["llm_meta"]; dse = m.get("dse") or {}; g = m.get("grounding") or {}
            be = m.get("backend"); be = be.get("name") if isinstance(be, dict) else be
            L.append(f"| {r['component_id']} | {be} | {m.get('weight', 0):.2f} | "
                     f"{dse.get('dse', 'n/a')} ({dse.get('n_clusters', '?')}/{dse.get('n_samples', '?')}) | {'yes' if m.get('abstained') else 'no'} | "
                     f"{g.get('verdict', 'not_run')} | {(r.get('llm_posterior') or {}).get('ECR', 'n/a')} |")
        L += ["", "The model reads the evidence frame only; it is weighted by sample agreement (discrete semantic entropy), set to zero "
                  "when samples disagree, and reduced when its answer does not change after the candidate is digitally filled in. "
                  "It never replaces the measured rule posterior.", ""]
    cov = c.get("coverage")
    if cov:
        L += ["## Coverage and missing prerequisites", "", f"Checklist completeness {cov['completeness']:.2f}; tools used: {', '.join(cov['tools_used']) or 'none'}", ""]
        L += [f"- **{m['item']}**: {m['why']} - {m['how_to_resolve']}" for m in cov["missing_prerequisites"]] + [""]
    kn = c.get("knowledge")
    if kn and kn.get("statements"):
        L += ["## Guideline statements cited", ""]
        L += [f"- `{sid}` ({st['source']}): {st['text']} {('[' + st['url'] + ']') if st.get('url') else ''}" for sid, st in kn["statements"].items()] + [""]
    L += ["## Clinician actions required", ""] + [f"- {a}" for a in c.get("clinician_actions_required", [])] + [""]
    tr = c.get("conference", {}).get("transcript", [])
    if tr:
        L += ["## Conference transcript", ""] + [f"- **{t['agent']}** (round {t['round']}): {t['text']}" for t in tr]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def record_diagnosis_feedback(log_path: str, consensus: Dict, confirmed_diagnosis: str,
                              radiologist: Optional[RadiologistAgent] = None, min_cases: int = 20) -> Dict:
    """Append the clinician-confirmed diagnosis for the conference's leading candidate and, when enough
    independent cases exist, let the radiologist agent refine its ECR criterion weights."""
    if confirmed_diagnosis not in HYPOTHESES:
        raise ValueError(f"confirmed_diagnosis must be one of {HYPOTHESES}")
    dx = consensus.get("suggested_diagnosis")
    if not dx:
        raise ValueError("Consensus has no candidate to label")
    cid = dx["candidate_id"]
    cand = next(e for e in consensus["conference"]["evidence"] if e["kind"] == "candidate" and e["refers_to"] == cid)
    from .visual_features import CandidateDescriptors
    fields = {k: v for k, v in cand["content"].items() if k in CandidateDescriptors.__dataclass_fields__}
    truth = evaluate_criteria(CandidateDescriptors(**fields))
    log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    key = f"{consensus.get('case_id')}:{cid}"
    log = [e for e in log if e.get("key") != key]  # idempotent per case/candidate
    log.append({"key": key, "case_id": consensus.get("case_id"), "component_id": cid, "criteria": truth,
                "ecr": confirmed_diagnosis == "ECR", "confirmed_diagnosis": confirmed_diagnosis,
                "p_ecr_at_suggestion": dx["p_ecr_uncalibrated"]})
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    result = {"logged": True, "n_labelled": len(log), "n_cases": len({e["case_id"] for e in log})}
    if radiologist is not None:
        per_case = {e["case_id"]: e for e in log}  # one label per case
        result["weights"] = radiologist.update_weights_from_feedback(list(per_case.values()), min_cases=min_cases)
    return result
