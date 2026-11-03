"""Reinforcement-learning layer: constrained contextual bandit over the
rule-derived safe set, with a hybrid reward and off-policy promotion gate.

Why a bandit (not full MDP) first
---------------------------------
ECR treatment is a one-shot decision per tooth with delayed outcome; the
"episode" is: context -> recommendation -> clinician decision -> outcome at
6/12/24 months. A constrained contextual bandit captures this with far less
data than deep RL, which matters because ECR series are small (Ravikumar et
al. trained on 3 patients; Jeng et al. 2024 had 42 teeth).

Reward design (adapted from OralGPT-Plus, arXiv 2603.06366)
-----------------------------------------------------------
R(τ) = R_rubric + R_format + R_diag

* R_rubric  continuous [0,1] partial credit: agreement between recommendation
            and the clinician's final decision, graded by a clinical distance
            matrix (e.g. external_repair vs external_repair_rct = small error;
            external_repair vs extraction = major error), plus the delayed
            outcome term when available (tooth survival / success).
* R_format  +0.1 if the recommendation was inside the safe set, carried a
            rationale, and irreversible actions were flagged for confirmation;
            -1.0 if any guardrail was violated (hard penalty).
* R_diag    conditioned reinspection reward: a bonus for triggering a
            re-measurement (zoom / re-slice along root axis / re-threshold)
            ONLY when (a) the classifier marked the case borderline and
            (b) the rubric confidence of the current hypothesis is > η. This
            is the mechanism the paper used to stop reward hacking (collapse-
            rebound of tool use). (H - Cu)_+ decays the bonus once the case
            has already been reinspected enough times.

Safety
------
* Action masking: the policy can only choose from the rule-derived safe set.
* Irreversible actions can never be auto-finalised; they are proposals.
* Policy updates are evaluated offline (IPS + doubly-robust) and must beat
  the deployed policy with a bootstrap lower bound before promotion.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .decision_matrix import SCHEME_PATEL, Treatment, Recommendation, ClinicalModifiers
from .patel_classifier import PatelParameters

ACTIONS: List[Treatment] = list(Treatment)
A_IDX = {a: i for i, a in enumerate(ACTIONS)}

# Invasiveness tier used for the rubric distance (0 = none ... 4 = tooth loss)
_TIER = {
    Treatment.MONITOR: 0,
    Treatment.EXTERNAL_REPAIR: 1,
    Treatment.EXTERNAL_REPAIR_VPT: 1,
    Treatment.EXTERNAL_REPAIR_RCT: 2,
    Treatment.INTERNAL_REPAIR_RCT: 2,
    Treatment.INTENTIONAL_REPLANT: 3,
    Treatment.HEMISECTION: 3,
    Treatment.DECORONATION: 4,
    Treatment.EXTRACTION: 4,
}


# --------------------------------------------------------------------------- #
@dataclass
class ClinicalContext:
    """Feature vector for the contextual model (all scaled roughly to [0,1]).

    Exactly one of `patel` (legacy Patel-indexed scheme) or `extent` (rev 2.9 measurement-based `ExtentSummary`)
    is given.  The eight lesion slots of the vector are filled from measured quantities in both cases; the two
    schemes use different cell keys so their tabular priors never mix.  Keep separate policy files per scheme
    (the linear part is shared by construction).
    """
    patel: Optional[PatelParameters] = None
    mods: ClinicalModifiers = field(default_factory=ClinicalModifiers)
    tooth_type: str = "unknown"      # 'anterior' | 'premolar' | 'molar'
    age_years: Optional[float] = None
    extent: Optional[object] = None  # ecr_cds.extent_summary.ExtentSummary

    def __post_init__(self):
        if (self.patel is None) == (self.extent is None):
            raise ValueError("ClinicalContext needs exactly one of patel= or extent=")

    @property
    def scheme(self) -> str:
        return SCHEME_PATEL if self.patel is not None else getattr(self.extent, "scheme", "measurement_v1")

    @property
    def borderline(self) -> bool:
        return bool(self.patel.borderline if self.patel is not None else self.extent.borderline)

    def _lesion_slots(self) -> List[float]:
        if self.patel is not None:
            p = self.patel
            return [(p.height - 1) / 3.0, "ABCD".index(p.spread) / 3.0, 1.0 if p.proximity == "p" else 0.0,
                    min(p.lesion_volume_mm3 / 200.0, 1.0), min(p.min_canal_distance_mm / 2.0, 1.0),
                    min(p.spread_deg / 360.0, 1.0), p.calcified_fraction, 1.0 if p.borderline else 0.0]
        e = self.extent
        ext_i = {"limited": 0.0, "moderate": 0.5, "extensive": 1.0}.get(e.extent_descriptor, 0.5)
        pulp_i = {"dentine_barrier_preserved": 0.0, "thin_dentine_barrier": 0.5, "canal_communication": 1.0}.get(e.pulp_relationship, 0.5)
        access_i = {"supracrestal_portal": 0.0, "shallow_subcrestal_portal": 0.33, "deep_subcrestal_portal": 1.0,
                    "no_surface_portal": 0.66}.get(e.access_descriptor, 0.5)
        return [min((e.long_axis_extent_mm or 0.0) / 15.0, 1.0), ext_i, pulp_i,
                min((e.lesion_volume_mm3 or 0.0) / 200.0, 1.0),
                0.5 if e.min_canal_distance_mm is None else min(e.min_canal_distance_mm / 2.0, 1.0),
                min((e.circumferential_extent_deg or 0.0) / 360.0, 1.0), access_i, 1.0 if e.borderline else 0.0]

    def vector(self) -> np.ndarray:
        m = self.mods

        def b(x: Optional[bool]) -> float:
            return 0.5 if x is None else float(x)

        return np.array([1.0, *self._lesion_slots(),
                         b(m.symptomatic), b(m.pulp_vital), b(m.restorable),
                         b(m.perio_support_adequate), b(m.multirooted),
                         b(m.surgically_accessible), b(m.persistent_bleeding),
                         b(m.growing_patient), b(m.aesthetic_zone),
                         1.0 if self.tooth_type == "anterior" else 0.0,
                         1.0 if self.tooth_type == "molar" else 0.0,
                         0.5 if self.age_years is None else min(self.age_years / 80.0, 1.0),
                         ], dtype=float)


@dataclass
class Decision:
    cell_key: str                        # Patel code (legacy) or measurement descriptor key
    chosen: Treatment
    ranking: List[Treatment]
    propensities: Dict[str, float]       # P(action) under the policy over the safe set
    safe_set: List[Treatment]
    requires_confirmation: bool
    reinspect_requested: bool
    rationale: List[str]
    cell_scheme: str = SCHEME_PATEL

    @property
    def patel_code(self) -> Optional[str]:
        return self.cell_key if self.cell_scheme == SCHEME_PATEL else None

    def to_dict(self) -> Dict:
        d = {
            "cell_key": self.cell_key,
            "cell_scheme": self.cell_scheme,
            "chosen": self.chosen.value,
            "ranking": [r.value for r in self.ranking],
            "propensities": self.propensities,
            "safe_set": [s.value for s in self.safe_set],
            "requires_confirmation": self.requires_confirmation,
            "reinspect_requested": self.reinspect_requested,
            "rationale": self.rationale,
        }
        if self.cell_scheme == SCHEME_PATEL:
            d["patel_code"] = self.cell_key
        return d


@dataclass
class Feedback:
    """Logged after the clinician decides and (later) when an outcome is observed.

    case_id      : stable per-tooth identifier (hashed), used to count independent cases
    event_type   : 'decision' | 'outcome' - derived automatically
    event_id     : idempotency key; re-submitting the same event never updates twice
    """
    clinician_choice: Treatment
    outcome_success: Optional[bool] = None   # at follow-up (e.g. 12 or 24 months)
    outcome_survival: Optional[bool] = None
    follow_up_months: Optional[float] = None
    reinspection_changed_code: Optional[bool] = None
    case_id: str = "unknown"
    event_id: Optional[str] = None

    def __post_init__(self):
        has_outcome = self.outcome_success is not None or self.outcome_survival is not None
        if has_outcome:
            if self.follow_up_months is None:
                raise ValueError("An outcome requires follow_up_months (observed recall time)")
            if not (0 < float(self.follow_up_months) <= 240):
                raise ValueError("follow_up_months must be in (0, 240]")
            if self.outcome_survival is False and self.outcome_success:
                raise ValueError("A lost tooth cannot be a success")
        elif self.follow_up_months is not None:
            raise ValueError("follow_up_months given without an observed outcome")
        if self.event_id is None:
            kind = "outcome" if has_outcome else "decision"
            tail = f"{kind}:{self.follow_up_months}" if has_outcome else kind
            self.event_id = f"{self.case_id}:{tail}"

    @property
    def event_type(self) -> str:
        return "outcome" if (self.outcome_success is not None or self.outcome_survival is not None) else "decision"


# --------------------------------------------------------------------------- #
def rubric_agreement(recommended: Treatment, chosen: Treatment) -> float:
    """Continuous partial credit in [0,1] based on invasiveness-tier distance."""
    if recommended == chosen:
        return 1.0
    d = abs(_TIER[recommended] - _TIER[chosen])
    # same tier (e.g. internal vs external repair + RCT) = minor error
    return {0: 0.75, 1: 0.5, 2: 0.25, 3: 0.1, 4: 0.0}[d]


def build_reward(decision: Decision, fb: Feedback, borderline: bool,
                 rubric_conf: float, reinspections_so_far: int,
                 eta: float = 0.6, alpha: float = 0.3, H: int = 2,
                 outcome_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Hybrid reward R = R_rubric + R_format + R_diag (OralGPT-Plus style)."""
    r_rubric = rubric_agreement(decision.chosen, fb.clinician_choice)
    if fb.outcome_success is not None:
        # delayed outcome term, discounted by follow-up horizon reliability
        horizon = 1.0 if (fb.follow_up_months or 0) >= 12 else 0.5
        r_rubric = 0.5 * r_rubric + 0.5 * outcome_weight * horizon * float(fb.outcome_success)
        if fb.outcome_survival is False:
            r_rubric -= 0.25  # tooth lost after an interventional recommendation

    # format / guardrail term
    in_safe = decision.chosen in decision.safe_set
    flagged = (not decision.chosen.irreversible) or decision.requires_confirmation
    has_rationale = len(decision.rationale) > 0
    r_format = 0.1 if (in_safe and flagged and has_rationale) else -1.0

    # conditioned reinspection reward (only when confident AND borderline)
    r_diag = 0.0
    if decision.reinspect_requested:
        gate = 1.0 if (rubric_conf > eta and borderline) else 0.0
        curiosity = max(H - reinspections_so_far, 0)
        useful = 1.0 if fb.reinspection_changed_code else 0.5
        r_diag = gate * alpha * curiosity * useful
        if not borderline:
            r_diag -= 0.1  # discourage indiscriminate tool use
    total = r_rubric + r_format + r_diag
    return total, {"rubric": r_rubric, "format": r_format, "diag": r_diag}


# --------------------------------------------------------------------------- #
class ConstrainedBanditPolicy:
    """Thompson-sampling bandit with a tabular prior per decision cell (Patel code in the legacy scheme,
    measurement descriptor key in rev 2.9) and a
    Bayesian linear model per action for generalisation across cells.

    score(a | x, cell) = w_tab * Beta_sample(cell, a) + (1 - w_tab) * θ_a·x
    restricted to the safe set. The rule-based primary gets a strong prior so
    the system starts as the guideline and only drifts with evidence.
    """

    def __init__(self, n_features: int = 21, prior_strength: float = 20.0,
                 w_tab: float = 0.6, lam: float = 1.0, noise: float = 0.25,
                 seed: Optional[int] = 0):
        self.rng = np.random.default_rng(seed)
        self.prior_strength = prior_strength
        self.w_tab = w_tab
        self.noise = noise
        self.n_features = n_features
        # tabular Beta(alpha, beta) per (cell, action)
        self.alpha: Dict[Tuple[str, str], float] = {}
        self.beta: Dict[Tuple[str, str], float] = {}
        # Bayesian linear regression per action: A = λI + Σ x xᵀ, b = Σ r x
        self.A = {a: lam * np.eye(n_features) for a in ACTIONS}
        self.b = {a: np.zeros(n_features) for a in ACTIONS}
        self.log: List[Dict] = []
        self.seen_events: set = set()

    @property
    def n_cases(self) -> int:
        return len({e.get("case_id", i) for i, e in enumerate(self.log)})

    # ---- prior from the rule matrix ----
    def _ensure_prior(self, rec: Recommendation):
        n = len(rec.safe_set)
        for rank, a in enumerate(rec.safe_set):
            key = (rec.cell_key, a.value)
            if key not in self.alpha:
                # primary: mean ≈ prognosis prior; alternatives decay with rank
                mean = rec.prognosis_prior * (0.85 ** rank)
                mean = min(max(mean, 0.05), 0.95)
                self.alpha[key] = self.prior_strength * mean
                self.beta[key] = self.prior_strength * (1 - mean)

    # ---- action selection ----
    def decide(self, ctx: ClinicalContext, rec: Recommendation,
               n_samples: int = 512) -> Decision:
        self._ensure_prior(rec)
        x = ctx.vector()
        safe = rec.safe_set
        scores = np.zeros((n_samples, len(safe)))
        for j, a in enumerate(safe):
            key = (rec.cell_key, a.value)
            tab = self.rng.beta(self.alpha[key], self.beta[key], size=n_samples)
            A_inv = np.linalg.inv(self.A[a])
            theta_mean = A_inv @ self.b[a]
            theta = self.rng.multivariate_normal(theta_mean, self.noise ** 2 * A_inv, size=n_samples)
            lin = theta @ x
            scores[:, j] = self.w_tab * tab + (1 - self.w_tab) * (0.5 + lin)  # centre linear term at 0.5
        best = scores.argmax(axis=1)
        counts = np.bincount(best, minlength=len(safe)).astype(float)
        props = counts / counts.sum()
        order = np.argsort(-props)
        ranking = [safe[i] for i in order]
        chosen = ranking[0]
        reinspect = bool(ctx.borderline)
        rationale = list(rec.rationale)
        # No feedback yet for this code (all posteriors at their priors): keep the guideline
        # order; Thompson noise alone must not re-rank treatments.
        n_obs = sum(self.alpha[(rec.cell_key, a.value)] + self.beta[(rec.cell_key, a.value)] - self.prior_strength for a in safe)
        if n_obs < 1.0:
            ranking = list(safe); chosen = rec.primary
            props = np.array([1.0 if a == rec.primary else 0.0 for a in safe]); order = np.arange(len(safe))
            rationale.append("RL policy: no feedback recorded for this code yet -> guideline default retained.")
        elif chosen != rec.primary:
            rationale.append(f"RL re-ranking: learned evidence favours {chosen.value} over "
                             f"guideline default {rec.primary.value} (P={props[order[0]]:.2f}).")
        if reinspect:
            rationale.append(("Borderline Patel parameters" if ctx.patel is not None else
                              "Measured descriptor(s) near a band edge") +
                             " -> reinspection (re-slice along root axis, re-threshold canal proximity) requested before finalising.")
        return Decision(
            cell_key=rec.cell_key, cell_scheme=rec.cell_scheme, chosen=chosen, ranking=ranking,
            propensities={safe[i].value: float(props[i]) for i in range(len(safe))},
            safe_set=safe, requires_confirmation=bool(chosen.irreversible or rec.requires_confirmation),
            reinspect_requested=reinspect, rationale=rationale,
        )

    # ---- learning ----
    def update(self, ctx: ClinicalContext, decision: Decision, fb: Feedback,
               reward: float, components: Dict[str, float]) -> bool:
        """Update posteriors with the hybrid reward (clipped to [0,1] for Beta).

        Returns False (and does nothing) if this event_id was already applied.
        """
        if fb.event_id in self.seen_events:
            return False
        self.seen_events.add(fb.event_id)
        x = ctx.vector()
        r01 = float(np.clip((reward + 1.0) / 2.4, 0.0, 1.0))  # map [-1, 1.4] -> [0,1]
        key = (decision.cell_key, decision.chosen.value)
        self.alpha[key] = self.alpha.get(key, 1.0) + r01
        self.beta[key] = self.beta.get(key, 1.0) + (1 - r01)
        a = decision.chosen
        self.A[a] += np.outer(x, x)
        self.b[a] += (r01 - 0.5) * x
        # Also credit the clinician's choice when it differed (supervised signal)
        if fb.clinician_choice != decision.chosen and fb.clinician_choice in decision.safe_set:
            k2 = (decision.cell_key, fb.clinician_choice.value)
            self.alpha[k2] = self.alpha.get(k2, 1.0) + 0.5
        self.log.append({
            "case_id": fb.case_id, "event_id": fb.event_id, "event_type": fb.event_type,
            "code": decision.cell_key, "scheme": decision.cell_scheme, "x": x.tolist(), "chosen": a.value,
            "propensity": decision.propensities[a.value],
            "clinician": fb.clinician_choice.value, "reward": reward,
            "components": components, "follow_up_months": fb.follow_up_months,
        })
        return True

    # ---- persistence ----
    def save(self, path: str):
        state = {
            "alpha": {f"{k[0]}|{k[1]}": v for k, v in self.alpha.items()},
            "beta": {f"{k[0]}|{k[1]}": v for k, v in self.beta.items()},
            "A": {a.value: self.A[a].tolist() for a in ACTIONS},
            "b": {a.value: self.b[a].tolist() for a in ACTIONS},
            "log": self.log,
            "seen_events": sorted(self.seen_events),
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str, **kw) -> "ConstrainedBanditPolicy":
        with open(path) as f:
            s = json.load(f)
        p = cls(**kw)
        p.alpha = {tuple(k.split("|")): v for k, v in s["alpha"].items()}  # type: ignore[misc]
        p.beta = {tuple(k.split("|")): v for k, v in s["beta"].items()}  # type: ignore[misc]
        p.A = {Treatment(a): np.array(m) for a, m in s["A"].items()}
        p.b = {Treatment(a): np.array(v) for a, v in s["b"].items()}
        p.log = s.get("log", [])
        p.seen_events = set(s.get("seen_events", [e.get("event_id") for e in p.log if e.get("event_id")]))
        return p


# --------------------------------------------------------------------------- #
# Off-policy evaluation and promotion gate
# --------------------------------------------------------------------------- #
def offline_evaluate(log: Sequence[Dict], target_props: Sequence[Dict[str, float]],
                     n_boot: int = 500, seed: int = 0) -> Dict[str, float]:
    """Doubly-robust estimate of a candidate policy's value on logged data.

    log[i]           : {'chosen', 'propensity', 'reward', ...} from the
                       behaviour (deployed) policy
    target_props[i]  : candidate policy's action distribution for context i
    Reward model is the per-action mean reward in the log (simple, robust).
    """
    rng = np.random.default_rng(seed)
    rewards = np.array([e["reward"] for e in log])
    chosen = [e["chosen"] for e in log]
    beh = np.array([max(e["propensity"], 1e-3) for e in log])
    # reward model q(a) = mean logged reward for a
    q: Dict[str, float] = {}
    for a in set(chosen):
        q[a] = float(np.mean([r for r, c in zip(rewards, chosen) if c == a]))
    dr = np.zeros(len(log))
    for i, e in enumerate(log):
        pi = target_props[i]
        direct = sum(pi.get(a, 0.0) * q.get(a, rewards.mean()) for a in pi)
        w = pi.get(chosen[i], 0.0) / beh[i]
        dr[i] = direct + w * (rewards[i] - q.get(chosen[i], rewards.mean()))
    # cluster bootstrap by case so repeated events from one tooth are not independent samples
    cases = np.array([str(e.get("case_id", i)) for i, e in enumerate(log)])
    uniq = np.unique(cases)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cases == c) for c in pick])
        boots.append(dr[idx].mean())
    return {
        "behaviour_value": float(rewards.mean()),
        "candidate_value_dr": float(dr.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_events": len(log),
        "n": int(len(uniq)),           # independent cases
    }


def split_log_by_case(log: Sequence[Dict], holdout_frac: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
    """Time-ordered split: the most recent `holdout_frac` of cases are held out.
    A candidate policy must be fitted on the first part and evaluated on the second."""
    order: List[str] = []
    for e in log:
        c = str(e.get("case_id", "unknown"))
        if c not in order:
            order.append(c)
    n_hold = max(1, int(round(holdout_frac * len(order)))) if order else 0
    hold = set(order[len(order) - n_hold:])
    return [e for e in log if str(e.get("case_id", "unknown")) not in hold], \
           [e for e in log if str(e.get("case_id", "unknown")) in hold]


def promotion_gate(eval_result: Dict[str, float], margin: float = 0.0, min_cases: int = 50) -> bool:
    """Promote only if the DR lower bound (on HELD-OUT cases) beats the deployed policy
    and at least `min_cases` independent cases were evaluated."""
    return eval_result["ci_low"] >= eval_result["behaviour_value"] - margin and eval_result["n"] >= min_cases
