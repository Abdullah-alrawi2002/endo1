"""ECR-CDS: Clinical decision support for external cervical resorption (ECR)
treatment planning from CBCT DICOM volumes.

Pipeline
--------
DICOM series (≈500 slices) -> volume + geometry -> preprocessing ->
tooth / canal / lesion segmentation -> Patel 3D parameters (height 1-4,
circumferential spread A-D, proximity d/p) -> rule-based decision matrix
(safe action set) -> RL policy (contextual bandit, constrained to safe set)
-> ranked recommendation + rationale -> clinician review -> feedback/outcome
-> policy update (offline-evaluated before promotion).

All recommendations are advisory. Irreversible actions (extraction,
decoronation, intentional replantation, hemisection) always require explicit
clinician confirmation and are never auto-finalised.
"""

from .dicom_loader import load_dicom_series, CBCTVolume
from .preprocess import preprocess_volume
from .patel_classifier import PatelParameters, classify_patel
from .decision_matrix import (
    Treatment,
    recommend_from_matrix,
    end_stage_matrix,
    DECISION_MATRIX,
)
from .rl_policy import ConstrainedBanditPolicy, ClinicalContext, build_reward
from .pipeline import run_pipeline

__all__ = [
    "load_dicom_series",
    "CBCTVolume",
    "preprocess_volume",
    "PatelParameters",
    "classify_patel",
    "Treatment",
    "recommend_from_matrix",
    "end_stage_matrix",
    "DECISION_MATRIX",
    "ConstrainedBanditPolicy",
    "ClinicalContext",
    "build_reward",
    "run_pipeline",
]
