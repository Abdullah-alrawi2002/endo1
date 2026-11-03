"""ECR-CDS: Clinical decision support for external cervical resorption (ECR)
treatment planning from CBCT DICOM volumes.

Pipeline
--------
DICOM series (≈500 slices) -> volume + geometry -> preprocessing ->
tooth / canal / lesion segmentation -> VLM visual features + voxel-geometry
measurements (ecr_cds.perception.vlm_features) -> descriptive extent summary
(ecr_cds.extent_summary; no staging codes) -> measurement-indexed decision
mapping (safe action set) -> RL policy (contextual bandit, constrained to safe set)
-> ranked recommendation + rationale -> clinician review -> feedback/outcome
-> policy update (offline-evaluated before promotion).

All recommendations are advisory. Irreversible actions (extraction,
decoronation, intentional replantation, hemisection) always require explicit
clinician confirmation and are never auto-finalised.

The single-tooth `run_pipeline` and the legacy `classify_patel` / `recommend_from_matrix`
path (rev <= 2.8 outputs) are retained for reproducibility; the conference uses the
measurement-based path by default (docs/VLM_FEATURE_MEASUREMENT.md).
"""

from .dicom_loader import load_dicom_series, CBCTVolume
from .preprocess import preprocess_volume
from .patel_classifier import PatelParameters, classify_patel
from .decision_matrix import (
    Treatment,
    recommend_from_matrix,
    recommend_from_measurements,
    end_stage_matrix,
    DECISION_MATRIX,
)
from .extent_summary import ExtentSummary, summarise_extent
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
    "recommend_from_measurements",
    "ExtentSummary",
    "summarise_extent",
    "end_stage_matrix",
    "DECISION_MATRIX",
    "ConstrainedBanditPolicy",
    "ClinicalContext",
    "build_reward",
    "run_pipeline",
]
