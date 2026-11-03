"""Perception layer: seeing, kept separate from reasoning.

Tools here identify teeth, propose suspicious regions, delineate structures, compute measurements, build
provenance-preserving multiplanar evidence and re-verify a region under alternative processing.  They
return numbers and masks with their coordinate transforms; they never produce a diagnosis.  The agent
layer (ecr_cds.agents) selects these tools, asks for more views, tracks coverage and drafts the reviewable
assessment.  Evaluate the two layers separately (see ecr_cds.evaluation).
"""
from .display import (apply_window, clahe2d, display_transform_record, metal_mask, tissue_reference_levels,
                      window_from_levels)
from .evidence import EvidenceFrame, Panel, build_evidence_frame
from .tools import PerceptionToolkit, ToolCall

__all__ = ["apply_window", "clahe2d", "display_transform_record", "metal_mask", "tissue_reference_levels",
           "window_from_levels", "EvidenceFrame", "Panel", "build_evidence_frame", "PerceptionToolkit", "ToolCall"]
