"""Diagnocat ingestion for the ECR conference.

Diagnocat (cloud dental AI) processes the CBCT, segments teeth / pulp / jaws and reports per-tooth
conditions with probabilities.  It has **no external/cervical resorption category** (its 4.0 condition
list includes internal resorption, root caries, abfraction, periapical radiolucency, PDL widening,
bone loss, restorations, endodontic treatment ...), and an independent validation found no Diagnocat
output for external resorption cases.  So in this system Diagnocat is the *upstream anatomy and context
provider* and our screen + conference supply the resorption-specific reasoning:

    DICOM -> Diagnocat (segmentation, numbering, per-tooth conditions, clinician-added flags)
          -> DiagnocatStudy (this module: Partner API JSON | exported JSON | report PDF | STL)
          -> per-tooth ToothContext (prior shifts, clinical modifiers, canal hint, triage priority)
          -> ECR screen + multi-agent conference with KnowledgeAgent  -> diagnosis, Patel, plan, CDT codes

Partner API access is gated by Diagnocat (api-support@diagnocat.com); the client here is configurable
and falls back to files exported from the Diagnocat UI.
"""
from .schema import CONDITION_ALIASES, DiagnocatCondition, DiagnocatStudy, DiagnocatTooth
from .adapter import DiagnocatClient, load_diagnocat_json, load_diagnocat_pdf, stl_to_mask
from .context import ToothContext, build_contexts, prior_from_shifts

__all__ = ["CONDITION_ALIASES", "DiagnocatCondition", "DiagnocatStudy", "DiagnocatTooth", "DiagnocatClient",
           "load_diagnocat_json", "load_diagnocat_pdf", "stl_to_mask", "ToothContext", "build_contexts",
           "prior_from_shifts"]
