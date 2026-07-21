from typing import Literal

from pydantic import BaseModel, Field


class VisionScreen(BaseModel):
    mimicsConsidered: list[str] = Field(default_factory=list)
    morphologyNotes: str
    confidence: Literal["low", "moderate", "high"]


class DensityInfo(BaseModel):
    """Same-scan relative density observation (not calibrated HU on CBCT)."""

    apexMean: float | None
    referenceMean: float | None
    unit: Literal["GV", "HU"]
    method: str
    valid: bool


class CTSupport(BaseModel):
    """Non-diagnostic CBCT research support payload."""

    status: Literal["completed", "partial", "failed"]
    targetToothUniversal: int
    clinicianSeedProvided: bool = False
    qualityGatePassed: bool
    candidateLowAttenuationRegion: bool | None = None
    candidateLocation: str | None = None
    candidateVolumeMm3: float | None = None
    relativeAttenuationDropPercent: float | None = None
    artifactWarnings: list[str] = Field(default_factory=list)
    qualityGateFailures: list[str] = Field(default_factory=list)
    morphologyNotes: str | None = None
    vertucciScreen: str | None = None
    canalLengthEstimateMm: float | None = None
    canalLengthEstimateNote: str | None = (
        "Anatomical low-density path estimate only — not a clinical working length; confirm with apex locator/radiograph."
    )
    clinicianReviewed: bool = False
    reviewRequired: Literal[True] = True
    provider: str = "ct-sidecar:research-v2"
