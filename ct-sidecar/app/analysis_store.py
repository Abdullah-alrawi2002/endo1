"""Persist CT analysis payloads for later retrieval by the web app."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from .models import CTSupport

_ID_PATTERN = re.compile(r"^[a-f0-9-]{8,120}$", re.I)


def store_dir() -> Path:
    configured = os.getenv("CT_ANALYSIS_STORE_PATH", "/data/ct-analyses").strip()
    path = Path(configured)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_analysis(payload: CTSupport) -> CTSupport:
    analysis_id = payload.analysisId or str(uuid.uuid4())
    stored = payload.model_copy(update={"analysisId": analysis_id})
    target = store_dir() / f"{analysis_id}.json"
    target.write_text(stored.model_dump_json(indent=2), encoding="utf-8")
    return stored


def load_analysis(analysis_id: str) -> CTSupport | None:
    if not _ID_PATTERN.match(analysis_id):
        return None
    target = store_dir() / f"{analysis_id}.json"
    if not target.is_file():
        return None
    try:
        return CTSupport.model_validate_json(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None
