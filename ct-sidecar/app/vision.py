from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI

from .models import VisionScreen


SYSTEM_PROMPT = """You are a cautious endodontic CBCT morphology screener.
You receive axial, coronal, and sagittal MPR images centered on a heuristically
localized target tooth. This is NOT a diagnosis and the tooth localization may
be wrong.

Tasks:
1. Describe visible canal/root morphology.
2. Propose a Vertucci class only if all three planes support it; otherwise null.
3. Explicitly list relevant normal-anatomy mimics considered: mental foramen,
mandibular canal, maxillary sinus/recess, incisive canal, marrow spaces,
nutrient canals, and metal/beam-hardening artifact.

Do not diagnose pulp vitality or apical disease. Do not claim a working length.
Prefer low confidence and uncertainty over guessing.

Return JSON only:
{
  "vertucciClassification": string | null,
  "mimicsConsidered": string[],
  "morphologyNotes": string,
  "confidence": "low" | "moderate" | "high"
}"""


def analyze_mpr(
    images: dict[str, Path], target_tooth: str
) -> tuple[str | None, VisionScreen, list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            None,
            VisionScreen(
                mimicsConsidered=[],
                morphologyNotes="Vision morphology screen was not run because OPENAI_API_KEY is unavailable in the CT sidecar.",
                confidence="low",
            ),
            ["Zero-shot MPR morphology screen was skipped."],
        )

    content: list[dict] = [
        {
            "type": "text",
            "text": f"Target tooth (Universal numbering): #{target_tooth}. Review all three labeled planes.",
        }
    ]
    for plane, path in images.items():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"{plane.upper()} plane:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                },
            ]
        )

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=os.getenv("CT_VISION_MODEL", "gpt-4o"),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("empty vision response")
        data = json.loads(raw)
        confidence = data.get("confidence")
        if confidence not in {"low", "moderate", "high"}:
            confidence = "low"
        screen = VisionScreen(
            mimicsConsidered=[
                str(item)[:200] for item in data.get("mimicsConsidered", [])
            ][:20],
            morphologyNotes=str(data.get("morphologyNotes", ""))[:4000],
            confidence=confidence,
        )
        classification = data.get("vertucciClassification")
        return (
            str(classification)[:120] if classification else None,
            screen,
            [],
        )
    except Exception as exc:  # API failures must not discard deterministic data.
        return (
            None,
            VisionScreen(
                mimicsConsidered=[],
                morphologyNotes="Vision morphology screen failed; deterministic CT results remain available.",
                confidence="low",
            ),
            [f"Zero-shot MPR screen failed: {str(exc)[:300]}"],
        )
