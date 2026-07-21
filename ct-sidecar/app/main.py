from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .analysis_store import load_analysis, save_analysis
from .dicom_io import safe_extract_zip
from .models import CTSupport
from .pipeline import analyze_ct_series


def _allowed_origins() -> list[str]:
    raw = os.getenv("CT_ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Endodontic CT Research Support",
    version="2.1.0",
    description=(
        "Experimental CBCT support module. Accepts DICOM uploads directly from "
        "the hosted web app. Returns non-diagnostic candidate features."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_analysis_lock = asyncio.Semaphore(
    int(os.getenv("CT_MAX_CONCURRENT_ANALYSES", "1"))
)
MAX_UPLOAD_BYTES = int(os.getenv("CT_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/analysis/{analysis_id}", response_model=CTSupport)
def get_analysis(analysis_id: str) -> CTSupport:
    stored = load_analysis(analysis_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return stored


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="CT upload is too large")
            output.write(chunk)
    return total


@app.post("/analyze", response_model=CTSupport)
async def analyze(
    files: list[UploadFile] = File(...),
    targetTooth: str = Form(...),
    clinicianSeedProvided: str = Form("false"),
) -> CTSupport:
    cleaned_tooth = targetTooth.strip().lstrip("#")
    try:
        tooth_number = int(cleaned_tooth)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="targetTooth must use Universal numbering 1–32"
        ) from exc
    if not 1 <= tooth_number <= 32:
        raise HTTPException(
            status_code=400, detail="targetTooth must use Universal numbering 1–32"
        )
    if not files:
        raise HTTPException(status_code=400, detail="At least one DICOM file is required")

    seed = clinicianSeedProvided.strip().lower() in {"1", "true", "yes"}

    with tempfile.TemporaryDirectory(prefix="endo-ct-") as temporary:
        root = Path(temporary)
        input_dir = root / "dicom"
        work_dir = root / "work"
        input_dir.mkdir()
        work_dir.mkdir()
        total = 0
        for index, upload in enumerate(files):
            suffix = Path(upload.filename or "").suffix.lower()
            safe_name = f"{index:06d}{suffix or '.dcm'}"
            path = input_dir / safe_name
            total += await _save_upload(upload, path)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="CT series is too large")
            if suffix == ".zip":
                extracted = input_dir / f"zip-{index:06d}"
                extracted.mkdir()
                try:
                    safe_extract_zip(path, extracted)
                except (ValueError, OSError) as exc:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid DICOM ZIP: {exc}"
                    ) from exc
                path.unlink(missing_ok=True)

        async with _analysis_lock:
            try:
                result = await run_in_threadpool(
                    analyze_ct_series,
                    input_dir,
                    work_dir,
                    str(tooth_number),
                    seed,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"CT preprocessing failed: {str(exc)[:500]}",
                ) from exc

    return save_analysis(result)
