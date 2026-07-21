from __future__ import annotations

import zipfile
import os
from pathlib import Path

import SimpleITK as sitk


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        max_bytes = int(
            os.getenv("CT_MAX_EXTRACTED_BYTES", str(2 * 1024 * 1024 * 1024))
        )
        if sum(member.file_size for member in zf.infolist()) > max_bytes:
            raise ValueError("ZIP expands beyond the configured size limit")
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("ZIP contains an unsafe path")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as output:
                output.write(source.read())


def load_dicom_volume(directory: Path) -> tuple[sitk.Image, list[str]]:
    """Load the largest DICOM series, or a single volumetric DICOM file."""
    candidates: list[tuple[int, list[str]]] = []
    candidate_directories = [directory, *[p for p in directory.rglob("*") if p.is_dir()]]
    for candidate_directory in candidate_directories:
        series_ids = (
            sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(candidate_directory)) or []
        )
        for series_id in series_ids:
            names = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(candidate_directory), series_id, True
                )
            )
            candidates.append((len(names), names))
    if candidates:
        _, names = max(candidates, key=lambda item: item[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOff()
        return reader.Execute(), names

    files = [p for p in directory.rglob("*") if p.is_file()]
    for path in files:
        try:
            image = sitk.ReadImage(str(path))
        except RuntimeError:
            continue
        if image.GetDimension() == 3:
            return image, [str(path)]
    raise ValueError("No readable 3D DICOM series was found")


def modality_value_image(image: sitk.Image) -> tuple[sitk.Image, bool, str]:
    """
    SimpleITK/GDCM applies RescaleSlope/Intercept while reading CT DICOM.
    True medical CT can therefore be treated as HU. Dental CBCT gray values
    remain scanner-dependent and are only valid for same-scan comparisons.
    """
    modality = ""
    for key in ("0008|0060", "Modality"):
        if image.HasMetaDataKey(key):
            modality = image.GetMetaData(key).strip().upper()
            break
    is_hu = modality == "CT"
    return sitk.Cast(image, sitk.sitkFloat32), is_hu, modality or "CBCT"
