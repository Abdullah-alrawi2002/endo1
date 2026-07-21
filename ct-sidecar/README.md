# CT/CBCT research support sidecar

Experimental FastAPI service that returns a **non-diagnostic** `CTSupport`
payload. The clinical MVP keeps this module **off** (`ENABLE_CT_MODULE=false`).

It does **not** establish endodontic diagnosis, pulp vitality, or treatment
working length.

## Pipeline

1. Require clinician tooth identification / seed (`clinicianSeedProvided`).
2. Read the largest DICOM series with SimpleITK.
3. Run pretrained DentalSegmentator (nnU-Net Dataset112) for **bulk** classes
   (maxilla, mandible, upper/lower teeth collectively, mandibular canal) —
   not validated individual-tooth, canal, or lesion segmentation.
4. Heuristically isolate the clinician-identified Universal tooth (experimental;
   connected components are **not** used as clinical tooth numbering).
5. Estimate a low-density path length (anatomical estimate only).
6. Compare a 3 mm apical sphere with a robust 4–8 mm adjacent cancellous shell
   and report `relativeAttenuationDropPercent` as an **observation**.
   There is **no universal ≥25% diagnostic threshold**.
7. Optionally send MPR PNGs (no DICOM headers) to a vision model for an
   exploratory morphology / Vertucci **screen**.

## Output fields (non-diagnostic)

- `candidateLowAttenuationRegion` — apex mean lower than same-scan reference
- `qualityGatePassed` / `qualityGateFailures`
- `candidateLocation`, `candidateVolumeMm3` (when available)
- `relativeAttenuationDropPercent`
- `artifactWarnings`
- `canalLengthEstimateMm` + explicit non-clinical note
- `clinicianReviewed` (must be set by the clinician before treating as reviewed)

The Next.js app stores the result server-side under an opaque `analysisId`.
Clients cannot forge CT metrics into `/api/diagnose`.

## Clinical limitations

- DentalSegmentator does not identify individual teeth.
- CBCT gray values are scanner/protocol-dependent; relative drops are not
  calibrated HU and must not drive diagnosis.
- Working length and Vertucci outputs are research estimates only.
- Use CBCT only when clinically indicated — not to feed the agent.

## Setup

```bash
./ct-sidecar/install-dental-segmentator.sh
npm run ct:up
```

```bash
curl http://127.0.0.1:8000/health
```

## Development without Docker

```bash
cd ct-sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```
