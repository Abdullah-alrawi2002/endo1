# Endodontic Diagnostic Agent — System Design

**Product:** Educational clinical decision-support web app that proposes AAE 2009 pulpal and apical diagnoses from structured findings, with optional correction RAG.

**Not a medical device.** Clinician confirmation is always required.

---

## Goals

1. Accept a structured clinical case (sensibility tests, mechanical tests, imaging notes, symptoms, confounders).
2. Run an agentic multi-stage pipeline with independent pulpal / apical / mimic agents.
3. Gate incomplete or conflicting cases before diagnosis.
4. Optionally retrieve dual-reviewed corrections as soft priors (RAG).
5. Verify proposals deterministically before returning a final envelope.

## Non-goals

- DICOM / CBCT processing or Patel ECR classification (removed from this product).
- Treating imaging gray values as calibrated Hounsfield units.
- Silent definitive treatment planning from imaging alone.

---

## Architecture

```text
Browser (ClinicalCaseForm)
    → POST /api/diagnose (NDJSON stream)
        → Stage 0 clinical gate
        → Correction RAG (approved-only)
        → Stage 1 interpretation
        → Independent pulpal / apical / mimic agents
        → Synthesis
        → Diagnostic critic
        → Deterministic verifier
    → FinalDiagnosis JSON
```

| Path | Role |
|------|------|
| `app/page.tsx` | Clinical UI |
| `components/ClinicalCaseForm.tsx` | Structured intake |
| `app/api/diagnose/route.ts` | Streaming diagnosis API |
| `app/api/corrections/route.ts` | Correction ingest / export |
| `lib/endodontic-agent/` | Pipeline + clinical verifier |
| `lib/correction-rag/` | Embeddings store |
| `lib/prompts/` | Stage prompts + curriculum |
| `lib/schemas/clinical-case.ts` | Zod contracts |

---

## Safety invariants

1. Missing or invalid sensibility tests → abstain rather than invent vitality.
2. Imaging alone cannot establish pulp vitality or endodontic origin.
3. Abscess labels require clinical prerequisites (swelling/pus or sinus tract).
4. Correction RAG retrieves only approved, dual-reviewed, PHI-free rows at the current taxonomy.
5. `clinicianConfirmationRequired` is always true on final output.

---

## Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required |
| `ENDO_MODEL` | Chat model (pin for research) |
| `ENABLE_CORRECTION_RAG` | Correction memory (default on) |
| `CORRECTIONS_JSON_PATH` | Persistence path |

Deploy with the root `Dockerfile` (Node-only) or `npm run dev` locally.
