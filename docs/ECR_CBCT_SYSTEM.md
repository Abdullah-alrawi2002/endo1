# ECR CBCT System Design (ECR-CBCT 1.0)

Separate product from the AAE clinical diagnostic agent.

**Evidence scope:** CBCT only  
**Classification:** Patel three-dimensional external cervical resorption  
**Status:** Research/educational decision support; specialist confirmation required  
**Phase implemented:** Phase 1 — reliable semiautomated MVP (measurement-driven)

Full product intent is defined in the user specification (ECR-CBCT 1.0). This document maps that design to the code that ships today.

---

## Separation from AAE agent

| AAE clinical agent (`/`) | ECR CBCT product (`/ecr`) |
|--------------------------|---------------------------|
| Clinical tests + imaging | CBCT-derived measurements only |
| AAE pulpal/apical enums | Patel height × circumference × canal proximity |
| LLM proposes; verifier gates | Deterministic geometry + rules; no LLM on classification path |
| Optional CT as non-diagnostic support | CBCT is the sole evidence source |

Do not apply Patel codes to non-cervical resorption. Differentials abstain.

---

## Phase-1 architecture

```text
Clinician measurement / mask review form
        │
        ▼
POST /api/v1/ecr/analyses
        │
        ├─ Patel geometry engine (height, circumference, d/p)
        ├─ Deterministic verifier (quality, differential, consistency)
        └─ Evidence-versioned option engine (ESE_RR_2023_CORRECTED_v1)
        │
        ▼
Structured ECR_CBCT_1.0 JSON + text report
        │
        ▼
Specialist review (confirm / modify / abstain / refer)
```

### Code map

| Path | Role |
|------|------|
| `lib/ecr/schemas.ts` | ECR_CBCT_1.0 Zod contract |
| `lib/ecr/geometry/patel.ts` | Deterministic Patel axes |
| `lib/ecr/rules/verifier.ts` | Accept / reject / abstain |
| `lib/ecr/rules/treatment-options.ts` | Multilabel conditional options |
| `lib/ecr/pipeline.ts` | Measurement → verified result |
| `lib/ecr/store.ts` | Analysis jobs + audit trail |
| `app/ecr/page.tsx` | Review UI |
| `app/api/v1/ecr/**` | API surface |
| `scripts/test-ecr.mts` | Boundary / safety tests |

---

## Encoded safety invariants

1. Output label always states CBCT-derived options are **not** a definitive treatment plan.
2. `definitiveTreatmentPlanAvailable` is always `false`.
3. No Patel code unless differential is `appearance_consistent_with_ecr` and all three axes complete.
4. Borderline measurements set `manualReviewRequired` and retain plausible values.
5. `p` means probable pulpal involvement **on imaging**, not pulp necrosis / AAE pulpal diagnosis.
6. Option cards always list clinical confirmation prerequisites.
7. No DICOM pixels are sent to an LLM (LLM is not on the classification path).

---

## API

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/v1/ecr/analyses` | POST/GET | Create / list |
| `/api/v1/ecr/analyses/{id}` | GET | Job + result |
| `/api/v1/ecr/analyses/{id}/recalculate` | POST | Re-run after corrections |
| `/api/v1/ecr/analyses/{id}/review` | POST | Specialist decision |
| `/api/v1/ecr/analyses/{id}/report` | GET | JSON or text report |
| `/api/v1/ecr/rulesets` | GET | Active rules metadata |
| `/api/v1/ecr/models` | GET | Deployed model cards |

---

## Roadmap (not yet implemented)

- Phase 2: task-specific assisted segmentation + OHIF/Cornerstone viewer  
- Phase 3: ECR detection/differential CNN with external validation  
- Phase 4: prospective silent-mode and monitored deployment  
- AuthN/Z, Orthanc/DICOMweb, Redis workers, PostgreSQL audit (institutional)

Phase-1 intentionally uses clinician-reviewed measurements so geometry and rules can be validated without claiming a generic segmenter identifies ECR.

---

## Tests

```bash
npm run test:ecr
```

Covers circumference boundaries (90/180/270), height thirds/crest, d/p topology, differential abstention, code consistency, overlapping options, and no definitive-plan flag.
