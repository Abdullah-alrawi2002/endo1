# ECR CBCT System Design (ECR-CBCT / ENDO_ECR_AGENTS_3.0)

Separate product from the AAE clinical diagnostic agent, optionally fused via `UnifiedCase`.

**Evidence scope:** CBCT only for Patel classification  
**Classification:** Patel three-dimensional external cervical resorption  
**Status:** Research/educational decision support; specialist confirmation required  
**Phases:** Phase-1 measurement-driven geometry + ENDO_ECR_AGENTS_3.0 evidence-first LLM network

---

## Separation from AAE agent

| AAE clinical agent (`/`) | ECR CBCT product (`/ecr`) |
|--------------------------|---------------------------|
| Clinical tests + imaging | CBCT-derived measurements only |
| AAE pulpal/apical enums | Patel height × circumference × canal proximity |
| Independent pulpal/apical/mimic + critic | Evidence package → three Patel agents → adjudicator → critic → deterministic verifier |
| Optional CT as non-diagnostic support | CBCT is the sole evidence source for Patel |

**Integration walls** (`lib/schemas/unified-case.ts`):

- Clinical diagnosis may influence the RCT branch.
- Clinical diagnosis cannot change the Patel code.
- Patel `p` cannot enter the pulpal agent’s evidence.
- CBCT cannot establish vitality.
- Without clinical evidence → `provisional_cbct_based`; with both verified → `provisional_integrated`.

Do not apply Patel codes to non-cervical resorption. Differentials abstain.

---

## Research comparison design

Agents receive **underlying CBCT evidence**, not precomputed Patel classes. The deterministic classifier is an **invisible reference / safety verifier**.

```text
Reviewed ECR measurements
        │
        ▼
Immutable evidence package (caseEvidenceHash)
        │
        ▼
Three independent Patel agents (height / circumference / canal)
        │
        ▼
LLM adjudicator → Independent critic
        │
        ▼
Deterministic verifier (hidden reference)
        │
        ▼
Treatment-agent network (overlapping ESE options)
        │
        ▼
Provisional plan + clinician review
```

Comparison arms: deterministic classifier · single LLM · multi-agent LLM network · specialist reference.

---

## Code map

| Path | Role |
|------|------|
| `lib/ecr/evidence/build-evidence-package.ts` | Immutable evidence package (no codes) |
| `lib/ecr/evidence/evidence-hash.ts` | Stable `caseEvidenceHash` |
| `lib/ecr/agents/*` | Patel + treatment agent network |
| `lib/ecr/geometry/patel.ts` | Deterministic Patel axes (verifier reference) |
| `lib/ecr/rules/verifier.ts` | Accept / reject / abstain |
| `lib/ecr/rules/treatment-options.ts` | Multilabel conditional ESE options |
| `lib/integration/build-integrated-plan.ts` | Unified clinical + ECR plan |
| `lib/schemas/unified-case.ts` | Shared domain object |
| `scripts/test-ecr.mts` | Boundary / safety / agent tests |

Set `ENDO_ECR_AGENT_NETWORK=1` or `useAgentNetwork: true` on `POST /api/v1/ecr/analyses` to run the agent network. `ENDO_ECR_SKIP_LLM=1` uses offline evidence interpreters for CI (still from measurements only — no answer injection into prompts).

Measurements are **required** — fabricated demo measurements are disabled.

---

## Encoded safety invariants

1. Output label always states CBCT-derived options are **not** a definitive treatment plan.
2. `definitiveTreatmentPlanAvailable` is always `false`.
3. No Patel code unless differential is `appearance_consistent_with_ecr` and quality/mask gates pass.
4. Unreviewed masks → **abstain** (not warn).
5. `p` means probable pulpal involvement **on imaging**, not pulp necrosis.
6. Component agents never see deterministic codes, categories, or peer conclusions.
7. Treatment synthesizer preserves overlapping alternatives (e.g. `2Bp`).
8. Correction RAG retrieves only `approvalStatus===approved`, current taxonomy, `reviewerCount>=2`, `containsPHI===false`.
9. CT sidecar stays isolated — no relative apical density override, no generic segmenter → ECR, no gray values as HU.

---

## Tests

```bash
ENDO_ECR_SKIP_LLM=1 npm run test:ecr
```

Covers 90/180/270° boundaries, uncertainty intervals, height crest/thirds, d/p, quality/differential abstention, forged evidence IDs, stale hashes, overlapping treatment options, RAG governance, and integration walls.

---

## Deployment note

Docker/supervisord is fine for demonstration. Before multi-user DICOM: auth/RBAC, PostgreSQL, encrypted object storage, durable workflows, idempotency keys, evidence checksums, audit log, `store: false` on privacy-sensitive model calls, and pinned model/prompt versions (do not silently switch mid-analysis).
