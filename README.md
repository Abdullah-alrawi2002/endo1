# Endodontic Diagnostic Agent

Clinical decision-support prototype that proposes **AAE 2009** pulpal + apical
diagnoses from structured findings. The first defensible release is
**clinical-only**: rule-verified, able to abstain, and clinician-confirmed.

**Not for real-patient diagnostic use without formal validation and regulatory review.**
Educational / research decision support only — not a substitute for licensed clinical judgment.

**New to installing apps from GitHub?** Use **[SETUP.md](SETUP.md)**.

---

## MVP posture (important)

| Module | Default | Role |
|--------|---------|------|
| Clinical form + Stage 0 gate + deterministic verifier | **On** | Permits AAE enums only when `status = diagnosable` |
| LLM candidate proposal | **On** | Proposes and explains; cannot override verifier abstention |
| Correction RAG | **Off** (`ENABLE_CORRECTION_RAG=false`) | Experimental adjudicated-reference memory |
| CBCT / CT sidecar | **Off** (`ENABLE_CT_MODULE=false`) | Experimental non-diagnostic support |

Taxonomy is frozen as **`AAE_2009`**. Previously treated / previously initiated teeth are **`out_of_scope`** at intake (not diagnosed in this MVP).

Diagnostic envelope statuses: `diagnosable` | `insufficient_data` | `conflicting_data` | `out_of_scope`. Final enums are populated only when `diagnosable`.

---

## Run locally

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/endo1.git
cd endo1
npm install
```

Node.js **20+** required. `better-sqlite3` needs native build tools if install fails.

### 2. Configure `.env`

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY`. Keep CT and RAG off unless you are intentionally evaluating those modules:

```env
ENABLE_CT_MODULE=false
ENABLE_CORRECTION_RAG=false
ENDO_MODEL=gpt-4o
```

Pin `ENDO_MODEL` after evaluation on your locked clinical set.

### 3. Start

```bash
npm run dev
```

Open **http://localhost:3000**

---

## How it works

1. **Expanded clinical contract** — tooth identity (Universal + optional FDI), dentition/apex maturity, treatment history, symptoms, control-tooth comparisons, test validity, severity for percussion/palpation/biting, abscess prerequisites (swelling/pus/rapid onset; traced sinus), imaging with `unknown`/`not_assessed` (never coerce missing → absent).
2. **Stage 0 gate** — scope, sufficiency, confounders, conflicts.
3. **LLM candidates** — pulpal/apical proposals with evidence; no streamed provisional diagnoses.
4. **Deterministic verifier** — rejects forced one-finding rules (e.g. cold−/EPT− alone ≠ necrosis; CT alone ≠ apical disease; AAA/CAA require clinical prerequisites). May abstain.
5. **Clinician confirmation required** on every output.

Curriculum: [`lib/prompts/curriculum.md`](lib/prompts/curriculum.md).

---

## CT / CBCT (experimental, off by default)

Enable only with `ENABLE_CT_MODULE=true`. The module:

- Requires clinician tooth identification / seed (no automated instance numbering for diagnosis).
- Uses pretrained DentalSegmentator (nnU-Net) for **bulk** anatomy — not validated per-tooth/canal/lesion segmentation.
- Returns **non-diagnostic** fields (`candidateLowAttenuationRegion`, relative drop %, quality gates, artifact warnings). **No universal ≥25% diagnostic threshold.**
- Canal length is an anatomical estimate only — never a substitute for apex locator + clinical confirmation.
- Stores an opaque server-side `analysisId`; the diagnose API never trusts client-forged CT metrics.
- Must not be acquired routinely solely to feed the agent (use only when clinically indicated).

```bash
./ct-sidecar/install-dental-segmentator.sh
npm run ct:up
```

Details: [`ct-sidecar/README.md`](ct-sidecar/README.md).

---

## Correction library (experimental, off by default)

Enable with `ENABLE_CORRECTION_RAG=true`. Labels are **`adjudicatedReferenceDiagnosis`** with specialist identity, taxonomy version, approval status, and structured error types. Retrieved memories cannot override evidence prerequisites. Keep validation patients out of the store.

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm run start` | Run production build |
| `npm run lint` | ESLint |
| `npm run ct:up` / `ct:down` | CT sidecar (when evaluating the research module) |
| `npm run batch -- input.csv output.csv` | Batch CSV/TSV — see **[BATCH.md](BATCH.md)** |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example`; restart `npm run dev`. |
| CT upload disabled | Expected when `ENABLE_CT_MODULE=false`. |
| Corrections API disabled | Expected when `ENABLE_CORRECTION_RAG=false`. |
| Status is `insufficient_data` | Complete control-tooth comparisons, test validity, and required findings — the system abstains rather than guessing. |

---

## License

MIT — see [LICENSE](LICENSE).
