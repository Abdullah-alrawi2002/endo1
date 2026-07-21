# Endodontic Diagnostic Agent

Next.js web app with an **agentic multi-stage** server pipeline that proposes
**AAE 2009** pulpal + apical diagnoses from structured clinical findings, plus a
**local correction RAG** so clinician corrections steer future similar cases.

**Educational decision support only** — not a substitute for licensed clinical judgment.

Beginner install: **[SETUP.md](SETUP.md)**.

---

## Features (plan)

| Piece | Behavior |
|-------|----------|
| Structured clinical form | Cold, EPT, percussion, palpation, PARL + expanded optional fields (controls, validity, abscess signs, confounders) |
| Stage 0 gate | Scope / sufficiency / abstention before LLM stages |
| Stages 1–4 | Test biology → pulpal → apical → synthesis JSON (streamed teaching trace) |
| Deterministic verifier | Rejects forced one-finding rules; may abstain after Stage 4 |
| Correction RAG | Embed + SQLite cosine top-K; injected before Stage 1 |
| Output envelope | `diagnosable` / `insufficient_data` / `conflicting_data` / `out_of_scope` |

CT/CBCT is an optional research module (`ENABLE_CT_MODULE=true`) and is **off by default**.

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/endo1.git
cd endo1
npm install
cp .env.example .env   # set OPENAI_API_KEY
npm run dev
```

Open **http://localhost:3000**

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required |
| `ENDO_MODEL` | `gpt-4o` | Chat model for all stages |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Correction embeddings |
| `ENABLE_CORRECTION_RAG` | `true` | Correction memory |
| `RAG_TOP_K` | `5` | Neighbors retrieved |
| `CORRECTIONS_DB_PATH` | `data/corrections.db` | Local SQLite |
| `ENABLE_CT_MODULE` | `false` | Experimental CT |

---

## How it works

1. **Inputs:** Visual / clinical / imaging findings (core: Cold, EPT, Percussion, Palpation, PARL) with control comparisons and validity.
2. **Pipeline:** Stage 0 gate → RAG retrieve → Stage 1 biology → Stage 2 pulpal → Stage 3 apical → Stage 4 JSON synthesis → deterministic verifier.
3. **RAG:** Similar saved corrections are retrieved before Stage 1; curriculum + raw inputs win on hard conflicts.
4. **Corrections:** After a run, **Submit correction** stores wrong vs right + reasoning for future retrieval.

Curriculum: [`lib/prompts/curriculum.md`](lib/prompts/curriculum.md) · Stage prompts: [`lib/prompts/agent-stages.ts`](lib/prompts/agent-stages.ts).

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server |
| `npm run build` / `start` | Production |
| `npm run lint` | ESLint |
| `npm run batch -- input.csv output.csv` | Batch CSV — see **[BATCH.md](BATCH.md)** |

---

## Privacy

- Do not put patient identifiers in notes or corrections.
- `data/corrections.db` is local and gitignored.
- Not a certified EHR.

---

## License

MIT — see [LICENSE](LICENSE).
