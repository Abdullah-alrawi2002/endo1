# Endodontic Diagnostic Agent

Agentic endodontic diagnosis (pulpal + apical) from structured clinical findings,
with correction RAG so clinician feedback steers future similar cases.

**Educational decision support only** — not a substitute for licensed clinical judgment.

---

## For non-technical users — open a link

If the app is already hosted, you only need a browser:

1. Open the shared URL (example: `https://your-app.vercel.app`)
2. Enter findings → **Run**

No install. See **[SHARE.md](SHARE.md)** for how the host publishes that URL.

Local/advanced setup: **[SETUP.md](SETUP.md)**.

---

## For the host — publish once, share forever

1. Push this repo to GitHub.
2. Deploy to [Vercel](https://vercel.com) (import repo) **or** [Railway](https://railway.app) (Dockerfile).
3. Set `OPENAI_API_KEY` (and optional `ENDO_MODEL`) in the host dashboard.
4. Copy the public HTTPS URL and send it to people.

Details: **[SHARE.md](SHARE.md)**.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required on the server (visitors do not need a key) |
| `ENDO_MODEL` | `gpt-4o` | Chat model for all stages |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Correction embeddings |
| `ENABLE_CORRECTION_RAG` | `true` | Correction memory |
| `ENABLE_CT_MODULE` | `false` | Experimental CT (keep off for the shared web app) |

---

## How it works

1. Structured clinical form (Cold, EPT, Percussion, Palpation, PARL + optional fields).
2. Stage 0 gate → RAG → Stages 1–4 (biology, pulpal, apical, synthesis) → verifier.
3. Streaming teaching trace + evidence table + final AAE enums (or abstention).
4. Optional **Submit correction** saves lessons for similar future cases.

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Local development |
| `npm run build` / `start` | Production locally |
| `npm run lint` | ESLint |
| `npm run batch -- input.csv output.csv` | Batch CSV — **[BATCH.md](BATCH.md)** |

---

## License

MIT — see [LICENSE](LICENSE).
