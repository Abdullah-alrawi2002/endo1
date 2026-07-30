# Endodontic Diagnostic Agent

Agentic endodontic diagnosis (pulpal + apical) from structured findings, with
optional CT research support and correction RAG.

**Educational decision support only** — not a substitute for licensed clinical judgment.

---

## System design

Full architecture, clinical contract, pipeline, CT/RAG, APIs, and deployment:

→ **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**

## One link for everyone (recommended)

Deploy **web + CT together** on Railway (single Docker service):

→ **[SHARE.md](SHARE.md)**

Visitors open your `https://….up.railway.app` URL. No install. CT upload is on the same site.

Local all-in-one:

```bash
cp .env.example .env   # set OPENAI_API_KEY
npm run app:up
```

Open **http://localhost:3000**

---

## Developer setup

```bash
npm install
cp .env.example .env
npm run dev
```

| Variable | Default in Docker | Purpose |
|----------|-------------------|---------|
| `OPENAI_API_KEY` | — | Required |
| `ENDO_MODEL` | `gpt-4o` | Chat model |
| `ENABLE_CT_MODULE` | `true` (Docker) | Show CT upload |
| `CT_SIDECAR_URL` | `http://127.0.0.1:8000` | Internal CT service |
| `ENABLE_CORRECTION_RAG` | `true` | Correction memory |

Beginner local Node install: **[SETUP.md](SETUP.md)**.

---

## License

MIT — see [LICENSE](LICENSE).
