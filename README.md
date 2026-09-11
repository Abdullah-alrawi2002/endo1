# Endodontic Diagnostic Agent

Agentic endodontic diagnosis (pulpal + apical) from structured clinical findings,
with optional correction RAG.

**Educational decision support only** — not a substitute for licensed clinical judgment.

---

## System design

See **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**.

## Quick start

```bash
npm install
cp .env.example .env   # set OPENAI_API_KEY
npm run dev
```

Open **http://localhost:3000**

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required |
| `ENDO_MODEL` | `gpt-4o` | Chat model |
| `ENABLE_CORRECTION_RAG` | `true` | Correction memory |

Docker:

```bash
cp .env.example .env
npm run app:up
```

Beginner local setup: **[SETUP.md](SETUP.md)**. Sharing a hosted link: **[SHARE.md](SHARE.md)**.

---

## License

MIT — see [LICENSE](LICENSE).
