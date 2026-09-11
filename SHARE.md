# Share one link

Deploy the Endodontic Diagnostic Agent once and share a single URL. Visitors only need a browser.

---

## Railway (recommended)

1. Push this repo to GitHub.
2. [railway.app](https://railway.app) → **New Project → Deploy from GitHub**.
3. Builder: **Dockerfile** (repo root).
4. Variables:

   | Name | Value |
   |------|--------|
   | `OPENAI_API_KEY` | your OpenAI key |
   | `ENDO_MODEL` | `gpt-4o` |
   | `ENABLE_CORRECTION_RAG` | `true` |

5. **Networking → Generate Domain**. Share that URL.

Optional: mount a volume at `/data` so correction memory persists.

### Local Docker

```bash
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:3000**

---

## Vercel

Import the repo at [vercel.com](https://vercel.com). Set `OPENAI_API_KEY` and `ENABLE_CORRECTION_RAG=true`.

---

## For visitors

1. Open the link.
2. Enter clinical findings → **Run**.
3. No install. No API key on their device.

Educational decision support only — not a certified medical device.
