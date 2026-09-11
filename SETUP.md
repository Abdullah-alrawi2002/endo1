# Local setup (Node.js)

1. Install Node.js 20+ from https://nodejs.org  
2. Clone this repo and open a terminal in the project folder.  
3. Install dependencies:

```bash
npm install
cp .env.example .env
```

4. Put your OpenAI key in `.env`:

```text
OPENAI_API_KEY=sk-...
ENABLE_CORRECTION_RAG=true
```

5. Start the app:

```bash
npm run dev
```

Open **http://localhost:3000**

| Problem | Fix |
|---------|-----|
| Want a public link | See **SHARE.md** |
| `OPENAI_API_KEY is not set` | Check `.env` and restart `npm run dev` |
| Model errors | Try `ENDO_MODEL=gpt-4o-mini` in `.env` |

Educational decision support only — not a medical device.
