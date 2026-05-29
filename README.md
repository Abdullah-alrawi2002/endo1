# Endodontic Diagnostic Agent

Agentic endodontic diagnosis (pulpal + apical) from structured clinical findings, with **local correction RAG** so clinician corrections can steer future similar cases.

**Educational decision support only** — not a substitute for licensed clinical judgment.

---

## Run locally (any machine)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/endo1.git
cd endo1
```

Replace `YOUR_USERNAME/endo1` with your actual GitHub repo URL after you create it.

### 2. Install Node.js

Use **Node.js 20 or newer** ([nodejs.org](https://nodejs.org/) or [nvm](https://github.com/nvm-sh/nvm)):

```bash
node -v   # should print v20.x or higher
```

If you use nvm and this repo includes `.nvmrc`:

```bash
nvm install
nvm use
```

### 3. Install dependencies

```bash
npm install
```

`better-sqlite3` compiles a small native module. If install fails:

- **macOS:** `xcode-select --install`
- **Windows:** install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++)
- **Linux:** `sudo apt install build-essential` (Debian/Ubuntu) or equivalent

Then run `npm install` again.

### 4. Add your OpenAI API key (required)

**Never commit your real key.** The repo only ships [`.env.example`](.env.example).

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
OPENAI_API_KEY=sk-proj-...your-key...
```

Other variables are optional (defaults match this project’s intended behavior):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENDO_MODEL` | `gpt-4o` | Chat model for all diagnosis stages |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for correction RAG |
| `RAG_TOP_K` | `5` | How many past corrections to retrieve |
| `CORRECTIONS_DB_PATH` | `data/corrections.db` | Local SQLite file (created automatically) |

Your key must have access to the models you configure. If `gpt-4o` is unavailable on your account, set e.g. `ENDO_MODEL=gpt-4o-mini` in `.env`.

### 5. Start the app

```bash
npm run dev
```

Open **http://localhost:3000**

### 6. Stop

Press **Ctrl+C** in the terminal.

---

## Production-style run (optional)

```bash
npm run build
npm run start
```

Still uses `.env` in the project root. Default port: **3000** (override with `PORT=3001 npm run start`).

---

## What gets stored locally

- **`data/corrections.db`** — correction RAG (created on first save). Git ignores `*.db` so each user keeps their own DB.
- **`.env`** — your secrets (gitignored).

Do not put patient identifiers in notes or corrections.

---

## How it works

1. **Inputs:** Visual (sinus tract, swelling, decay, trauma), Clinical (cold, percussion, palpation; optional EPT, fluorescent light, Tooth Slooth), Imaging (PARL, J-shaped RL, widened PDL, resorption), plus notes.
2. **Pipeline:** Stage 1 (interpretation) → 2 (pulpal) → 3 (apical) → 4 (JSON synthesis + checks).
3. **RAG:** Similar saved corrections are retrieved before reasoning.
4. **Corrections:** After a run, **Correct diagnosis** saves lessons for future cases.

Curriculum and rules live in [`lib/prompts/curriculum.md`](lib/prompts/curriculum.md).

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm run start` | Run production build |
| `npm run lint` | ESLint |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | Create `.env` from `.env.example` and set the key; restart `npm run dev`. |
| Model not found / 404 from OpenAI | Set `ENDO_MODEL` to a model your API key supports. |
| `better-sqlite3` install error | Install build tools (see step 3), then `npm install`. |
| Port 3000 in use | `PORT=3001 npm run dev` |
| Hydration warning on `<html>` | Often a browser extension; safe to ignore, or use incognito. |

---

## Publishing to GitHub (maintainers)

1. Ensure `.env` is **not** tracked: `git status` should not list `.env`.
2. Commit application source (not `node_modules`, `.next`, or `data/*.db`).
3. Push to GitHub; collaborators follow **Run locally** above with their own keys.

```bash
git add .
git status   # confirm .env and node_modules are absent
git commit -m "Add endodontic diagnostic agent"
git remote add origin https://github.com/YOUR_USERNAME/endo1.git
git push -u origin main
```

---

## License

MIT — see [LICENSE](LICENSE).
