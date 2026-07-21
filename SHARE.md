# Share this app as a normal website link (no install for visitors)

Visitors should **never** install Node or copy API keys. You (the host) deploy
once, put your OpenAI key on the server, then send people a URL.

---

## For visitors (zero technical knowledge)

1. Open the link your host sent you (looks like `https://….vercel.app`).
2. Fill in the clinical findings.
3. Click **Run** (and optionally upload a CT scan if the host enabled it).
4. No install. Works in Chrome, Safari, or Edge.

---

## For the host — basic website (clinical only, no CT)

### Vercel

1. [vercel.com](https://vercel.com) → sign in with GitHub → **Add New → Project** → import `endo1`.
2. **Environment Variables** (before or after first deploy):

   | Name | Value |
   |------|-------|
   | `OPENAI_API_KEY` | your OpenAI secret key |
   | `ENDO_MODEL` | `gpt-4o` |
   | `ENABLE_CORRECTION_RAG` | `true` |
   | `ENABLE_CT_MODULE` | `false` |

3. **Deploy** → copy the `https://….vercel.app` URL and share it.

You pay OpenAI for everyone's usage. Do not put patient identifiers in the app.

---

## For the host — website **with CT upload for everyone**

CT needs **two** hosted pieces:

| Piece | Where | What it does |
|-------|--------|----------------|
| Web app | **Vercel** | Form, diagnosis, link you share |
| CT processor | **Railway** | Accepts DICOM/ZIP (large files) |

Vercel cannot process large DICOM files itself. The browser uploads **directly** to Railway.

### Step 1 — Deploy the CT processor on Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo** → select `endo1`.
2. Open the new service → **Settings**:
   - **Root Directory:** `ct-sidecar`
   - **Builder:** Dockerfile (Railway should pick up `ct-sidecar/Dockerfile`)
3. **Variables** tab → add:

   | Name | Value |
   |------|-------|
   | `OPENAI_API_KEY` | same key as Vercel (optional; only for MPR vision screen) |
   | `CT_ALLOWED_ORIGINS` | `https://YOUR-VERCEL-URL.vercel.app` (replace with your real Vercel URL; no trailing slash) |
   | `CT_ANALYSIS_STORE_PATH` | `/data/ct-analyses` |

4. **Settings → Networking → Generate Domain** → copy the public URL, e.g.  
   `https://endo1-ct-production.up.railway.app`
5. Wait until deploy is green. Test: open `https://YOUR-RAILWAY-URL/health` — should show `{"status":"ok"}`.

**Optional but recommended:** add a **Volume** mounted at `/data` so CT results survive restarts (Railway → service → Volumes).

**Note:** Full DentalSegmentator segmentation needs large model weights and may not fit free Railway tiers. Upload UI still works; analysis may return `partial` until models are installed.

### Step 2 — Connect Vercel to the CT processor

1. Vercel → your project → **Settings → Environment Variables** → add or edit:

   | Name | Value |
   |------|-------|
   | `ENABLE_CT_MODULE` | `true` |
   | `CT_SIDECAR_URL` | `https://YOUR-RAILWAY-URL` (same Railway domain, **no** trailing slash) |

   Keep `OPENAI_API_KEY` and the other vars from the basic setup.

2. **Deployments → ⋯ → Redeploy** (required after env changes).

### Step 3 — Verify

1. Open your Vercel URL.
2. Near the top you should see **CT module: on**.
3. Below the case form: **CT support (experimental / non-diagnostic)** with file upload.
4. If you see “CT processor URL missing”, `CT_SIDECAR_URL` is wrong or you forgot to redeploy.

### Troubleshooting CT on the live link

| Problem | Fix |
|---------|-----|
| No upload section | `ENABLE_CT_MODULE=true` on Vercel + **Redeploy** |
| “CT processor URL missing” | Set `CT_SIDECAR_URL` to Railway URL + redeploy |
| Upload fails / CORS error | Set `CT_ALLOWED_ORIGINS` on Railway to your exact Vercel URL |
| “Could not reach CT sidecar” | Railway service not running; check `/health` |
| Upload works but Run fails | Railway volume or store path; ensure `CT_ANALYSIS_STORE_PATH` is writable |

---

## Security notes

- Never commit `.env` or paste API keys into GitHub.
- Rotate keys if they leak.
- Educational decision support only — not a certified medical device.
- Do not upload identifiable patient DICOM to shared hosts without proper agreements.
