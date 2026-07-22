# Share one link — web app + CT in one place

Visitors only need a browser. You deploy **once** to one host and share one URL.

---

## Recommended: all-in-one on Railway (one service, one URL)

This runs the website **and** the CT processor in a **single Docker container**.

### Steps

1. Push / merge the latest `endo1` code to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub** → select **endo1**.
3. Open the service → **Settings**:
   - **Builder:** Dockerfile  
   - **Dockerfile path:** `Dockerfile` (repo root)  
   - Root directory: leave empty (whole repo)
4. **Variables** → add:

   | Name | Value |
   |------|--------|
   | `OPENAI_API_KEY` | your OpenAI secret key |
   | `ENDO_MODEL` | `gpt-4o` |
   | `ENABLE_CT_MODULE` | `true` |
   | `ENABLE_CORRECTION_RAG` | `true` |

   (CT sidecar URL is already set inside the image to `http://127.0.0.1:8000`.)

5. **Settings → Networking → Generate Domain**.
6. Wait for deploy to finish. Open the Railway URL.
7. You should see **CT module: on** and the DICOM upload section.
8. Send that **one** URL to everyone.

Optional: add a **Volume** mounted at `/data` so corrections and CT results persist across restarts.

### Local all-in-one (same image)

```bash
cp .env.example .env   # set OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:3000**

---

## For visitors

1. Open the link you sent them.
2. Fill findings → optionally upload CT → **Run**.
3. No install. No API key on their device.

---

## Alternative: Vercel-only (clinical, no CT)

Vercel cannot run the Python CT engine in the same project. Clinical-only:

| Name | Value |
|------|--------|
| `OPENAI_API_KEY` | your key |
| `ENABLE_CT_MODULE` | `false` |

Import the repo at [vercel.com](https://vercel.com). Share the `*.vercel.app` link.

If you later want CT on Vercel, you must add a second CT host (see older split setup). Prefer the Railway all-in-one instead.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Site up but no CT upload | Set `ENABLE_CT_MODULE=true` and redeploy |
| CT upload errors / sidecar unreachable | Check Railway logs for `ct-sidecar` and `web` both running |
| Deploy OOM / build fails | Use a larger Railway plan; CT deps are heavy |
| Segmentation always `partial` | Install DentalSegmentator weights into `/models` (optional volume) |

---

## Security

- Never commit `.env` or paste keys into GitHub.
- Educational decision support only — not a certified medical device.
- Avoid identifiable patient DICOM on shared hosts without proper agreements.
