# Share this app as a normal website link (no install for visitors)

Visitors should **never** install Node or copy API keys. You (the host) deploy
once, put your OpenAI key on the server, then send people a URL.

---

## For visitors (zero technical knowledge)

1. Open the link your host sent you (looks like `https://….vercel.app`).
2. Fill in the clinical findings.
3. Click **Run**.

That is the entire product experience.

---

## For the host — publish a shareable link (about 10 minutes, once)

### Option A — Vercel (easiest for Next.js)

1. Create a free account at [vercel.com](https://vercel.com) (sign in with GitHub).
2. Click **Add New… → Project**.
3. Import the `endo1` GitHub repository.
4. Before deploying, open **Environment Variables** and add:

   | Name | Value |
   |------|-------|
   | `OPENAI_API_KEY` | your OpenAI secret key |
   | `ENDO_MODEL` | `gpt-4o` (or another model your key supports) |
   | `ENABLE_CORRECTION_RAG` | `true` |
   | `ENABLE_CT_MODULE` | `false` |

5. Click **Deploy**.
6. When it finishes, Vercel shows a URL like `https://endo1-….vercel.app`.
7. Copy that URL and send it to people (email, text, Slack).

**Important:** You pay OpenAI for everyone's usage. Do not put patient identifiers into the app.

**Timeouts:** Diagnosis runs several AI steps and can take 1–3 minutes. On the free Vercel plan, very long runs may time out. If that happens often, use Option B (Railway) or upgrade Vercel.

### Option B — Railway (good if Vercel times out)

1. Create an account at [railway.app](https://railway.app).
2. **New Project → Deploy from GitHub** → select `endo1`.
3. Railway detects the `Dockerfile`.
4. Add the same environment variables as above (`OPENAI_API_KEY`, etc.).
5. Generate a public domain in Railway settings.
6. Share that `https://….up.railway.app` link.

---

## After you have a URL

Put it at the top of any invite:

> Open the Endodontic Diagnostic Agent: **https://YOUR-URL-HERE**  
> No install. Works in Chrome, Safari, or Edge.

Update this file (or your README) with the real URL once deployed so collaborators always know where to go.

---

## Security notes for the host

- Never commit `.env` or paste your OpenAI key into GitHub.
- Rotate the key if it leaks.
- This is educational decision support, not a certified medical device.
- Correction memory on serverless hosts is best-effort (may reset when the server sleeps). Diagnosis still works.
