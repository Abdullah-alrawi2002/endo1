# How to use the Endodontic Diagnostic Agent

## Easiest path — open the website link

If someone already hosted the app for you, you only need a browser:

1. Click the link they sent (or open it in Chrome / Safari / Edge).
2. Fill in the findings.
3. Click **Run**.

No Node.js. No GitHub. No API key on your computer.

Hosting instructions for the person who owns the OpenAI key: **[SHARE.md](SHARE.md)**.

---

## Advanced — run on your own computer

Only needed if you want a private local copy. You will install a few free tools once and use **your own** OpenAI key.

**Time needed:** about 20–30 minutes the first time.

**What you need:**
- A computer (Mac or Windows)
- Internet
- An **OpenAI account** and API key ([platform.openai.com](https://platform.openai.com))

---

## Part 1 — Install Node.js (runs the app)

Node.js is free software that runs this program on your computer.

### On Mac

1. Open Safari or Chrome and go to: **https://nodejs.org**
2. Click the big green button that says something like **“Download Node.js (LTS)”**. LTS means “stable version.”
3. Open the downloaded file (usually in your **Downloads** folder) and follow the installer—click **Continue** and **Install**, enter your Mac password if asked.
4. When finished, close the installer.

### On Windows

1. Go to: **https://nodejs.org**
2. Click **“Download Node.js (LTS)”**.
3. Run the downloaded `.msi` file. Accept the defaults and click **Next** until **Finish**.
4. **Restart your computer** after installing (recommended).

### Check that it worked

1. **Mac:** Open **Terminal** (press Command+Space, type `Terminal`, press Enter).
2. **Windows:** Open **Command Prompt** or **PowerShell** (search for it in the Start menu).

3. Type this exactly and press **Enter**:

   ```text
   node -v
   ```

4. You should see something like `v20.15.1` or `v22.x.x`.  
   - If you see “command not found” or an error, Node did not install correctly—repeat Part 1 or restart the computer.

---

## Part 2 — Get the project files from GitHub

1. Install **Git** if needed (`git --version`).
2. In Terminal / Command Prompt:

   ```text
   cd ~/Documents
   git clone https://github.com/Abdullah-alrawi2002/endo1.git
   cd endo1
   ```

---

## Part 3 — Install app packages

```text
npm install
```

---

## Part 4 — Add your OpenAI API key

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY=sk-...` (your real key).
3. Leave:

   ```text
   ENABLE_CORRECTION_RAG=true
   ENABLE_CT_MODULE=false
   ```

4. Save. Do not upload `.env` to GitHub.

---

## Part 5 — Start the app

```text
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Want a public link for others | See **[SHARE.md](SHARE.md)** — deploy once, then share the URL. |
| `OPENAI_API_KEY is not set` | Check `.env` exists and restart `npm run dev`. |
| Model errors | Try `ENDO_MODEL=gpt-4o-mini` in `.env`. |
