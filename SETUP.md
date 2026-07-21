# How to run the Endodontic Diagnostic Agent (beginner guide)

This guide is for **non-technical** users. You will install a few free tools once, add your own OpenAI key, then open the app in your web browser—like using a website, but it runs on **your computer** so your key stays private.

**Time needed:** about 20–30 minutes the first time.

**What you need:**
- A computer (Mac or Windows)
- Internet
- An **OpenAI account** and API key ([platform.openai.com](https://platform.openai.com))
- A credit card may be required on OpenAI for API usage (you pay OpenAI directly for model usage; this app does not charge you)

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

“GitHub” is where the app’s code is stored online. You will copy it to your computer once.

1. Install **Git** (only if `git` does not work in the next step):
   - **Mac:** Git may already be installed. In Terminal, type `git --version` and press Enter. If you see a version number, skip to step 2.
   - If Git is missing on Mac, install “Xcode Command Line Tools” when prompted, or download Git from **https://git-scm.com/downloads**
   - **Windows:** Download from **https://git-scm.com/downloads** and install with default options.

2. In Terminal (Mac) or Command Prompt (Windows), go to a folder where you keep projects—for example your **Documents** folder:

   **Mac:**
   ```text
   cd ~/Documents
   ```

   **Windows:**
   ```text
   cd %USERPROFILE%\Documents
   ```

3. Copy the app from GitHub (this downloads the folder called `endo1`):

   ```text
   git clone https://github.com/Abdullah-alrawi2002/endo1.git
   ```

4. Open that folder:

   ```text
   cd endo1
   ```

You should now be “inside” the project folder in the terminal.

---

## Part 3 — Install the app’s dependencies

The app needs extra code libraries. This step downloads them (one time).

1. Make sure you are still in the `endo1` folder in the terminal (`cd endo1` if needed).

2. Type and press **Enter**:

   ```text
   npm install
   ```

3. Wait until it finishes (can take 2–10 minutes). You should see the cursor ready for a new command with no big red error at the end.

   **If you see an error about “better-sqlite3” or “build tools”:**
   - **Mac:** In Terminal, run: `xcode-select --install` and install, then run `npm install` again.
   - **Windows:** Install “Build Tools for Visual Studio” from Microsoft (C++ build tools), then run `npm install` again.

---

## Part 4 — Add your OpenAI API key (private)

Your API key is like a password that lets the app talk to OpenAI. **Never share it** or post it online.

### Get a key from OpenAI

1. Go to **https://platform.openai.com** and sign in (or create an account).
2. Open **API keys** (menu or **https://platform.openai.com/api-keys**).
3. Click **Create new secret key**, copy the key (starts with `sk-`).  
   **Save it somewhere safe**—you may not see it again.

### Put the key in the project

1. In the `endo1` folder on your computer, find the file **`.env.example`**.  
   - On Mac, in Finder you may need to press **Command + Shift + .** to show hidden files (files starting with a dot).
   - On Windows, enable “Hidden items” in File Explorer view options.

2. **Duplicate** `.env.example` and rename the copy to **`.env`** (exactly—starts with a dot, no “.example”).

3. Open **`.env`** in **TextEdit** (Mac) or **Notepad** (Windows).

4. Find the line:

   ```text
   OPENAI_API_KEY=
   ```

5. Paste your key **right after the equals sign**, with no spaces:

   ```text
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

6. Correction RAG is **on** by default. Leave CT off unless you are evaluating that research module:

   ```text
   ENABLE_CORRECTION_RAG=true
   ENABLE_CT_MODULE=false
   ```

7. **Save** the file and close it.

**Important:** Do not upload `.env` to GitHub or email it to anyone.

---

## Part 5 — Start the app

1. In the terminal, make sure you are in the `endo1` folder.

2. Type and press **Enter**:

   ```text
   npm run dev
   ```

3. Wait until you see a message like **“Ready”** and a line mentioning **http://localhost:3000**.

4. Open your web browser (Chrome, Safari, Edge) and go to:

   **http://localhost:3000**

5. You should see the **Endodontic diagnosis** form. Fill in the case findings and click **Run**.

---

## Part 6 — Stop the app when you are done

1. Click the terminal window where `npm run dev` is running.
2. Press **Ctrl + C** (Mac and Windows).
3. The app stops; the website will no longer load until you run `npm run dev` again.

---

## Every time after the first setup

You only do Parts 1–4 once. Next times:

1. Open Terminal / Command Prompt.
2. Go to the folder:
   ```text
   cd ~/Documents/endo1
   ```
   (adjust the path if you saved it somewhere else)
3. Start the app:
   ```text
   npm run dev
   ```
4. Open **http://localhost:3000** in your browser.

---

## Using the app (short)

1. Fill in **Visual**, **Clinical**, and **Imaging** sections.
2. Click **Run** and wait for the diagnosis (may take a minute).
3. Read the **Diagnosis** at the bottom.
4. Optional: click **Correct diagnosis** if the AI was wrong—this helps it learn from your corrections on **your computer only**.

---

## Optional: enable CT / CBCT upload

The CT feature needs Docker Desktop and a large pre-trained DentalSegmentator
model. A technical user should complete this one-time setup:

```text
./ct-sidecar/install-dental-segmentator.sh
npm run ct:up
```

Keep that terminal open, then start the main app with `npm run dev` in a second
terminal. CT processing may take several minutes on a computer without a
compatible GPU. See `ct-sidecar/README.md` for clinical and technical
limitations.

---

## Common problems

| What you see | What to do |
|--------------|------------|
| `OPENAI_API_KEY is not set` | Check that `.env` exists (not `.env.example`) and has your key on the `OPENAI_API_KEY=` line. Restart `npm run dev`. |
| Page won’t load at localhost:3000 | Make sure `npm run dev` is still running in the terminal. |
| “Repository not found” when cloning | Use the exact link: `https://github.com/Abdullah-alrawi2002/endo1.git` |
| OpenAI error about model | In `.env`, try adding: `ENDO_MODEL=gpt-4o-mini` and save, then restart. |
| `npm` or `node` not found | Install Node.js again (Part 1) and restart the computer. |
| “Could not reach CT sidecar” | Install Docker Desktop, run `npm run ct:up`, and keep that terminal open. |

---

## Privacy reminder

- This is **educational software**, not a medical device or official health record.
- Do **not** type patient names, dates of birth, or other identifying information in the form or corrections.
- Your OpenAI key and correction database stay on **your computer** unless you choose to share them.

---

## Need help?

Ask whoever shared this project with you, or open an **Issue** on GitHub:  
**https://github.com/Abdullah-alrawi2002/endo1/issues**
