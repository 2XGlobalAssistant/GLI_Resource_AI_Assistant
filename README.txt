# 2X GLI Assistant — How-To Guide

**For 2X Global team members managing the Gender Lens Investing AI Assistant**

---

## What this tool does

The GLI Assistant is an AI chatbot embedded on the 2X website. It answers questions from gender lens investors — fund managers, DFIs, banks, and companies — drawing on a curated library of ~70 GLI research documents and the 2X Criteria benchmarks. It can:

- Answer questions about GLI practice, deal structuring, due diligence, and gender KPIs
- Look up 2X Criteria thresholds by country and sector
- Suggest relevant resources from the library to read further

The assistant runs on [Render.com](https://render.com) and is accessed through a chat interface embedded on the 2X website.

---

## Folder structure

```
Resource Bot/
├── app.py                  ← Flask server (entry point)
├── rag.py                  ← Core AI pipeline
├── thresholds.py           ← Benchmark lookup logic
├── logging_utils.py        ← Query and feedback logging
├── update_library.py       ← Tool for adding new resources (see below)
├── requirements.txt        ← Python dependencies
├── runtime.txt             ← Python version for Render
├── render.yaml             ← Render deployment config
├── .env                    ← API keys (never share or commit this file)
│
└── data/
    ├── index.pkl                          ← Search index (auto-generated)
    ├── resources_manifest_llm.json        ← Document library metadata
    ├── chunks.jsonl                       ← Chunked document text (auto-generated)
    ├── Benchmark Thresholds-2X Global Benchmarks (1).xlsx  ← 2X thresholds data
    └── Pdfs/                              ← Source PDF documents
```

---

## The most common task: adding new resources

When new GLI reports, toolkits, or frameworks are published, you can add them to the assistant's library. This takes about 15–30 minutes depending on how many documents you're adding.

### Step 1 — Drop the PDFs in

Copy the new PDF files into the `data/Pdfs/` folder.

### Step 2 — Run the update tool

Open a terminal, navigate to the Resource Bot folder, and run:

```
python update_library.py
```

You'll see this menu:

```
1.   Full update — Part A   (new PDFs added)
1b.  Full update — Part B   (after reviewing metadata_review.csv)
2.   Re-enrich existing resources
3.   Apply edits from enrichment_review.csv
4.   Exit
```

Enter **1** and press Enter.

The tool will:
- Extract text from the new PDFs
- Build the search index
- Use AI to extract the title, publisher, year, and URL for each new document
- Write a file called `metadata_review.csv` and pause

### Step 3 — Review the metadata

Open `metadata_review.csv` in Excel or Google Sheets. You'll see one row per new document with columns like:

| resource_id | title | publisher | year | url | title_snippet | needs_review |
|---|---|---|---|---|---|---|
| 2025_IFC_new-report | A New Report | IFC | 2025 | https://... | "A NEW REPORT ON..." | False |

The `title_snippet`, `publisher_snippet`, and `year_snippet` columns show exactly what the AI found in the document — use these to check its work. Rows marked `needs_review = True` are the ones most likely to need fixing (usually because the PDF cover page was unclear).

Fix any errors directly in the spreadsheet, save, and close.

### Step 4 — Apply your edits

Run the tool again:

```
python update_library.py
```

This time enter **1b**. The tool will save your reviewed metadata to the library, rename the PDFs to a clean format, and confirm it's done.

### Step 5 — Enrich the new resources (recommended)

Run the tool again and enter **2**. This assigns each new document a quality tier (tier 1 = most authoritative, tier 2 = useful, tier 3 = background) and topic tags that help the assistant surface the right documents for each question.

The tool writes `enrichment_review.csv`. Open it, check the tier and tag assignments, correct any that look wrong, save, and close.

Then run the tool again and enter **3** to apply your edits.

### Step 6 — Restart the app on Render

Log in to [render.com](https://render.com), open the **2x-criteria-assistant** service, and click **Manual Deploy → Deploy latest commit**. The app will restart and pick up the new resources automatically.

---

## If something goes wrong

If the tool stops with an error, check the file `update_library.log` in the Resource Bot folder. It contains a plain-English description of what went wrong and can be emailed to the person who set this up for help.

Common issues:

| Problem | What to check |
|---|---|
| "OPENAI_API_KEY is not set" | Open `.env` and make sure the line `OPENAI_API_KEY=sk-...` is present |
| "No PDF files found" | Make sure you copied the PDFs into `data/Pdfs/` (not the root folder) |
| "metadata_review.csv not found" | You skipped step 2 — run option 1 first |
| App not responding on the website | Check Render dashboard for deployment errors |

---

## Viewing feedback

User feedback from the chat interface is collected in a Google Sheet called **GLI Assistant Feedback**. You can find it in the 2X Global Google Drive. Each row contains the feedback text, the question that triggered it, the user's experience level and organisation type, and a timestamp.

---

## What NOT to change

These files control how the AI thinks and retrieves answers. Changes to them can break the assistant in subtle ways — leave them alone unless you know what you're doing:

- `rag.py`
- `thresholds.py`
- `logging_utils.py`

If you need changes to how the assistant answers questions, reach out to the person who built it.

---

## Deployment summary

The app runs on Render's free tier. Key settings:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Environment variable:** `OPENAI_API_KEY` — set in the Render dashboard (Settings → Environment), never stored in the code

The app will spin down after 15 minutes of inactivity on the free tier and take ~30 seconds to wake up on the next request. This is normal.

---

## Contact

If you're reading this after the original developer has left, the codebase is well-documented and the update workflow above covers the most common tasks. For anything beyond adding new resources, the full development history is preserved in the project's Claude conversation logs.