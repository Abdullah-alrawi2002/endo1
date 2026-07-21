# Batch CSV diagnosis

Run many hospital/chart cases from a **CSV or TSV file** and get pulpal + apical diagnoses in an output file.

Each row is processed in two steps (same logic as the web app):

1. **Extract** structured test findings from free-text columns (especially **Oral Check**, **Main appeal**, **Subsequent**) using the LLM.
2. **Diagnose** with the full multi-stage agent (+ optional RAG).

---

## Prepare your file

Supported formats:

- **Tab-separated (TSV)** — common in Excel exports (your examples use tabs)
- **Comma-separated (CSV)**

Expected columns (names are flexible; these are recognized):

| Column | Role |
|--------|------|
| Filename | Row id |
| Sex, Age | Context |
| Main appeal | Chief complaint |
| Subsequent | Additional complaint |
| Present / Past medical history | Context |
| **Oral Check** | **Most important** — cold, EPT, percussion, palpation, imaging notes |
| Diagnosis, Treatment plan, etc. | Extra context |

Example shorthand in Oral Check that the extractor understands:

- `cold (-)` → cold negative  
- `cold (+)` → vital response  
- `EPT (+/-)`, `percussion (+/-)`, `palpation (+/-)`  
- `PARL`, `radiolucency`, `sinus tract`, `swelling`, `decay`  

Rows with **no endodontic test data** (e.g. implant consult only, missing teeth only) are marked **`skipped`** in the output—not diagnosed.

Sample file: [`data/sample-cases.tsv`](data/sample-cases.tsv)

---

## Run

From the project folder, with `.env` containing `OPENAI_API_KEY`:

```bash
npm run batch -- input.csv output.csv
```

### Options

```bash
npm run batch -- input.tsv results.csv --limit 10      # first 10 rows only
npm run batch -- input.tsv results.csv --start 5       # skip first 5 rows
npm run batch -- input.tsv results.csv --no-rag          # skip correction RAG (faster)
npm run batch -- input.tsv results.csv --delay 1000      # 1s pause between rows
```

---

## Output columns (appended to your file)

| Column | Meaning |
|--------|---------|
| `agent_pulpal_diagnosis` | Pulpal label |
| `agent_apical_diagnosis` | Apical label |
| `agent_final_diagnosis` | Combined one-line diagnosis |
| `agent_biological_justification` | Short reasoning |
| `agent_warnings` | Warnings joined with ` \| ` |
| `batch_status` | `ok`, `skipped`, or `error` |
| `batch_error` | Error message if failed |
| `extracted_eligible` | `true` / `false` |
| `extracted_ineligible_reason` | Why a row was skipped |

---

## Cost and time

Each **eligible** row uses **several OpenAI calls** (extract + 4 diagnosis stages). A 100-row file can take a long time and incur meaningful API cost. Use `--limit` while testing.

---

## Example

```bash
npm run batch -- data/sample-cases.tsv data/sample-results.csv --limit 2 --no-rag
```

Open `data/sample-results.csv` in Excel or Google Sheets.
