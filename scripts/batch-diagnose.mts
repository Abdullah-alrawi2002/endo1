/**
 * Batch CSV → endodontic diagnoses.
 *
 * Usage:
 *   npm run batch -- input.csv output.csv
 *   npm run batch -- input.csv output.csv --limit 5 --no-rag
 */
import { config } from "dotenv";
import { resolve } from "path";
import { readCsvFile, writeCsvFile } from "../lib/batch/csv-io";
import {
  extractClinicalCaseFromRowText,
  runDiagnosisOnce,
} from "../lib/batch/diagnose-once";
import { rowToNarrative } from "../lib/batch/row-to-narrative";

config({ path: resolve(process.cwd(), ".env") });

const OUTPUT_COLUMNS = [
  "agent_pulpal_diagnosis",
  "agent_apical_diagnosis",
  "agent_final_diagnosis",
  "agent_biological_justification",
  "agent_warnings",
  "batch_status",
  "batch_error",
  "extracted_eligible",
  "extracted_ineligible_reason",
];

type CliArgs = {
  input: string;
  output: string;
  limit?: number;
  start: number;
  skipRag: boolean;
  delayMs: number;
};

function parseArgs(argv: string[]): CliArgs {
  const positional = argv.filter((a) => !a.startsWith("--"));
  if (positional.length < 2) {
    console.error(`
Usage: npm run batch -- <input.csv> <output.csv> [options]

Options:
  --limit N       Process at most N rows (after --start)
  --start N       Skip first N data rows (default 0)
  --no-rag        Skip correction RAG retrieval (faster/cheaper)
  --delay MS      Pause between rows (default 500, rate-limit friendly)

Example:
  npm run batch -- cases.csv results.csv --limit 10
`);
    process.exit(1);
  }

  const getNum = (flag: string): number | undefined => {
    const i = argv.indexOf(flag);
    if (i === -1 || !argv[i + 1]) return undefined;
    const n = parseInt(argv[i + 1]!, 10);
    return Number.isNaN(n) ? undefined : n;
  };

  return {
    input: resolve(positional[0]!),
    output: resolve(positional[1]!),
    limit: getNum("--limit"),
    start: getNum("--start") ?? 0,
    skipRag: argv.includes("--no-rag"),
    delayMs: getNum("--delay") ?? 500,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!process.env.OPENAI_API_KEY?.trim()) {
    console.error("OPENAI_API_KEY is not set. Add it to .env in the project root.");
    process.exit(1);
  }

  console.log(`Reading ${args.input}`);
  const rows = readCsvFile(args.input);
  console.log(`Found ${rows.length} row(s)`);

  const slice = rows.slice(args.start, args.limit ? args.start + args.limit : undefined);
  console.log(`Processing ${slice.length} row(s) (start=${args.start})`);

  const originalHeaders = rows.length
    ? Object.keys(rows[0]!)
    : Object.keys(slice[0] ?? {});

  const outHeaders = [...originalHeaders, ...OUTPUT_COLUMNS];
  const outRows: Record<string, string>[] = [];

  for (let i = 0; i < slice.length; i++) {
    const row = slice[i]!;
    const rowNum = args.start + i + 1;
    const label =
      row["Filename"]?.trim() ||
      row["filename"]?.trim() ||
      `row ${rowNum}`;
    console.log(`\n[${i + 1}/${slice.length}] ${label}`);

    const out: Record<string, string> = { ...row };
    for (const col of OUTPUT_COLUMNS) out[col] = "";

    try {
      const narrative = rowToNarrative(row);
      console.log("  Extracting structured findings…");
      const extraction = await extractClinicalCaseFromRowText(narrative);
      out.extracted_eligible = String(extraction.eligible);

      if (!extraction.eligible || !extraction.case) {
        out.batch_status = "skipped";
        out.extracted_ineligible_reason =
          extraction.ineligibleReason ?? "Not eligible for endodontic diagnosis";
        console.log(`  Skipped: ${out.extracted_ineligible_reason}`);
        outRows.push(out);
        if (i < slice.length - 1) await sleep(args.delayMs);
        continue;
      }

      console.log("  Running diagnosis pipeline…");
      const result = await runDiagnosisOnce(extraction.case, {
        skipRag: args.skipRag,
      });

      out.agent_pulpal_diagnosis = result.final.pulpalDiagnosis ?? "";
      out.agent_apical_diagnosis = result.final.apicalDiagnosis ?? "";
      out.agent_final_diagnosis = result.final.finalDiagnosisLine ?? "";
      out.agent_biological_justification =
        result.final.biologicalJustification ?? "";
      out.agent_warnings = [
        `status=${result.final.status}`,
        ...(result.final.warnings ?? []),
        ...(result.final.conflicts ?? []).map((c) => `conflict:${c}`),
        ...(result.final.missingRequiredData ?? []).map((m) => `missing:${m}`),
      ].join(" | ");
      out.batch_status = "ok";
      console.log(
        `  → [${result.final.status}] ${result.final.pulpalDiagnosis ?? "—"} | ${result.final.apicalDiagnosis ?? "—"}`,
      );
    } catch (e) {
      out.batch_status = "error";
      out.batch_error = e instanceof Error ? e.message : String(e);
      console.error(`  Error: ${out.batch_error}`);
    }

    outRows.push(out);
    if (i < slice.length - 1) await sleep(args.delayMs);
  }

  writeCsvFile(args.output, outHeaders, outRows);
  console.log(`\nWrote ${outRows.length} row(s) to ${args.output}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
