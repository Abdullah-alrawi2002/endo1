import { readFileSync, writeFileSync } from "fs";
import { parse } from "csv-parse/sync";

export function readCsvFile(filePath: string): Record<string, string>[] {
  const text = readFileSync(filePath, "utf8");
  const delimiter = text.includes("\t") && !text.slice(0, 200).includes(",")
    ? "\t"
    : ",";

  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: true,
    trim: true,
    delimiter,
    bom: true,
  }) as Record<string, string>[];

  return records.map((row) => {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(row)) {
      out[k.trim()] = typeof v === "string" ? v : String(v ?? "");
    }
    return out;
  });
}

export function writeCsvFile(
  filePath: string,
  headers: string[],
  rows: Record<string, string>[],
): void {
  const lines = [
    headers.join(","),
    ...rows.map((row) =>
      headers.map((h) => escapeCell(row[h] ?? "")).join(","),
    ),
  ];
  writeFileSync(filePath, lines.join("\n") + "\n", "utf8");
}

function escapeCell(value: string): string {
  if (/[",\n\r]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}
