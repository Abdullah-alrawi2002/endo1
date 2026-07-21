/** Build a single narrative block from a CSV row for LLM extraction. */

const PREFERRED_COLUMNS = [
  "Filename",
  "Sex",
  "Age",
  "Main appeal",
  "Subsequent",
  "Present medical history",
  "Past medical history",
  "Oral Check",
  "Diagnosis",
  "Treatment plan",
  "Handle",
  "Doctor advices",
];

function normalizeKey(key: string): string {
  return key.trim().toLowerCase().replace(/\s+/g, " ");
}

export function rowToNarrative(row: Record<string, string>): string {
  const normalized = new Map<string, string>();
  for (const [k, v] of Object.entries(row)) {
    if (v?.trim()) normalized.set(normalizeKey(k), v.trim());
  }

  const lines: string[] = [];
  for (const col of PREFERRED_COLUMNS) {
    const val = normalized.get(normalizeKey(col));
    if (val) lines.push(`${col}: ${val}`);
  }

  for (const [k, v] of normalized) {
    const already = PREFERRED_COLUMNS.some((c) => normalizeKey(c) === k);
    if (!already) lines.push(`${k}: ${v}`);
  }

  return lines.join("\n");
}
