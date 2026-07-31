import { createHash } from "crypto";

/** Canonical JSON for stable hashing (sorted object keys). */
export function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  return `{${keys
    .map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`)
    .join(",")}}`;
}

export function sha256EvidenceHash(payload: unknown): `sha256:${string}` {
  const digest = createHash("sha256")
    .update(stableStringify(payload))
    .digest("hex");
  return `sha256:${digest}`;
}

export function assertEvidenceHash(
  expected: string,
  actual: string,
  context: string,
): void {
  if (expected !== actual) {
    throw new Error(
      `Stale evidence hash for ${context}: expected ${expected}, got ${actual}`,
    );
  }
}
