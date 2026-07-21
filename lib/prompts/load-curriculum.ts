import { readFileSync } from "fs";
import { join } from "path";

let cached: string | null = null;

/**
 * Load the curriculum markdown from disk.
 * Hosted builds include this path via next.config outputFileTracingIncludes.
 */
export function loadCurriculum(): string {
  if (cached) return cached;
  const path = join(
    /* turbopackIgnore: true */ process.cwd(),
    "lib",
    "prompts",
    "curriculum.md",
  );
  cached = readFileSync(path, "utf8");
  return cached;
}
