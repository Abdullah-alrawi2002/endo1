import { readFileSync } from "fs";
import { join } from "path";

let cached: string | null = null;

export function loadCurriculum(): string {
  if (cached) return cached;
  const path = join(process.cwd(), "lib", "prompts", "curriculum.md");
  cached = readFileSync(path, "utf8");
  return cached;
}
