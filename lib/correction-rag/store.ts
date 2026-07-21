import { mkdirSync, readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { blobToFloat32, embeddingToBlob } from "@/lib/correction-rag/embed";

export type CorrectionRow = {
  id: string;
  createdAt: number;
  caseCanonical: string;
  agentPulpal: string;
  agentApical: string;
  correctedPulpal: string;
  correctedApical: string;
  reasoning: string;
  misunderstood: string | null;
  embedDocument: string;
  embeddingDim: number;
};

export type CorrectionInsert = Omit<CorrectionRow, "embeddingDim"> & {
  queryVector: number[];
};

type StoredCorrection = CorrectionRow & {
  /** base64 float32 embedding */
  embeddingB64: string;
};

type StoreFile = {
  version: 1;
  corrections: StoredCorrection[];
};

type GlobalStore = typeof globalThis & {
  __endoCorrectionStore?: StoreFile;
};

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return -1;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function storePath(): string {
  if (process.env.VERCEL || process.env.CORRECTIONS_BACKEND === "memory") {
    return join("/tmp", "endo-corrections.json");
  }
  if (process.env.CORRECTIONS_JSON_PATH) {
    return process.env.CORRECTIONS_JSON_PATH;
  }
  return join(/*turbopackIgnore: true*/ process.cwd(), "data", "corrections.json");
}

function emptyStore(): StoreFile {
  return { version: 1, corrections: [] };
}

function loadStore(): StoreFile {
  const g = globalThis as GlobalStore;
  if (g.__endoCorrectionStore) {
    return g.__endoCorrectionStore;
  }
  const path = storePath();
  try {
    if (existsSync(path)) {
      const parsed = JSON.parse(readFileSync(path, "utf8")) as StoreFile;
      if (parsed?.version === 1 && Array.isArray(parsed.corrections)) {
        g.__endoCorrectionStore = parsed;
        return parsed;
      }
    }
  } catch {
    /* fall through */
  }
  const store = emptyStore();
  g.__endoCorrectionStore = store;
  return store;
}

function persistStore(store: StoreFile): void {
  (globalThis as GlobalStore).__endoCorrectionStore = store;
  const path = storePath();
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify(store), "utf8");
  } catch {
    // Serverless filesystems may be read-only outside /tmp; memory still works.
  }
}

function toRow(c: StoredCorrection): CorrectionRow {
  return {
    id: c.id,
    createdAt: c.createdAt,
    caseCanonical: c.caseCanonical,
    agentPulpal: c.agentPulpal,
    agentApical: c.agentApical,
    correctedPulpal: c.correctedPulpal,
    correctedApical: c.correctedApical,
    reasoning: c.reasoning,
    misunderstood: c.misunderstood,
    embedDocument: c.embedDocument,
    embeddingDim: c.embeddingDim,
  };
}

export function insertCorrection(row: CorrectionInsert): void {
  const store = loadStore();
  const blob = embeddingToBlob(row.queryVector);
  const stored: StoredCorrection = {
    id: row.id,
    createdAt: row.createdAt,
    caseCanonical: row.caseCanonical,
    agentPulpal: row.agentPulpal,
    agentApical: row.agentApical,
    correctedPulpal: row.correctedPulpal,
    correctedApical: row.correctedApical,
    reasoning: row.reasoning,
    misunderstood: row.misunderstood,
    embedDocument: row.embedDocument,
    embeddingDim: row.queryVector.length,
    embeddingB64: Buffer.from(blob).toString("base64"),
  };
  store.corrections.push(stored);
  persistStore(store);
}

export function searchSimilarCorrections(
  queryVector: number[],
  topK: number,
): Array<CorrectionRow & { score: number }> {
  const q = new Float32Array(queryVector);
  const store = loadStore();
  return store.corrections
    .map((r) => {
      const v = blobToFloat32(Buffer.from(r.embeddingB64, "base64"));
      if (v.length !== q.length) return null;
      return { ...toRow(r), score: cosineSimilarity(q, v) };
    })
    .filter((x): x is CorrectionRow & { score: number } => x !== null)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export function listRecentCorrections(limit: number): CorrectionRow[] {
  const store = loadStore();
  return [...store.corrections]
    .sort((a, b) => b.createdAt - a.createdAt)
    .slice(0, limit)
    .map(toRow);
}
