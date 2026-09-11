import { mkdirSync, readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { blobToFloat32, embeddingToBlob } from "@/lib/correction-rag/embed";
import { TAXONOMY_VERSION } from "@/lib/schemas/clinical-case";

export type CorrectionApprovalStatus =
  | "pending_review"
  | "approved"
  | "rejected"
  | "evaluation_only";

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
  /** Governance — required before retrieval into prompt context. */
  approvalStatus: CorrectionApprovalStatus;
  taxonomyVersion: string;
  reviewerCount: number;
  containsPHI: boolean;
};

export type CorrectionInsert = Omit<CorrectionRow, "embeddingDim"> & {
  queryVector: number[];
};

type StoredCorrection = CorrectionRow & {
  /** base64 float32 embedding */
  embeddingB64: string;
};

type StoreFile = {
  version: 2;
  corrections: StoredCorrection[];
};

type GlobalStore = typeof globalThis & {
  __endoCorrectionStore?: StoreFile;
};

export type CorrectionRetrieveFilter = {
  approvalStatus?: CorrectionApprovalStatus;
  taxonomyVersion?: string;
  minReviewerCount?: number;
  allowPhi?: boolean;
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
  return { version: 2, corrections: [] };
}

function migrateRow(raw: Record<string, unknown>): StoredCorrection | null {
  if (!raw || typeof raw.id !== "string" || typeof raw.embeddingB64 !== "string") {
    return null;
  }
  return {
    id: raw.id,
    createdAt: Number(raw.createdAt) || 0,
    caseCanonical: String(raw.caseCanonical ?? ""),
    agentPulpal: String(raw.agentPulpal ?? ""),
    agentApical: String(raw.agentApical ?? ""),
    correctedPulpal: String(raw.correctedPulpal ?? ""),
    correctedApical: String(raw.correctedApical ?? ""),
    reasoning: String(raw.reasoning ?? ""),
    misunderstood:
      raw.misunderstood == null ? null : String(raw.misunderstood),
    embedDocument: String(raw.embedDocument ?? ""),
    embeddingDim: Number(raw.embeddingDim) || 0,
    approvalStatus:
      (raw.approvalStatus as CorrectionApprovalStatus) ?? "pending_review",
    taxonomyVersion: String(raw.taxonomyVersion ?? TAXONOMY_VERSION),
    reviewerCount: Number(raw.reviewerCount) || 0,
    containsPHI: Boolean(raw.containsPHI),
    embeddingB64: raw.embeddingB64,
  };
}

function loadStore(): StoreFile {
  const g = globalThis as GlobalStore;
  if (g.__endoCorrectionStore) {
    return g.__endoCorrectionStore;
  }
  const path = storePath();
  try {
    if (existsSync(path)) {
      const parsed = JSON.parse(readFileSync(path, "utf8")) as {
        version?: number;
        corrections?: Record<string, unknown>[];
      };
      if (Array.isArray(parsed.corrections)) {
        const store: StoreFile = {
          version: 2,
          corrections: parsed.corrections
            .map(migrateRow)
            .filter((x): x is StoredCorrection => x !== null),
        };
        g.__endoCorrectionStore = store;
        return store;
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
    approvalStatus: c.approvalStatus,
    taxonomyVersion: c.taxonomyVersion,
    reviewerCount: c.reviewerCount,
    containsPHI: c.containsPHI,
  };
}

/** Eligible for prompt retrieval — governance gate before similarity. */
export function isEligibleForPromptRetrieval(row: CorrectionRow): boolean {
  return (
    row.approvalStatus === "approved" &&
    row.taxonomyVersion === TAXONOMY_VERSION &&
    row.reviewerCount >= 2 &&
    row.containsPHI === false
  );
}

/**
 * Unreviewed / single-reviewer corrections become evaluation cases —
 * never prompt context. Similarity alone cannot establish clinical validity.
 */
export function isEvaluationOnlyCase(row: CorrectionRow): boolean {
  return (
    row.approvalStatus === "evaluation_only" ||
    row.approvalStatus === "pending_review" ||
    !isEligibleForPromptRetrieval(row)
  );
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
    approvalStatus: row.approvalStatus ?? "pending_review",
    taxonomyVersion: row.taxonomyVersion ?? TAXONOMY_VERSION,
    reviewerCount: row.reviewerCount ?? 0,
    containsPHI: row.containsPHI ?? false,
    embeddingB64: Buffer.from(blob).toString("base64"),
  };
  // Unreviewed → evaluation_only when not explicitly approved
  if (stored.approvalStatus === "pending_review" && stored.reviewerCount < 2) {
    stored.approvalStatus = "evaluation_only";
  }
  store.corrections.push(stored);
  persistStore(store);
}

export function searchSimilarCorrections(
  queryVector: number[],
  topK: number,
  filter: CorrectionRetrieveFilter = {
    approvalStatus: "approved",
    taxonomyVersion: TAXONOMY_VERSION,
    minReviewerCount: 2,
    allowPhi: false,
  },
): Array<CorrectionRow & { score: number }> {
  const q = new Float32Array(queryVector);
  const store = loadStore();
  const approval = filter.approvalStatus ?? "approved";
  const taxonomy = filter.taxonomyVersion ?? TAXONOMY_VERSION;
  const minReviewers = filter.minReviewerCount ?? 2;
  const allowPhi = filter.allowPhi ?? false;

  return store.corrections
    .map((r) => {
      const row = toRow(r);
      // Hard governance gate BEFORE similarity ranking
      if (row.approvalStatus !== approval) return null;
      if (row.taxonomyVersion !== taxonomy) return null;
      if (row.reviewerCount < minReviewers) return null;
      if (!allowPhi && row.containsPHI) return null;
      const v = blobToFloat32(Buffer.from(r.embeddingB64, "base64"));
      if (v.length !== q.length) return null;
      return { ...row, score: cosineSimilarity(q, v) };
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

/** List evaluation-only cases (not for prompt context). */
export function listEvaluationCases(limit: number): CorrectionRow[] {
  return listRecentCorrections(Math.max(limit * 3, 50))
    .filter(isEvaluationOnlyCase)
    .slice(0, limit);
}
