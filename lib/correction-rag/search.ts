import type Database from "better-sqlite3";
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

export type CorrectionInsert = Omit<CorrectionRow, "embeddingDim"> & {
  queryVector: number[];
};

export function insertCorrection(
  db: Database.Database,
  row: CorrectionInsert,
): void {
  const stmt = db.prepare(`
    INSERT INTO corrections (
      id, created_at, case_canonical, agent_pulpal, agent_apical,
      corrected_pulpal, corrected_apical, reasoning, misunderstood,
      embed_document, embedding_dim, embedding
    ) VALUES (
      @id, @created_at, @case_canonical, @agent_pulpal, @agent_apical,
      @corrected_pulpal, @corrected_apical, @reasoning, @misunderstood,
      @embed_document, @embedding_dim, @embedding
    )
  `);
  stmt.run({
    id: row.id,
    created_at: row.createdAt,
    case_canonical: row.caseCanonical,
    agent_pulpal: row.agentPulpal,
    agent_apical: row.agentApical,
    corrected_pulpal: row.correctedPulpal,
    corrected_apical: row.correctedApical,
    reasoning: row.reasoning,
    misunderstood: row.misunderstood,
    embed_document: row.embedDocument,
    embedding_dim: row.queryVector.length,
    embedding: embeddingToBlob(row.queryVector),
  });
}

export function searchSimilarCorrections(
  db: Database.Database,
  queryVector: number[],
  topK: number,
): Array<CorrectionRow & { score: number }> {
  const q = new Float32Array(queryVector);
  const rows = db
    .prepare(
      `SELECT id, created_at, case_canonical, agent_pulpal, agent_apical,
        corrected_pulpal, corrected_apical, reasoning, misunderstood,
        embed_document, embedding_dim, embedding FROM corrections`,
    )
    .all() as Array<{
      id: string;
      created_at: number;
      case_canonical: string;
      agent_pulpal: string;
      agent_apical: string;
      corrected_pulpal: string;
      corrected_apical: string;
      reasoning: string;
      misunderstood: string | null;
      embed_document: string;
      embedding_dim: number;
      embedding: Buffer;
    }>;

  const scored = rows
    .map((r) => {
      const v = blobToFloat32(r.embedding);
      if (v.length !== q.length) return null;
      const score = cosineSimilarity(q, v);
      const out: CorrectionRow & { score: number } = {
        id: r.id,
        createdAt: r.created_at,
        caseCanonical: r.case_canonical,
        agentPulpal: r.agent_pulpal,
        agentApical: r.agent_apical,
        correctedPulpal: r.corrected_pulpal,
        correctedApical: r.corrected_apical,
        reasoning: r.reasoning,
        misunderstood: r.misunderstood,
        embedDocument: r.embed_document,
        embeddingDim: r.embedding_dim,
        score,
      };
      return out;
    })
    .filter((x): x is CorrectionRow & { score: number } => x !== null)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return scored;
}

export function listRecentCorrections(
  db: Database.Database,
  limit: number,
): CorrectionRow[] {
  const rows = db
    .prepare(
      `SELECT id, created_at, case_canonical, agent_pulpal, agent_apical,
        corrected_pulpal, corrected_apical, reasoning, misunderstood,
        embed_document, embedding_dim FROM corrections
        ORDER BY created_at DESC LIMIT ?`,
    )
    .all(limit) as Array<{
      id: string;
      created_at: number;
      case_canonical: string;
      agent_pulpal: string;
      agent_apical: string;
      corrected_pulpal: string;
      corrected_apical: string;
      reasoning: string;
      misunderstood: string | null;
      embed_document: string;
      embedding_dim: number;
    }>;

  return rows.map((r) => ({
    id: r.id,
    createdAt: r.created_at,
    caseCanonical: r.case_canonical,
    agentPulpal: r.agent_pulpal,
    agentApical: r.agent_apical,
    correctedPulpal: r.corrected_pulpal,
    correctedApical: r.corrected_apical,
    reasoning: r.reasoning,
    misunderstood: r.misunderstood,
    embedDocument: r.embed_document,
    embeddingDim: r.embedding_dim,
  }));
}
