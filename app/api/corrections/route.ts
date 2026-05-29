import { randomUUID } from "crypto";
import {
  buildCorrectionEmbedDocument,
  serializeCaseCanonical,
} from "@/lib/correction-rag/format-case";
import { getDb } from "@/lib/correction-rag/db";
import { insertCorrection, listRecentCorrections } from "@/lib/correction-rag/search";
import { embedCorrectionDocument } from "@/lib/endodontic-agent/run-pipeline";
import { correctionIngestSchema } from "@/lib/schemas/clinical-case";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = correctionIngestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  try {
    const b = parsed.data;
    const caseCanonical = serializeCaseCanonical(b.case);
    const embedDocument = buildCorrectionEmbedDocument(
      caseCanonical,
      b.agentPulpal,
      b.agentApical,
      b.correctedPulpal,
      b.correctedApical,
      b.correctionReasoning,
      b.misunderstoodSummary,
    );
    const queryVector = await embedCorrectionDocument(embedDocument);
    const db = getDb();
    insertCorrection(db, {
      id: randomUUID(),
      createdAt: Date.now(),
      caseCanonical,
      agentPulpal: b.agentPulpal,
      agentApical: b.agentApical,
      correctedPulpal: b.correctedPulpal,
      correctedApical: b.correctedApical,
      reasoning: b.correctionReasoning,
      misunderstood: b.misunderstoodSummary?.trim() || null,
      embedDocument,
      queryVector,
    });
    return Response.json({ ok: true });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Failed to save correction";
    return Response.json({ error: msg }, { status: 500 });
  }
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit = Math.min(
    100,
    Math.max(1, parseInt(searchParams.get("limit") ?? "30", 10) || 30),
  );
  try {
    const db = getDb();
    const rows = listRecentCorrections(db, limit);
    return Response.json({ corrections: rows });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "List failed" },
      { status: 500 },
    );
  }
}
