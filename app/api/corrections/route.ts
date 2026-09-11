import { randomUUID } from "crypto";
import {
  buildCorrectionEmbedDocument,
  serializeCaseCanonical,
} from "@/lib/correction-rag/format-case";
import {
  insertCorrection,
  listRecentCorrections,
} from "@/lib/correction-rag/store";
import { embedCorrectionDocument } from "@/lib/endodontic-agent/run-pipeline";
import { correctionIngestSchema } from "@/lib/schemas/clinical-case";
import { isCorrectionRagEnabled } from "@/lib/features";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(req: Request) {
  if (!isCorrectionRagEnabled()) {
    return Response.json(
      {
        error:
          "Correction RAG is disabled (ENABLE_CORRECTION_RAG=false).",
      },
      { status: 403 },
    );
  }

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
      b.agentStatus,
      b.agentPulpal,
      b.agentApical,
      b.adjudicatedReferenceDiagnosis.status,
      b.adjudicatedReferenceDiagnosis.pulpal,
      b.adjudicatedReferenceDiagnosis.apical,
      b.correctionReasoning,
      b.errorTypes,
      b.specialistIdentity,
      b.misunderstoodSummary,
    );
    const queryVector = await embedCorrectionDocument(embedDocument);
    const approvalRaw = b.approvalStatus;
    const approvalStatus =
      approvalRaw === "approved"
        ? "approved"
        : approvalRaw === "rejected" || approvalRaw === "withdrawn"
          ? "rejected"
          : approvalRaw === "evaluation_only"
            ? "evaluation_only"
            : "pending_review";
    insertCorrection({
      id: randomUUID(),
      createdAt: Date.now(),
      caseCanonical,
      agentPulpal: b.agentPulpal ?? "null",
      agentApical: b.agentApical ?? "null",
      correctedPulpal: b.adjudicatedReferenceDiagnosis.pulpal ?? "null",
      correctedApical: b.adjudicatedReferenceDiagnosis.apical ?? "null",
      reasoning: b.correctionReasoning,
      misunderstood: b.misunderstoodSummary?.trim() || null,
      embedDocument,
      queryVector,
      approvalStatus,
      taxonomyVersion: b.taxonomyVersion,
      reviewerCount: b.reviewerCount,
      containsPHI: b.containsPHI,
    });
    return Response.json({
      ok: true,
      approvalStatus:
        approvalStatus === "approved" && b.reviewerCount >= 2 && !b.containsPHI
          ? "approved"
          : "evaluation_only",
      note:
        "Unreviewed or single-reviewer corrections are evaluation cases only — not prompt context. Dual-reviewed approved PHI-free rows at current taxonomy may be retrieved.",
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Failed to save correction";
    return Response.json({ error: msg }, { status: 500 });
  }
}

export async function GET(req: Request) {
  if (!isCorrectionRagEnabled()) {
    return Response.json({ corrections: [], enabled: false });
  }
  const { searchParams } = new URL(req.url);
  const limit = Math.min(
    100,
    Math.max(1, parseInt(searchParams.get("limit") ?? "30", 10) || 30),
  );
  try {
    const rows = listRecentCorrections(limit);
    return Response.json({ corrections: rows, enabled: true });
  } catch (e) {
    return Response.json(
      { error: e instanceof Error ? e.message : "List failed" },
      { status: 500 },
    );
  }
}
