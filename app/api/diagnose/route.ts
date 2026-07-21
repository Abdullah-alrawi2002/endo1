import { clinicalCaseSchema } from "@/lib/schemas/clinical-case";
import { runDiagnosisPipeline } from "@/lib/endodontic-agent/run-pipeline";
import { isCtModuleEnabled } from "@/lib/features";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const parsed = clinicalCaseSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json(
      { error: "Validation failed", details: parsed.error.flatten() },
      { status: 400 },
    );
  }

  // Strip any client-supplied ctSupport; only opaque ctAnalysisId is accepted.
  const clinicalCase = {
    ...parsed.data,
    ctSupport: undefined,
    ctAnalysisId:
      isCtModuleEnabled() && parsed.data.ctAnalysisId
        ? parsed.data.ctAnalysisId
        : undefined,
  };

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => {
        controller.enqueue(encoder.encode(`${JSON.stringify(obj)}\n`));
      };
      try {
        for await (const ev of runDiagnosisPipeline(clinicalCase)) {
          send(ev);
        }
      } catch (e) {
        send({
          type: "error",
          message: e instanceof Error ? e.message : "Diagnosis failed",
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}
