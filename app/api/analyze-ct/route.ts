import { requestCtAnalysis } from "@/lib/ct-analysis/client";
import { saveCtSupport } from "@/lib/ct-analysis/store";
import { isCtModuleEnabled } from "@/lib/features";
import type { CTSupport } from "@/lib/schemas/clinical-case";

export const runtime = "nodejs";
export const maxDuration = 1800;

const MAX_UPLOAD_BYTES = 1024 * 1024 * 1024;
const MAX_FILES = 3000;

function allowedFile(file: File): boolean {
  const name = file.name.toLowerCase();
  const lastSegment = name.split("/").pop() ?? name;
  return (
    !lastSegment.includes(".") ||
    name.endsWith(".dcm") ||
    name.endsWith(".dicom") ||
    name.endsWith(".zip")
  );
}

export async function POST(req: Request) {
  if (!isCtModuleEnabled()) {
    return Response.json(
      {
        error:
          "CT module is disabled (ENABLE_CT_MODULE=false). It remains an experimental research feature, off by default.",
      },
      { status: 403 },
    );
  }

  let input: FormData;
  try {
    input = await req.formData();
  } catch {
    return Response.json(
      { error: "Expected multipart/form-data with DICOM files." },
      { status: 400 },
    );
  }

  const targetTooth = input.get("targetTooth");
  if (typeof targetTooth !== "string" || !/^(?:[1-9]|[12]\d|3[0-2])$/.test(targetTooth)) {
    return Response.json(
      { error: "Select a target tooth using Universal numbering (1–32)." },
      { status: 400 },
    );
  }

  const clinicianSeedProvided = input.get("clinicianSeedProvided") === "true";
  if (!clinicianSeedProvided) {
    return Response.json(
      {
        error:
          "MVP requires clinician tooth identification / seed confirmation. Automated connected-component numbering is disabled.",
      },
      { status: 400 },
    );
  }

  const files = input
    .getAll("files")
    .filter((item): item is File => item instanceof File);
  if (!files.length) {
    return Response.json(
      { error: "Select a DICOM series or ZIP archive." },
      { status: 400 },
    );
  }
  if (files.length > MAX_FILES) {
    return Response.json(
      { error: `Too many files (maximum ${MAX_FILES}). Use a ZIP archive.` },
      { status: 413 },
    );
  }
  if (files.some((file) => !allowedFile(file))) {
    return Response.json(
      {
        error:
          "Only DICOM files (including extensionless DICOM) or .zip archives are accepted.",
      },
      { status: 415 },
    );
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > MAX_UPLOAD_BYTES) {
    return Response.json(
      { error: "CT upload exceeds the 1 GB local-processing limit." },
      { status: 413 },
    );
  }

  const outbound = new FormData();
  outbound.set("targetTooth", targetTooth);
  outbound.set("clinicianSeedProvided", "true");
  for (const file of files) outbound.append("files", file, file.name);

  try {
    const raw = await requestCtAnalysis(outbound);
    const stored = saveCtSupport({
      ...raw,
      clinicianSeedProvided: true,
      reviewRequired: true,
    } as Omit<CTSupport, "analysisId">);
    return Response.json({
      analysisId: stored.analysisId,
      ctSupportSummary: {
        status: stored.status,
        qualityGatePassed: stored.qualityGatePassed,
        candidateLowAttenuationRegion: stored.candidateLowAttenuationRegion,
        artifactWarnings: stored.artifactWarnings,
        qualityGateFailures: stored.qualityGateFailures,
        clinicianReviewed: stored.clinicianReviewed,
      },
    });
  } catch (error) {
    return Response.json(
      {
        error:
          error instanceof Error ? error.message : "CT preprocessing failed",
      },
      { status: 502 },
    );
  }
}
