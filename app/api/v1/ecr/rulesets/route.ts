import { NextResponse } from "next/server";
import { RULESET_VERSION, CANNOT_DETERMINE_FROM_CBCT } from "@/lib/ecr/rules/treatment-options";
import { MANAGEMENT_OPTIONS } from "@/lib/ecr/schemas";

export const runtime = "nodejs";

/** GET /api/v1/ecr/rulesets */
export async function GET() {
  return NextResponse.json({
    active: {
      rulesetVersion: RULESET_VERSION,
      classificationSystem: { name: "PATEL_ECR_3D", version: "2018" },
      schemaVersion: "ECR_CBCT_1.0",
      evidenceScope: "CBCT_ONLY",
      outputType: "conditional_imaging_option_set",
      definitiveTreatmentPlanAvailable: false,
      managementOptions: MANAGEMENT_OPTIONS,
      cannotDetermineFromCbct: CANNOT_DETERMINE_FROM_CBCT,
      notes: [
        "Patel class describes lesion anatomy; it does not prescribe one treatment.",
        "Options are multilabel and conditional on imaging proxies plus clinical confirmation.",
      ],
    },
  });
}
