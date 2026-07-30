import { NextResponse } from "next/server";

export const runtime = "nodejs";

/** GET /api/v1/ecr/models */
export async function GET() {
  return NextResponse.json({
    deployed: [
      {
        id: "manual-or-semiauto-phase1",
        task: "measurement_driven_patel_geometry",
        phase: 1,
        status: "active",
        description:
          "Phase-1 MVP: clinician-reviewed measurements drive deterministic Patel geometry and option ranking. No ECR detection CNN is deployed yet.",
        validation: {
          externalSites: 0,
          releaseGate: "phase1_semiautomated_mvp",
          clinicalFacingAutomation: false,
        },
      },
    ],
    roadmap: [
      "Phase 2: task-specific tooth/canal/bone assisted segmentation",
      "Phase 3: ECR detection and differential model with external validation",
      "Phase 4: prospective silent-mode then monitored specialist deployment",
    ],
  });
}
