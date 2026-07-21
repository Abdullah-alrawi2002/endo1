import { publicFeatureFlags } from "@/lib/features";

export const runtime = "nodejs";

export async function GET() {
  return Response.json(publicFeatureFlags());
}
