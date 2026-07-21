import { publicFeatureFlags } from "@/lib/features";
import { publicCtUploadUrl } from "@/lib/ct-analysis/remote";

export const runtime = "nodejs";

export async function GET() {
  return Response.json({
    ...publicFeatureFlags(),
    ctUploadUrl: publicCtUploadUrl(),
  });
}
