import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  turbopack: {
    root: projectRoot,
  },
  // Ensure curriculum markdown ships with serverless API functions on Vercel.
  outputFileTracingIncludes: {
    "/api/**/*": ["./lib/prompts/**/*"],
    "/*": ["./lib/prompts/**/*"],
  },
};

export default nextConfig;
