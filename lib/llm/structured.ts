import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import type { z } from "zod";
import { getChatModelId } from "@/lib/llm/openai-provider";

/**
 * Responses API + Structured Outputs (strict schema).
 * Prefer over free-form JSON for typed agent contracts.
 */
export async function structuredResponse<T extends z.ZodType>(opts: {
  name: string;
  system: string;
  user: string;
  schema: T;
  store?: boolean;
}): Promise<{ data: z.infer<T>; modelId: string; raw: string }> {
  const key = process.env.OPENAI_API_KEY;
  if (!key?.trim()) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  const baseURL = process.env.OPENAI_BASE_URL?.trim() || undefined;
  const client = new OpenAI({ apiKey: key, baseURL });
  const modelId = getChatModelId();
  const store = opts.store ?? process.env.LLM_STORE === "true";

  const res = await client.responses.parse({
    model: modelId,
    store,
    input: [
      { role: "system", content: opts.system },
      { role: "user", content: opts.user },
    ],
    text: {
      format: zodTextFormat(opts.schema, opts.name),
    },
  });

  const data = res.output_parsed;
  if (data == null) {
    throw new Error(`Structured output empty for ${opts.name}`);
  }
  return {
    data: opts.schema.parse(data),
    modelId,
    raw: res.output_text ?? JSON.stringify(data),
  };
}

export function shouldSkipLlm(): boolean {
  return (
    process.env.ENDO_ECR_SKIP_LLM === "1" ||
    process.env.ENDO_ECR_SKIP_LLM === "true" ||
    !process.env.OPENAI_API_KEY?.trim()
  );
}

/** Pinned model id for audit — never silently switch mid-analysis. */
export function pinnedModelId(): string {
  return getChatModelId();
}
