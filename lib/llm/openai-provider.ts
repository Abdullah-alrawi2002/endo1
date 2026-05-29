import OpenAI from "openai";
import type { ChatMessage, LlmClient } from "@/lib/llm/types";

/** Default chat model: stronger than mini; override with `ENDO_MODEL`. */
export const DEFAULT_CHAT_MODEL = "gpt-4o";

export function createLlmClient(): LlmClient {
  const key = process.env.OPENAI_API_KEY;
  if (!key?.trim()) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  const baseURL = process.env.OPENAI_BASE_URL?.trim() || undefined;
  const client = new OpenAI({ apiKey: key, baseURL });

  const chatModel =
    process.env.ENDO_MODEL?.trim() || DEFAULT_CHAT_MODEL;
  const embeddingModel =
    process.env.EMBEDDING_MODEL?.trim() || "text-embedding-3-small";

  return {
    async chatComplete(messages: ChatMessage[], options) {
      const res = await client.chat.completions.create({
        model: chatModel,
        messages,
        temperature: options?.responseFormatJson ? 0.1 : 0.2,
        ...(options?.responseFormatJson
          ? { response_format: { type: "json_object" as const } }
          : {}),
      });
      const text = res.choices[0]?.message?.content;
      if (!text) throw new Error("Empty model response");
      return text;
    },
    async embedText(input: string) {
      const res = await client.embeddings.create({
        model: embeddingModel,
        input,
      });
      const vec = res.data[0]?.embedding;
      if (!vec?.length) throw new Error("Empty embedding response");
      return vec;
    },
  };
}

export function getChatModelId(): string {
  return process.env.ENDO_MODEL?.trim() || DEFAULT_CHAT_MODEL;
}
