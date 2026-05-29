export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type LlmClient = {
  chatComplete(
    messages: ChatMessage[],
    options?: { responseFormatJson?: boolean },
  ): Promise<string>;
  embedText(input: string): Promise<number[]>;
};
