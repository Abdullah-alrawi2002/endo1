import { z } from "zod";
import {
  AGENT_CONTRACT_VERSION,
  componentAgentResultSchema,
  type ComponentAgentResult,
  type HeightEvidenceView,
} from "@/lib/ecr/agents/schemas";
import { interpretHeightOffline } from "@/lib/ecr/agents/offline-interpreters";
import { pinnedModelId, shouldSkipLlm, structuredResponse } from "@/lib/llm/structured";

export const HEIGHT_PROMPT_VERSION = "patel-height-3.0";

const llmHeightSchema = z.object({
  status: z.enum(["complete", "indeterminate", "abstained", "rejected"]),
  conclusion: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.null()]),
  alternatives: z.array(z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4)])).max(4),
  evidenceIds: z.array(z.string()).min(1).max(20),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string()).max(20),
  abstain: z.boolean(),
});

const SYSTEM = `You are the Patel HEIGHT specialist for external cervical resorption (ECR) on CBCT.
Assign Patel height 1–4 from the evidence package ONLY.
Rules:
- Height 1: lesion at CEJ level or coronal to / at the local alveolar crest.
- Height 2–4: lesion apical to crest, mapped to coronal / middle / apical root thirds of CEJ–apex length.
- If required structures are invisible or measurements null → indeterminate/abstain.
- Cite only evidenceIds provided. Do not invent measurements or alter numeric values.
- Do NOT output a full Patel code. Do NOT infer pulp vitality.
- Never copy a precomputed classification — none is provided.`;

export async function runHeightAgent(
  view: HeightEvidenceView,
): Promise<ComponentAgentResult> {
  if (shouldSkipLlm()) {
    return interpretHeightOffline(view);
  }

  const { data, modelId } = await structuredResponse({
    name: "patel_height_agent",
    system: SYSTEM,
    user: JSON.stringify(view, null, 2),
    schema: llmHeightSchema,
    store: false,
  });

  return componentAgentResultSchema.parse({
    agentRole: "patel_height",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: HEIGHT_PROMPT_VERSION,
    modelId: modelId || pinnedModelId(),
    caseEvidenceHash: view.caseEvidenceHash,
    ...data,
  });
}
