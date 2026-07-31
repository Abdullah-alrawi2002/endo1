import { z } from "zod";
import {
  AGENT_CONTRACT_VERSION,
  componentAgentResultSchema,
  type CanalEvidenceView,
  type ComponentAgentResult,
} from "@/lib/ecr/agents/schemas";
import { PATEL_CANAL_PROXIMITIES } from "@/lib/ecr/schemas";
import { interpretCanalOffline } from "@/lib/ecr/agents/offline-interpreters";
import { pinnedModelId, shouldSkipLlm, structuredResponse } from "@/lib/llm/structured";

export const CANAL_PROMPT_VERSION = "patel-canal-3.0";

const llmCanalSchema = z.object({
  status: z.enum(["complete", "indeterminate", "abstained", "rejected"]),
  conclusion: z.enum([...PATEL_CANAL_PROXIMITIES]).nullable(),
  alternatives: z.array(z.enum([...PATEL_CANAL_PROXIMITIES])).max(4),
  evidenceIds: z.array(z.string()).min(1).max(20),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string()).max(20),
  abstain: z.boolean(),
});

const SYSTEM = `You are the Patel CANAL-PROXIMITY specialist for ECR on CBCT.
Assign d (dentine barrier) or p (probable pulpal involvement) from lesion–canal evidence ONLY.
- p if contact/intersection OR no continuous dentine barrier within spatial uncertainty.
- d if continuous dentine barrier visible AND lower uncertainty bound of separation > 0.
- p means probable pulpal involvement on imaging — NOT pulp necrosis and NOT a vitality diagnosis.
Cite only provided evidenceIds. Do not invent measurements.
Do NOT output a full Patel code. Do NOT diagnose pulp vitality.`;

export async function runCanalAgent(
  view: CanalEvidenceView,
): Promise<ComponentAgentResult> {
  if (shouldSkipLlm()) {
    return interpretCanalOffline(view);
  }

  const { data, modelId } = await structuredResponse({
    name: "patel_canal_agent",
    system: SYSTEM,
    user: JSON.stringify(view, null, 2),
    schema: llmCanalSchema,
    store: false,
  });

  return componentAgentResultSchema.parse({
    agentRole: "patel_canal",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: CANAL_PROMPT_VERSION,
    modelId: modelId || pinnedModelId(),
    caseEvidenceHash: view.caseEvidenceHash,
    ...data,
  });
}
