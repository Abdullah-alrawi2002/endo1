import { z } from "zod";
import {
  AGENT_CONTRACT_VERSION,
  componentAgentResultSchema,
  type CircumferenceEvidenceView,
  type ComponentAgentResult,
} from "@/lib/ecr/agents/schemas";
import { PATEL_CIRCUMFERENCES } from "@/lib/ecr/schemas";
import { interpretCircumferenceOffline } from "@/lib/ecr/agents/offline-interpreters";
import { pinnedModelId, shouldSkipLlm, structuredResponse } from "@/lib/llm/structured";

export const CIRCUMFERENCE_PROMPT_VERSION = "patel-circumference-3.0";

const llmCircSchema = z.object({
  status: z.enum(["complete", "indeterminate", "abstained", "rejected"]),
  conclusion: z.enum([...PATEL_CIRCUMFERENCES]).nullable(),
  alternatives: z.array(z.enum([...PATEL_CIRCUMFERENCES])).max(4),
  evidenceIds: z.array(z.string()).min(1).max(20),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string()).max(20),
  abstain: z.boolean(),
});

const SYSTEM = `You are the Patel CIRCUMFERENCE specialist for ECR on CBCT.
Assign A/B/C/D from maximum angular spread ONLY.
Boundaries (inclusive upper): ≤90° A; >90–≤180 B; >180–≤270 C; >270 D.
If the uncertainty interval crosses a boundary, set status indeterminate and list both plausible letters.
Cite only provided evidenceIds. Do not invent or alter angles.
Do NOT output a full Patel code or peer conclusions. Do NOT infer pulp vitality.`;

export async function runCircumferenceAgent(
  view: CircumferenceEvidenceView,
): Promise<ComponentAgentResult> {
  if (shouldSkipLlm()) {
    return interpretCircumferenceOffline(view);
  }

  const { data, modelId } = await structuredResponse({
    name: "patel_circumference_agent",
    system: SYSTEM,
    user: JSON.stringify(view, null, 2),
    schema: llmCircSchema,
    store: false,
  });

  return componentAgentResultSchema.parse({
    agentRole: "patel_circumference",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: CIRCUMFERENCE_PROMPT_VERSION,
    modelId: modelId || pinnedModelId(),
    caseEvidenceHash: view.caseEvidenceHash,
    ...data,
  });
}
