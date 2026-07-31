import { z } from "zod";
import {
  AGENT_CONTRACT_VERSION,
  criticResultSchema,
  type AdjudicatorResult,
  type ComponentAgentResult,
  type CriticResult,
  type EcrEvidencePackage,
} from "@/lib/ecr/agents/schemas";
import { PATEL_CANAL_PROXIMITIES, PATEL_CIRCUMFERENCES } from "@/lib/ecr/schemas";
import { buildPatelCode } from "@/lib/ecr/geometry/patel";
import { pinnedModelId, shouldSkipLlm, structuredResponse } from "@/lib/llm/structured";

export const CRITIC_PROMPT_VERSION = "classification-critic-3.0";

const llmCriticSchema = z.object({
  status: z.enum([
    "complete",
    "indeterminate",
    "abstained",
    "rejected",
    "needs_specialist_review",
  ]),
  criticStatus: z.enum(["reproduced", "contradicted", "abstained"]),
  reconstructedCode: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  contradiction: z.string().max(1000).nullable(),
  evidenceIds: z.array(z.string()).max(40),
  conciseRationale: z.string().max(2000),
  abstain: z.boolean(),
});

const SYSTEM = `You are an independent classification critic for Patel ECR codes.
Independently reconstruct the Patel code from the three component-agent results.
Compare to the adjudicator's code.
- reproduced: your reconstruction matches the adjudicator.
- contradicted: mismatch — explain briefly.
- abstained: insufficient evidence to reconstruct.
You do NOT receive a deterministic reference code. Do not invent vitality diagnoses.`;

function offlineCritic(
  pkg: EcrEvidencePackage,
  components: ComponentAgentResult[],
  adjudicator: AdjudicatorResult,
): CriticResult {
  const height = components.find((c) => c.agentRole === "patel_height");
  const circ = components.find((c) => c.agentRole === "patel_circumference");
  const canal = components.find((c) => c.agentRole === "patel_canal");

  const h =
    typeof height?.conclusion === "number" ? (height.conclusion as 1 | 2 | 3 | 4) : null;
  const c =
    typeof circ?.conclusion === "string" &&
    (PATEL_CIRCUMFERENCES as readonly string[]).includes(circ.conclusion)
      ? (circ.conclusion as "A" | "B" | "C" | "D")
      : null;
  const p =
    typeof canal?.conclusion === "string" &&
    (PATEL_CANAL_PROXIMITIES as readonly string[]).includes(canal.conclusion)
      ? (canal.conclusion as "d" | "p")
      : null;

  const reconstructed = buildPatelCode(h, c, p);
  if (!reconstructed) {
    return criticResultSchema.parse({
      agentRole: "classification_critic",
      agentContractVersion: AGENT_CONTRACT_VERSION,
      promptVersion: CRITIC_PROMPT_VERSION,
      modelId: "offline_evidence_interpreter",
      caseEvidenceHash: pkg.caseEvidenceHash,
      status: "abstained",
      criticStatus: "abstained",
      reconstructedCode: null,
      contradiction: null,
      evidenceIds: [],
      conciseRationale: "Cannot reconstruct code from incomplete components.",
      abstain: true,
    });
  }

  const match = reconstructed === adjudicator.patelCode;
  return criticResultSchema.parse({
    agentRole: "classification_critic",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: CRITIC_PROMPT_VERSION,
    modelId: "offline_evidence_interpreter",
    caseEvidenceHash: pkg.caseEvidenceHash,
    status: match ? "complete" : "needs_specialist_review",
    criticStatus: match ? "reproduced" : "contradicted",
    reconstructedCode: reconstructed,
    contradiction: match
      ? null
      : `Critic ${reconstructed} ≠ adjudicator ${adjudicator.patelCode}`,
    evidenceIds: [...new Set(components.flatMap((x) => x.evidenceIds))],
    conciseRationale: match
      ? `Independently reproduced ${reconstructed}.`
      : `Contradiction: reconstructed ${reconstructed} vs adjudicator ${adjudicator.patelCode}.`,
    abstain: false,
  });
}

export async function runClassificationCritic(input: {
  evidence: EcrEvidencePackage;
  components: ComponentAgentResult[];
  adjudicator: AdjudicatorResult;
}): Promise<CriticResult> {
  if (shouldSkipLlm()) {
    return offlineCritic(input.evidence, input.components, input.adjudicator);
  }

  const payload = {
    caseEvidenceHash: input.evidence.caseEvidenceHash,
    componentAgents: input.components.map((c) => ({
      agentRole: c.agentRole,
      status: c.status,
      conclusion: c.conclusion,
      alternatives: c.alternatives,
      evidenceIds: c.evidenceIds,
      conciseRationale: c.conciseRationale,
    })),
    adjudicatorCode: input.adjudicator.patelCode,
    adjudicatorRationale: input.adjudicator.conciseRationale,
  };

  const { data, modelId } = await structuredResponse({
    name: "classification_critic",
    system: SYSTEM,
    user: JSON.stringify(payload, null, 2),
    schema: llmCriticSchema,
    store: false,
  });

  return criticResultSchema.parse({
    agentRole: "classification_critic",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: CRITIC_PROMPT_VERSION,
    modelId: modelId || pinnedModelId(),
    caseEvidenceHash: input.evidence.caseEvidenceHash,
    ...data,
  });
}
