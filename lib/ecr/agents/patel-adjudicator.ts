import { z } from "zod";
import {
  AGENT_CONTRACT_VERSION,
  adjudicatorResultSchema,
  type AdjudicatorResult,
  type ComponentAgentResult,
  type EcrEvidencePackage,
} from "@/lib/ecr/agents/schemas";
import { PATEL_CANAL_PROXIMITIES, PATEL_CIRCUMFERENCES } from "@/lib/ecr/schemas";
import { buildPatelCode } from "@/lib/ecr/geometry/patel";
import { pinnedModelId, shouldSkipLlm, structuredResponse } from "@/lib/llm/structured";

export const ADJUDICATOR_PROMPT_VERSION = "patel-adjudicator-3.0";

const llmAdjSchema = z.object({
  status: z.enum([
    "complete",
    "indeterminate",
    "abstained",
    "rejected",
    "needs_specialist_review",
  ]),
  patelCode: z.string().regex(/^[1-4][A-D][dp]$/).nullable(),
  height: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4), z.null()]),
  circumference: z.enum([...PATEL_CIRCUMFERENCES]).nullable(),
  canalProximity: z.enum([...PATEL_CANAL_PROXIMITIES]).nullable(),
  evidenceIds: z.array(z.string()).max(40),
  conciseRationale: z.string().max(2000),
  uncertainties: z.array(z.string()).max(20),
  abstain: z.boolean(),
});

const SYSTEM = `You are the Patel adjudicator. Combine three independent component-agent results into one Patel code (e.g. 2Bp).
Rules:
- Use only the component conclusions and cited evidence IDs.
- If any component is abstained/indeterminate without a conclusion, abstain or mark needs_specialist_review.
- Prefer the primary conclusion; note alternatives as uncertainties.
- Do NOT invent measurements. Do NOT diagnose pulp vitality from p.
- You will NOT receive a deterministic reference code.`;

function offlineAdjudicate(
  pkg: EcrEvidencePackage,
  components: ComponentAgentResult[],
): AdjudicatorResult {
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
    circ && canal && typeof canal.conclusion === "string" &&
    (PATEL_CANAL_PROXIMITIES as readonly string[]).includes(canal.conclusion)
      ? (canal.conclusion as "d" | "p")
      : typeof canal?.conclusion === "string" &&
          (PATEL_CANAL_PROXIMITIES as readonly string[]).includes(canal.conclusion)
        ? (canal.conclusion as "d" | "p")
        : null;

  const anyIndet = [height, circ, canal].some(
    (x) => !x || x.abstain || x.status !== "complete" || x.conclusion == null,
  );
  const code = buildPatelCode(h, c, p);

  return adjudicatorResultSchema.parse({
    agentRole: "patel_adjudicator",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: ADJUDICATOR_PROMPT_VERSION,
    modelId: "offline_evidence_interpreter",
    caseEvidenceHash: pkg.caseEvidenceHash,
    status: anyIndet || !code ? "indeterminate" : "complete",
    patelCode: anyIndet ? null : code,
    height: h,
    circumference: c,
    canalProximity: p,
    evidenceIds: [
      ...new Set([
        ...(height?.evidenceIds ?? []),
        ...(circ?.evidenceIds ?? []),
        ...(canal?.evidenceIds ?? []),
      ]),
    ],
    conciseRationale: anyIndet
      ? "One or more components incomplete — cannot form definitive Patel code."
      : `Adjudicated ${code} from independent height/circumference/canal agents.`,
    uncertainties: [
      ...(height?.uncertainties ?? []),
      ...(circ?.uncertainties ?? []),
      ...(canal?.uncertainties ?? []),
    ],
    abstain: anyIndet || !code,
  });
}

export async function runPatelAdjudicator(input: {
  evidence: EcrEvidencePackage;
  components: ComponentAgentResult[];
  reevaluationHint?: string;
}): Promise<AdjudicatorResult> {
  if (shouldSkipLlm()) {
    return offlineAdjudicate(input.evidence, input.components);
  }

  const payload = {
    caseEvidenceHash: input.evidence.caseEvidenceHash,
    quality: input.evidence.quality,
    ecrDifferential: input.evidence.ecrDifferential,
    componentAgents: input.components.map((c) => ({
      agentRole: c.agentRole,
      status: c.status,
      conclusion: c.conclusion,
      alternatives: c.alternatives,
      evidenceIds: c.evidenceIds,
      conciseRationale: c.conciseRationale,
      uncertainties: c.uncertainties,
      abstain: c.abstain,
    })),
    reevaluationHint: input.reevaluationHint ?? null,
    note: "No deterministic Patel code is provided. Do not invent one from memory of prior cases.",
  };

  const { data, modelId } = await structuredResponse({
    name: "patel_adjudicator",
    system: SYSTEM,
    user: JSON.stringify(payload, null, 2),
    schema: llmAdjSchema,
    store: false,
  });

  return adjudicatorResultSchema.parse({
    agentRole: "patel_adjudicator",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: ADJUDICATOR_PROMPT_VERSION,
    modelId: modelId || pinnedModelId(),
    caseEvidenceHash: input.evidence.caseEvidenceHash,
    ...data,
  });
}
