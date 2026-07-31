/**
 * Offline evidence interpreters for CI / ENDO_ECR_SKIP_LLM.
 * Derive component conclusions from measurements in the evidence package only.
 * Never inject a precomputed Patel code into agent prompts.
 */
import {
  AGENT_CONTRACT_VERSION,
  type CanalEvidenceView,
  type CircumferenceEvidenceView,
  type ComponentAgentResult,
  type HeightEvidenceView,
} from "@/lib/ecr/agents/schemas";
import {
  circumferenceFromAngle,
  computeCanalProximity,
  computePatelHeight,
} from "@/lib/ecr/geometry/patel";

const OFFLINE_MODEL = "offline_evidence_interpreter";
const PROMPT_VERSION = "patel-component-offline-3.0";

export function interpretHeightOffline(
  view: HeightEvidenceView,
): ComponentAgentResult {
  const h = computePatelHeight({
    mostApicalExtentMmFromCEJ: view.mostApicalExtentMmFromCEJ,
    rootLengthCejToApexMm: view.rootLengthCejToApexMm,
    localCrestDistanceMmFromCEJ: view.localCrestDistanceMmFromCEJ,
  });
  const status =
    h.status === "complete"
      ? "complete"
      : h.status === "not_assessable"
        ? "rejected"
        : "indeterminate";
  return {
    agentRole: "patel_height",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: PROMPT_VERSION,
    modelId: OFFLINE_MODEL,
    caseEvidenceHash: view.caseEvidenceHash,
    status,
    conclusion: h.value,
    alternatives: h.plausibleValues.filter((v) => v !== h.value),
    evidenceIds: view.evidenceIds.length ? view.evidenceIds : ["eh_offline"],
    conciseRationale:
      h.notes.join(" ") ||
      `Height from CEJ extent ${view.mostApicalExtentMmFromCEJ} mm vs crest ${view.localCrestDistanceMmFromCEJ} mm and root length ${view.rootLengthCejToApexMm} mm.`,
    uncertainties: h.borderline ? ["borderline height boundary"] : [],
    abstain: status !== "complete",
  };
}

export function interpretCircumferenceOffline(
  view: CircumferenceEvidenceView,
): ComponentAgentResult {
  let angle = view.maximumCircumferenceDegrees;
  const unc = view.circumferenceUncertaintyDegrees;
  // When uncertainty straddles a boundary, mark indeterminate alternatives.
  const result = circumferenceFromAngle(angle);
  let status:
    | "complete"
    | "indeterminate"
    | "rejected"
    | "abstained" =
    result.status === "complete"
      ? "complete"
      : result.status === "not_assessable"
        ? "rejected"
        : "indeterminate";

  const alternatives = [...result.plausibleValues.filter((v) => v !== result.value)];
  const uncertainties: string[] = [];
  if (result.borderline) uncertainties.push("near 90/180/270° boundary");
  if (unc != null && angle != null) {
    const lo = angle - unc;
    const hi = angle + unc;
    const loC = circumferenceFromAngle(lo);
    const hiC = circumferenceFromAngle(hi);
    if (loC.value && hiC.value && loC.value !== hiC.value) {
      uncertainties.push(
        `uncertainty interval [${lo.toFixed(1)}, ${hi.toFixed(1)}] crosses circumference boundary`,
      );
      for (const v of [loC.value, hiC.value]) {
        if (v !== result.value && !alternatives.includes(v)) alternatives.push(v);
      }
      status = "indeterminate";
    }
  }

  return {
    agentRole: "patel_circumference",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: PROMPT_VERSION,
    modelId: OFFLINE_MODEL,
    caseEvidenceHash: view.caseEvidenceHash,
    status,
    conclusion: result.value,
    alternatives,
    evidenceIds: view.evidenceIds.length ? view.evidenceIds : ["ec_offline"],
    conciseRationale:
      result.notes.join(" ") ||
      `Maximum angular spread ${angle ?? "unresolved"}°.`,
    uncertainties,
    abstain: status !== "complete",
  };
}

export function interpretCanalOffline(
  view: CanalEvidenceView,
): ComponentAgentResult {
  const c = computeCanalProximity({
    minimumSeparationMm: view.minimumLesionCanalSeparationMm,
    lowerUncertaintyBoundMm: view.separationUncertaintyLowerBoundMm,
    lesionCanalContactOrIntersection: view.lesionCanalContactOrIntersection,
    continuousDentineBarrierVisible: view.continuousDentineBarrierVisible,
  });
  const status =
    c.status === "complete"
      ? "complete"
      : c.status === "not_assessable"
        ? "rejected"
        : "indeterminate";
  return {
    agentRole: "patel_canal",
    agentContractVersion: AGENT_CONTRACT_VERSION,
    promptVersion: PROMPT_VERSION,
    modelId: OFFLINE_MODEL,
    caseEvidenceHash: view.caseEvidenceHash,
    status,
    conclusion: c.value,
    alternatives: c.plausibleValues.filter((v) => v !== c.value),
    evidenceIds: view.evidenceIds.length ? view.evidenceIds : ["en_offline"],
    conciseRationale:
      c.notes.join(" ") ||
      "Canal proximity from separation, contact, and barrier continuity evidence.",
    uncertainties: c.borderline
      ? ["borderline d/p — spatial uncertainty near barrier"]
      : [],
    abstain: status !== "complete",
  };
}
