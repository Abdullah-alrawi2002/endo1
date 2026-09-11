import type { ClinicalCase } from "@/lib/schemas/clinical-case";

function label(v: string): string {
  return v.replaceAll("_", " ");
}

/** Canonical text used for embedding / RAG similarity. */
export function serializeCaseCanonical(c: ClinicalCase): string {
  const lines: string[] = [
    `Taxonomy: ${c.taxonomyVersion}`,
    `Tooth: Universal #${c.tooth.universal}${c.tooth.fdi ? ` / FDI ${c.tooth.fdi}` : ""}; dentition=${c.tooth.dentition}; apex=${c.tooth.apexMaturity}`,
    `Treatment history: ${c.treatmentHistory}`,
    "",
    "=== Symptoms ===",
    `Spontaneous: ${c.symptoms.spontaneousPain}; nocturnal: ${c.symptoms.nocturnalPain}; postural: ${c.symptoms.posturalPain}; referred: ${c.symptoms.referredPain}`,
    `Thermal pain history: ${c.symptoms.thermalPainHistory}; heat: ${c.symptoms.heatResponse}; cold relieves pain: ${c.symptoms.coldRelievesPain}`,
    "",
    "=== Visual ===",
    `Sinus tract: ${c.visual.sinusTract} (traced=${c.visual.sinusTractTraced})`,
    `Swelling: ${c.visual.swelling} (severity=${c.visual.swellingSeverity}; rapid=${c.visual.rapidOnsetSwelling}; fluctuance=${c.visual.fluctuance}; pus=${c.visual.pus})`,
    `Fever: ${c.visual.fever}; lymphadenopathy: ${c.visual.lymphadenopathy}`,
    `Decay: ${c.visual.decay}; deep caries/exposure: ${c.visual.deepCariesOrExposure}; crown: ${c.visual.crownPresent}; crack: ${c.visual.crackSuspected}; trauma signs: ${c.visual.traumaSigns}`,
    "",
    "=== Sensibility & mechanical tests ===",
    `Cold: ${label(c.clinical.cold)}; linger_s=${c.clinical.coldLingerSeconds ?? "n/a"}; vs control=${c.clinical.coldComparedToControl}; validity=${c.clinical.coldValidity}; repeated=${c.clinical.coldRepeated}`,
    `EPT: ${c.clinical.ept}; vs control=${c.clinical.eptComparedToControl}; validity=${c.clinical.eptValidity}; repeated=${c.clinical.eptRepeated}`,
    `Percussion: ${c.clinical.percussion}; palpation: ${c.clinical.palpation}; biting: ${c.clinical.biting}`,
    `Tooth Slooth: ${c.clinical.toothSloothBiting}; transillumination: ${c.clinical.transillumination}; fluorescent light: ${c.clinical.fluorescentLight}`,
    "",
    "=== Periodontal ===",
    `Isolated deep pocket: ${c.periodontal.isolatedDeepPocket}; mobility: ${c.periodontal.mobility}; occlusion trauma: ${c.periodontal.occlusionTrauma}`,
    ...(c.periodontal.probingDepthsMm
      ? [`Probing depths: ${c.periodontal.probingDepthsMm}`]
      : []),
    "",
    "=== Imaging (clinician-assessed) ===",
    `PARL: ${c.imaging.periapicalRadiolucency}; J-shaped: ${c.imaging.jShapedPeriapicalRadiolucency}; widened PDL: ${c.imaging.widenedPeriodontalLigament}; lamina dura loss: ${c.imaging.laminaDuraLoss}`,
    `Multiple PA views: ${c.imaging.multiplePaViews}; internal resorption: ${c.imaging.internalResorption}; external resorption: ${c.imaging.externalResorption}`,
    ...(c.imaging.parlLocationNotes
      ? [`PARL notes: ${c.imaging.parlLocationNotes}`]
      : []),
    "",
    "=== Confounders ===",
    `Recent anesthesia: ${c.confounders.recentAnesthesia}; calcification: ${c.confounders.calcificationSuspected}; poor isolation: ${c.confounders.poorIsolation}; generalized low responsiveness: ${c.confounders.generalizedLowResponsiveness}; recent trauma: ${c.confounders.recentTrauma}`,
  ];

  if (c.additionalNotes?.trim()) {
    lines.push("", `Additional notes: ${c.additionalNotes.trim()}`);
  }
  return lines.join("\n");
}

export function buildCorrectionEmbedDocument(
  caseBlock: string,
  agentStatus: string,
  agentPulpal: string | null,
  agentApical: string | null,
  adjudicatedStatus: string,
  correctedPulpal: string | null,
  correctedApical: string | null,
  reasoning: string,
  errorTypes: string[],
  specialistIdentity: string,
  misunderstood?: string,
): string {
  return [
    "=== CLINICAL CASE ===",
    caseBlock,
    "",
    "=== AGENT OUTPUT ===",
    `Status: ${agentStatus}`,
    `Pulpal: ${agentPulpal ?? "null"}`,
    `Apical: ${agentApical ?? "null"}`,
    "",
    "=== ADJUDICATED REFERENCE DIAGNOSIS ===",
    `Status: ${adjudicatedStatus}`,
    `Pulpal: ${correctedPulpal ?? "null"}`,
    `Apical: ${correctedApical ?? "null"}`,
    `Error types: ${errorTypes.join(", ")}`,
    `Specialist: ${specialistIdentity}`,
    "",
    "=== WHY THE AGENT WAS WRONG ===",
    reasoning,
    ...(misunderstood?.trim()
      ? ["", "=== WHAT WAS MISUNDERSTOOD ===", misunderstood.trim()]
      : []),
  ].join("\n");
}
