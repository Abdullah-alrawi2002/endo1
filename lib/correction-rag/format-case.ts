import type { ClinicalCase } from "@/lib/schemas/clinical-case";

const coldLabels: Record<ClinicalCase["clinical"]["cold"], string> = {
  normal: "Normal (mild/sharp, subsides within 1–2s after removal)",
  exaggerated_non_lingering:
    "Exaggerated but non-lingering (sharp, disappears within seconds)",
  lingering: "Lingering (severe pain continues after removal)",
  negative: "Negative (no response)",
};

function visualLabel(value: ClinicalCase["visual"]["sinusTract"]): string {
  if (value === "present") return "Present";
  if (value === "absent") return "Absent";
  return "Unknown / not assessed";
}

function optionalTestLabel(
  value: ClinicalCase["clinical"]["ept"],
  positive: string,
  negative: string,
): string {
  if (value === "not_performed") return "Not performed";
  if (value === "positive") return positive;
  return negative;
}

function imagingLabel(value: ClinicalCase["imaging"]["periapicalRadiolucency"]): string {
  return value === "present" ? "Present" : "Absent";
}

export function serializeCaseCanonical(c: ClinicalCase): string {
  const lines: string[] = [
    "=== Visual examination ===",
    `Sinus tract: ${visualLabel(c.visual.sinusTract)}`,
    `Swelling: ${visualLabel(c.visual.swelling)}`,
    `Decay: ${visualLabel(c.visual.decay)}`,
    `Signs of trauma: ${visualLabel(c.visual.traumaSigns)}`,
    "",
    "=== Clinical tests ===",
    `Cold test: ${coldLabels[c.clinical.cold]}`,
    ...(c.clinical.cold === "lingering" && c.clinical.coldLingerSeconds
      ? [
          `Cold linger duration (reported): ~${c.clinical.coldLingerSeconds} seconds`,
        ]
      : []),
    `Percussion: ${c.clinical.percussion === "positive" ? "Positive (pain)" : "Negative"}`,
    `Palpation: ${c.clinical.palpation === "positive" ? "Positive (pain/swelling)" : "Negative"}`,
    `EPT: ${optionalTestLabel(c.clinical.ept, "Positive (vital conduction)", "Negative (no conduction)")}`,
    `Fluorescent light: ${optionalTestLabel(c.clinical.fluorescentLight, "Positive (fluorescence / caries indicator)", "Negative")}`,
    `Tooth Slooth biting: ${optionalTestLabel(c.clinical.toothSloothBiting, "Positive (pain on cusp bite)", "Negative")}`,
    "",
    "=== Imaging ===",
    `Periapical radiolucency: ${imagingLabel(c.imaging.periapicalRadiolucency)}`,
    `J-shaped periapical radiolucency: ${imagingLabel(c.imaging.jShapedPeriapicalRadiolucency)}`,
    `Widening of periodontal ligament: ${imagingLabel(c.imaging.widenedPeriodontalLigament)}`,
    `Internal resorption: ${imagingLabel(c.imaging.internalResorption)}`,
    `External resorption: ${imagingLabel(c.imaging.externalResorption)}`,
  ];
  if (c.additionalNotes?.trim()) {
    lines.push("", `Additional notes: ${c.additionalNotes.trim()}`);
  }
  return lines.join("\n");
}

export function buildCorrectionEmbedDocument(
  caseBlock: string,
  agentPulpal: string,
  agentApical: string,
  correctedPulpal: string,
  correctedApical: string,
  reasoning: string,
  misunderstood?: string,
): string {
  return [
    "=== CLINICAL CASE ===",
    caseBlock,
    "",
    "=== AGENT (WRONG) ===",
    `Pulpal: ${agentPulpal}`,
    `Apical: ${agentApical}`,
    "",
    "=== CLINICIAN CORRECTION ===",
    `Pulpal: ${correctedPulpal}`,
    `Apical: ${correctedApical}`,
    "",
    "=== WHY THE AGENT WAS WRONG ===",
    reasoning,
    ...(misunderstood?.trim()
      ? ["", "=== WHAT WAS MISUNDERSTOOD ===", misunderstood.trim()]
      : []),
  ].join("\n");
}
