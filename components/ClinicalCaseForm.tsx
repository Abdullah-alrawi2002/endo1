"use client";

import type { ClinicalCase } from "@/lib/schemas/clinical-case";

type Props = {
  value: ClinicalCase;
  onChange: (next: ClinicalCase) => void;
};

const TRIAD = [
  { value: "unknown", label: "Unknown" },
  { value: "absent", label: "Absent" },
  { value: "present", label: "Present" },
] as const;

const SEVERITY = [
  { value: "not_assessed", label: "Not assessed" },
  { value: "unknown", label: "Unknown" },
  { value: "none", label: "None" },
  { value: "mild", label: "Mild" },
  { value: "moderate", label: "Moderate" },
  { value: "severe", label: "Severe" },
] as const;

const IMAGING = [
  { value: "not_assessed", label: "Not assessed" },
  { value: "unknown", label: "Unknown" },
  { value: "absent", label: "Absent" },
  { value: "present", label: "Present" },
] as const;

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: ReadonlyArray<{ value: T; label: string }>;
  onChange: (v: T) => void;
}) {
  return (
    <label className="input-field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ClinicalCaseForm({ value, onChange }: Props) {
  return (
    <>
      <div className="input-category">
        <h3>Tooth & scope</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Tooth (Universal #)</span>
            <input
              type="number"
              min={1}
              max={32}
              value={value.tooth.universal}
              onChange={(e) =>
                onChange({
                  ...value,
                  tooth: {
                    ...value.tooth,
                    universal: Number(e.target.value) || 1,
                  },
                })
              }
            />
          </label>
          <label className="input-field">
            <span>FDI (optional)</span>
            <input
              type="text"
              value={value.tooth.fdi ?? ""}
              onChange={(e) =>
                onChange({
                  ...value,
                  tooth: { ...value.tooth, fdi: e.target.value || undefined },
                })
              }
            />
          </label>
          <SelectField
            label="Dentition"
            value={value.tooth.dentition}
            options={[
              { value: "permanent", label: "Permanent" },
              { value: "primary", label: "Primary (out of scope)" },
              { value: "unknown", label: "Unknown" },
            ]}
            onChange={(dentition) =>
              onChange({ ...value, tooth: { ...value.tooth, dentition } })
            }
          />
          <SelectField
            label="Apex maturity"
            value={value.tooth.apexMaturity}
            options={[
              { value: "mature", label: "Mature" },
              { value: "open", label: "Open / immature" },
              { value: "unknown", label: "Unknown" },
            ]}
            onChange={(apexMaturity) =>
              onChange({ ...value, tooth: { ...value.tooth, apexMaturity } })
            }
          />
          <SelectField
            label="Treatment history"
            value={value.treatmentHistory}
            options={[
              { value: "untreated", label: "Untreated" },
              {
                value: "previously_initiated",
                label: "Previously initiated (out of scope)",
              },
              {
                value: "previously_obturated",
                label: "Previously obturated (out of scope)",
              },
              { value: "regenerative", label: "Regenerative (out of scope)" },
              { value: "unknown", label: "Unknown" },
            ]}
            onChange={(treatmentHistory) =>
              onChange({ ...value, treatmentHistory })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Symptoms</h3>
        <div className="input-grid">
          {(
            [
              ["spontaneousPain", "Spontaneous pain"],
              ["nocturnalPain", "Nocturnal pain"],
              ["posturalPain", "Postural pain"],
              ["referredPain", "Referred pain"],
              ["thermalPainHistory", "Thermal pain history"],
              ["coldRelievesPain", "Cold relieves pain"],
            ] as const
          ).map(([key, label]) => (
            <SelectField
              key={key}
              label={label}
              value={value.symptoms[key]}
              options={[...TRIAD]}
              onChange={(v) =>
                onChange({
                  ...value,
                  symptoms: { ...value.symptoms, [key]: v },
                })
              }
            />
          ))}
          <SelectField
            label="Heat response"
            value={value.symptoms.heatResponse}
            options={[
              { value: "not_performed", label: "Not performed" },
              { value: "unknown", label: "Unknown" },
              { value: "none", label: "None" },
              { value: "mild", label: "Mild" },
              { value: "severe", label: "Severe" },
              { value: "relieved_by_cold", label: "Relieved by cold" },
              { value: "unable_to_test", label: "Unable to test" },
            ]}
            onChange={(heatResponse) =>
              onChange({
                ...value,
                symptoms: { ...value.symptoms, heatResponse },
              })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Visual / soft tissue</h3>
        <div className="input-grid">
          {(
            [
              ["sinusTract", "Sinus tract"],
              ["sinusTractTraced", "Sinus tract traced"],
              ["swelling", "Swelling"],
              ["fluctuance", "Fluctuance"],
              ["pus", "Pus"],
              ["rapidOnsetSwelling", "Rapid-onset swelling"],
              ["fever", "Fever"],
              ["lymphadenopathy", "Lymphadenopathy"],
              ["decay", "Decay"],
              ["deepCariesOrExposure", "Deep caries / exposure"],
              ["crownPresent", "Crown present"],
              ["crackSuspected", "Crack suspected"],
              ["traumaSigns", "Trauma signs"],
            ] as const
          ).map(([key, label]) => (
            <SelectField
              key={key}
              label={label}
              value={value.visual[key]}
              options={[...TRIAD]}
              onChange={(v) =>
                onChange({
                  ...value,
                  visual: { ...value.visual, [key]: v },
                })
              }
            />
          ))}
          <SelectField
            label="Swelling severity"
            value={value.visual.swellingSeverity}
            options={[...SEVERITY]}
            onChange={(swellingSeverity) =>
              onChange({
                ...value,
                visual: { ...value.visual, swellingSeverity },
              })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Sensibility tests (with controls)</h3>
        <div className="input-grid">
          <SelectField
            label="Cold"
            value={value.clinical.cold}
            options={[
              { value: "not_performed", label: "Not performed" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "normal", label: "Normal" },
              {
                value: "exaggerated_non_lingering",
                label: "Exaggerated, non-lingering",
              },
              { value: "lingering", label: "Lingering" },
              { value: "negative", label: "Negative" },
            ]}
            onChange={(cold) =>
              onChange({ ...value, clinical: { ...value.clinical, cold } })
            }
          />
          <label className="input-field">
            <span>Cold linger seconds (raw, optional)</span>
            <input
              type="number"
              min={1}
              value={value.clinical.coldLingerSeconds ?? ""}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    coldLingerSeconds: e.target.value
                      ? Number(e.target.value)
                      : undefined,
                  },
                })
              }
            />
          </label>
          <SelectField
            label="Cold vs control"
            value={value.clinical.coldComparedToControl}
            options={[
              { value: "not_compared", label: "Not compared" },
              { value: "unknown", label: "Unknown" },
              { value: "similar_to_control", label: "Similar to control" },
              {
                value: "exaggerated_vs_control",
                label: "Exaggerated vs control",
              },
              { value: "reduced_vs_control", label: "Reduced vs control" },
              {
                value: "absent_vs_control_positive",
                label: "Absent vs control positive",
              },
            ]}
            onChange={(coldComparedToControl) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, coldComparedToControl },
              })
            }
          />
          <SelectField
            label="Cold validity"
            value={value.clinical.coldValidity}
            options={[
              { value: "unknown", label: "Unknown" },
              { value: "valid", label: "Valid" },
              { value: "invalid", label: "Invalid" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "not_performed", label: "Not performed" },
            ]}
            onChange={(coldValidity) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, coldValidity },
              })
            }
          />
          <SelectField
            label="Cold repeated"
            value={value.clinical.coldRepeated}
            options={[...TRIAD]}
            onChange={(coldRepeated) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, coldRepeated },
              })
            }
          />
          <SelectField
            label="EPT"
            value={value.clinical.ept}
            options={[
              { value: "not_performed", label: "Not performed" },
              { value: "positive", label: "Positive" },
              { value: "negative", label: "Negative" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "invalid", label: "Invalid" },
            ]}
            onChange={(ept) =>
              onChange({ ...value, clinical: { ...value.clinical, ept } })
            }
          />
          <SelectField
            label="EPT vs control"
            value={value.clinical.eptComparedToControl}
            options={[
              { value: "not_compared", label: "Not compared" },
              { value: "unknown", label: "Unknown" },
              { value: "similar_to_control", label: "Similar to control" },
              {
                value: "exaggerated_vs_control",
                label: "Exaggerated vs control",
              },
              { value: "reduced_vs_control", label: "Reduced vs control" },
              {
                value: "absent_vs_control_positive",
                label: "Absent vs control positive",
              },
            ]}
            onChange={(eptComparedToControl) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, eptComparedToControl },
              })
            }
          />
          <SelectField
            label="EPT validity"
            value={value.clinical.eptValidity}
            options={[
              { value: "unknown", label: "Unknown" },
              { value: "valid", label: "Valid" },
              { value: "invalid", label: "Invalid" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "not_performed", label: "Not performed" },
            ]}
            onChange={(eptValidity) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, eptValidity },
              })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Mechanical & structural</h3>
        <div className="input-grid">
          <SelectField
            label="Percussion severity"
            value={value.clinical.percussion}
            options={[...SEVERITY]}
            onChange={(percussion) =>
              onChange({ ...value, clinical: { ...value.clinical, percussion } })
            }
          />
          <SelectField
            label="Palpation severity"
            value={value.clinical.palpation}
            options={[...SEVERITY]}
            onChange={(palpation) =>
              onChange({ ...value, clinical: { ...value.clinical, palpation } })
            }
          />
          <SelectField
            label="Biting severity"
            value={value.clinical.biting}
            options={[...SEVERITY]}
            onChange={(biting) =>
              onChange({ ...value, clinical: { ...value.clinical, biting } })
            }
          />
          <SelectField
            label="Tooth Slooth (crack/bite — not sensibility)"
            value={value.clinical.toothSloothBiting}
            options={[
              { value: "not_performed", label: "Not performed" },
              { value: "positive", label: "Positive" },
              { value: "negative", label: "Negative" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "invalid", label: "Invalid" },
            ]}
            onChange={(toothSloothBiting) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, toothSloothBiting },
              })
            }
          />
          <SelectField
            label="Transillumination (structural)"
            value={value.clinical.transillumination}
            options={[
              { value: "not_performed", label: "Not performed" },
              { value: "positive", label: "Positive" },
              { value: "negative", label: "Negative" },
              { value: "unable_to_test", label: "Unable to test" },
              { value: "invalid", label: "Invalid" },
            ]}
            onChange={(transillumination) =>
              onChange({
                ...value,
                clinical: { ...value.clinical, transillumination },
              })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Periodontal</h3>
        <div className="input-grid">
          <SelectField
            label="Isolated deep pocket"
            value={value.periodontal.isolatedDeepPocket}
            options={[...TRIAD]}
            onChange={(isolatedDeepPocket) =>
              onChange({
                ...value,
                periodontal: { ...value.periodontal, isolatedDeepPocket },
              })
            }
          />
          <SelectField
            label="Mobility"
            value={value.periodontal.mobility}
            options={[
              { value: "not_assessed", label: "Not assessed" },
              { value: "unknown", label: "Unknown" },
              { value: "0", label: "0" },
              { value: "1", label: "1" },
              { value: "2", label: "2" },
              { value: "3", label: "3" },
            ]}
            onChange={(mobility) =>
              onChange({
                ...value,
                periodontal: { ...value.periodontal, mobility },
              })
            }
          />
          <label className="input-field">
            <span>Probing depths (optional text)</span>
            <input
              type="text"
              value={value.periodontal.probingDepthsMm ?? ""}
              onChange={(e) =>
                onChange({
                  ...value,
                  periodontal: {
                    ...value.periodontal,
                    probingDepthsMm: e.target.value || undefined,
                  },
                })
              }
            />
          </label>
        </div>
      </div>

      <div className="input-category">
        <h3>Imaging (clinician-assessed)</h3>
        <div className="input-grid">
          {(
            [
              ["periapicalRadiolucency", "PARL"],
              ["jShapedPeriapicalRadiolucency", "J-shaped RL"],
              ["widenedPeriodontalLigament", "Widened PDL"],
              ["laminaDuraLoss", "Lamina dura loss"],
              ["internalResorption", "Internal resorption"],
              ["externalResorption", "External resorption"],
            ] as const
          ).map(([key, label]) => (
            <SelectField
              key={key}
              label={label}
              value={value.imaging[key]}
              options={[...IMAGING]}
              onChange={(v) =>
                onChange({
                  ...value,
                  imaging: { ...value.imaging, [key]: v },
                })
              }
            />
          ))}
          <SelectField
            label="Multiple PA views"
            value={value.imaging.multiplePaViews}
            options={[...TRIAD]}
            onChange={(multiplePaViews) =>
              onChange({
                ...value,
                imaging: { ...value.imaging, multiplePaViews },
              })
            }
          />
        </div>
      </div>

      <div className="input-category">
        <h3>Confounders</h3>
        <div className="input-grid">
          {(
            [
              ["recentAnesthesia", "Recent anesthesia"],
              ["calcificationSuspected", "Calcification suspected"],
              ["poorIsolation", "Poor isolation"],
              ["generalizedLowResponsiveness", "Generalized low responsiveness"],
              ["recentTrauma", "Recent trauma"],
            ] as const
          ).map(([key, label]) => (
            <SelectField
              key={key}
              label={label}
              value={value.confounders[key]}
              options={[...TRIAD]}
              onChange={(v) =>
                onChange({
                  ...value,
                  confounders: { ...value.confounders, [key]: v },
                })
              }
            />
          ))}
        </div>
      </div>

      <label className="input-field input-notes">
        <span>Additional notes (no patient identifiers)</span>
        <textarea
          rows={3}
          value={value.additionalNotes ?? ""}
          onChange={(e) =>
            onChange({ ...value, additionalNotes: e.target.value })
          }
        />
      </label>
    </>
  );
}
