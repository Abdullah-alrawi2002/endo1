import type { ClinicalCase } from "@/lib/schemas/clinical-case";

type Props = {
  value: ClinicalCase;
  onChange: (next: ClinicalCase) => void;
};

export function ClinicalCaseForm({ value, onChange }: Props) {
  return (
    <>
      <div className="input-category">
        <h3>Visual</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Sinus tract</span>
            <select
              value={value.visual.sinusTract}
              onChange={(e) =>
                onChange({
                  ...value,
                  visual: {
                    ...value.visual,
                    sinusTract: e.target.value as ClinicalCase["visual"]["sinusTract"],
                  },
                })
              }
            >
              <option value="unknown">Unknown</option>
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>Swelling</span>
            <select
              value={value.visual.swelling}
              onChange={(e) =>
                onChange({
                  ...value,
                  visual: {
                    ...value.visual,
                    swelling: e.target.value as ClinicalCase["visual"]["swelling"],
                  },
                })
              }
            >
              <option value="unknown">Unknown</option>
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>Decay</span>
            <select
              value={value.visual.decay}
              onChange={(e) =>
                onChange({
                  ...value,
                  visual: {
                    ...value.visual,
                    decay: e.target.value as ClinicalCase["visual"]["decay"],
                  },
                })
              }
            >
              <option value="unknown">Unknown</option>
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>Signs of trauma</span>
            <select
              value={value.visual.traumaSigns}
              onChange={(e) =>
                onChange({
                  ...value,
                  visual: {
                    ...value.visual,
                    traumaSigns: e.target
                      .value as ClinicalCase["visual"]["traumaSigns"],
                  },
                })
              }
            >
              <option value="unknown">Unknown</option>
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
        </div>
      </div>

      <div className="input-category">
        <h3>Clinical</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Cold test</span>
            <select
              value={value.clinical.cold}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    cold: e.target.value as ClinicalCase["clinical"]["cold"],
                  },
                })
              }
            >
              <option value="normal">Normal</option>
              <option value="exaggerated_non_lingering">
                Exaggerated, non-lingering
              </option>
              <option value="lingering">Lingering</option>
              <option value="negative">Negative</option>
            </select>
          </label>
          {value.clinical.cold === "lingering" ? (
            <label className="input-field">
              <span>Linger duration (seconds)</span>
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
          ) : (
            <div />
          )}
          <label className="input-field">
            <span>Percussion</span>
            <select
              value={value.clinical.percussion}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    percussion: e.target
                      .value as ClinicalCase["clinical"]["percussion"],
                  },
                })
              }
            >
              <option value="negative">Negative</option>
              <option value="positive">Positive (pain)</option>
            </select>
          </label>
          <label className="input-field">
            <span>Palpation</span>
            <select
              value={value.clinical.palpation}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    palpation: e.target
                      .value as ClinicalCase["clinical"]["palpation"],
                  },
                })
              }
            >
              <option value="negative">Negative</option>
              <option value="positive">Positive</option>
            </select>
          </label>
          <label className="input-field">
            <span>EPT (optional)</span>
            <select
              value={value.clinical.ept}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    ept: e.target.value as ClinicalCase["clinical"]["ept"],
                  },
                })
              }
            >
              <option value="not_performed">Not performed</option>
              <option value="positive">Positive</option>
              <option value="negative">Negative</option>
            </select>
          </label>
          <label className="input-field">
            <span>Fluorescent light (optional)</span>
            <select
              value={value.clinical.fluorescentLight}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    fluorescentLight: e.target
                      .value as ClinicalCase["clinical"]["fluorescentLight"],
                  },
                })
              }
            >
              <option value="not_performed">Not performed</option>
              <option value="positive">Positive</option>
              <option value="negative">Negative</option>
            </select>
          </label>
          <label className="input-field">
            <span>Tooth Slooth biting (optional)</span>
            <select
              value={value.clinical.toothSloothBiting}
              onChange={(e) =>
                onChange({
                  ...value,
                  clinical: {
                    ...value.clinical,
                    toothSloothBiting: e.target
                      .value as ClinicalCase["clinical"]["toothSloothBiting"],
                  },
                })
              }
            >
              <option value="not_performed">Not performed</option>
              <option value="positive">Positive (pain on cusp bite)</option>
              <option value="negative">Negative</option>
            </select>
          </label>
        </div>
      </div>

      <div className="input-category">
        <h3>Imaging</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Periapical radiolucency</span>
            <select
              value={value.imaging.periapicalRadiolucency}
              onChange={(e) =>
                onChange({
                  ...value,
                  imaging: {
                    ...value.imaging,
                    periapicalRadiolucency: e.target
                      .value as ClinicalCase["imaging"]["periapicalRadiolucency"],
                  },
                })
              }
            >
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>J-shaped periapical radiolucency</span>
            <select
              value={value.imaging.jShapedPeriapicalRadiolucency}
              onChange={(e) =>
                onChange({
                  ...value,
                  imaging: {
                    ...value.imaging,
                    jShapedPeriapicalRadiolucency: e.target
                      .value as ClinicalCase["imaging"]["jShapedPeriapicalRadiolucency"],
                  },
                })
              }
            >
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>Widening of periodontal ligament</span>
            <select
              value={value.imaging.widenedPeriodontalLigament}
              onChange={(e) =>
                onChange({
                  ...value,
                  imaging: {
                    ...value.imaging,
                    widenedPeriodontalLigament: e.target
                      .value as ClinicalCase["imaging"]["widenedPeriodontalLigament"],
                  },
                })
              }
            >
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>Internal resorption</span>
            <select
              value={value.imaging.internalResorption}
              onChange={(e) =>
                onChange({
                  ...value,
                  imaging: {
                    ...value.imaging,
                    internalResorption: e.target
                      .value as ClinicalCase["imaging"]["internalResorption"],
                  },
                })
              }
            >
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
          <label className="input-field">
            <span>External resorption</span>
            <select
              value={value.imaging.externalResorption}
              onChange={(e) =>
                onChange({
                  ...value,
                  imaging: {
                    ...value.imaging,
                    externalResorption: e.target
                      .value as ClinicalCase["imaging"]["externalResorption"],
                  },
                })
              }
            >
              <option value="absent">Absent</option>
              <option value="present">Present</option>
            </select>
          </label>
        </div>
      </div>

      <label className="input-field input-notes">
        <span>Additional notes (optional)</span>
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
