"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ACCESS_PROXY_VALUES,
  ECR_DIFFERENTIAL_STATUSES,
  REVIEW_DECISIONS,
  STRUCTURAL_CONTINUITY_VALUES,
  type EcrResult,
} from "@/lib/ecr/schemas";

type FormState = {
  toothLabel: string;
  mostApicalExtentMmFromCEJ: string;
  rootLengthCejToApexMm: string;
  localCrestDistanceMmFromCEJ: string;
  maximumCircumferenceDegrees: string;
  minimumLesionCanalSeparationMm: string;
  lesionCanalContactOrIntersection: boolean;
  continuousDentineBarrierVisible: boolean;
  ecrDifferential: (typeof ECR_DIFFERENTIAL_STATUSES)[number];
  maskReviewStatus: "clinician_reviewed" | "clinician_corrected" | "proposed";
  externalAccessProxy: (typeof ACCESS_PROXY_VALUES)[number];
  internalAccessProxy: (typeof ACCESS_PROXY_VALUES)[number];
  structuralContinuity: (typeof STRUCTURAL_CONTINUITY_VALUES)[number];
  qualityStatus: "pass" | "conditional" | "fail";
};

const initial: FormState = {
  toothLabel: "11",
  mostApicalExtentMmFromCEJ: "4.8",
  rootLengthCejToApexMm: "16",
  localCrestDistanceMmFromCEJ: "2.0",
  maximumCircumferenceDegrees: "142",
  minimumLesionCanalSeparationMm: "0",
  lesionCanalContactOrIntersection: true,
  continuousDentineBarrierVisible: false,
  ecrDifferential: "appearance_consistent_with_ecr",
  maskReviewStatus: "clinician_reviewed",
  externalAccessProxy: "favorable",
  internalAccessProxy: "possible",
  structuralContinuity: "reduced",
  qualityStatus: "pass",
};

export default function EcrPage() {
  const [form, setForm] = useState<FormState>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EcrResult | null>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [reviewerId, setReviewerId] = useState("");
  const [reviewNotes, setReviewNotes] = useState("");
  const [reviewMsg, setReviewMsg] = useState<string | null>(null);

  const num = (s: string) => {
    const n = Number.parseFloat(s);
    return Number.isFinite(n) ? n : null;
  };

  async function runAnalysis() {
    setBusy(true);
    setError(null);
    setReviewMsg(null);
    try {
      const measurements = {
        toothLabel: form.toothLabel,
        rootsAffected: ["single_root"],
        mostApicalExtentMmFromCEJ: num(form.mostApicalExtentMmFromCEJ),
        rootLengthCejToApexMm: num(form.rootLengthCejToApexMm),
        localCrestDistanceMmFromCEJ: num(form.localCrestDistanceMmFromCEJ),
        maximumCircumferenceDegrees: num(form.maximumCircumferenceDegrees),
        minimumLesionCanalSeparationMm: num(form.minimumLesionCanalSeparationMm),
        separationUncertaintyLowerBoundMm: num(form.minimumLesionCanalSeparationMm),
        lesionCanalContactOrIntersection: form.lesionCanalContactOrIntersection,
        continuousDentineBarrierVisible: form.continuousDentineBarrierVisible,
        ecrDifferential: form.ecrDifferential,
        maskReviewStatus: form.maskReviewStatus,
        scan: {
          qualityStatus: form.qualityStatus,
          oodStatus: "in_domain" as const,
          nativeVoxelMm: [0.2, 0.2, 0.2] as [number, number, number],
          fovCompleteForTarget: true,
          targetStructuresVisible: {
            crownRootComplex: true,
            cejRegion: true,
            localAlveolarCrest: true,
            apex: true,
            canalBoundaryNearLesion: true,
            lesionMargins: true,
          },
          artifactWarnings: [],
          qualityGateFailures: [],
        },
        planning: {
          portalSurface: "buccal",
          portalAreaMm2: 3.1,
          portalSupracrestal: "present" as const,
          lesionVolumeMm3: 18.4,
          lesionMaxDepthMm: 2.2,
          lesionMaxWidthMm: 3.0,
          externalAccessProxy: form.externalAccessProxy,
          internalAccessProxy: form.internalAccessProxy,
          structuralContinuity: form.structuralContinuity,
          furcationInvolvement: "not_applicable" as const,
          boneCrestLossProxy: "present" as const,
          adjacentAnatomyWarnings: [] as string[],
          existingTreatmentFindings: [] as string[],
          fractureRiskProxy: "moderate" as const,
          rootFormForReplantation: "compatible" as const,
        },
      };

      const res = await fetch("/api/v1/ecr/analyses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          toothLabel: form.toothLabel,
          seriesLabel: "phase1-measurement-session",
          measurements,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setAnalysisId(data.analysisId);
      setResult(data.result as EcrResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitReview(decision: (typeof REVIEW_DECISIONS)[number]) {
    if (!analysisId || reviewerId.trim().length < 2) return;
    setReviewMsg(null);
    try {
      const res = await fetch(`/api/v1/ecr/analyses/${analysisId}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision,
          reviewerId,
          reviewerNotes: reviewNotes || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setResult(data.result as EcrResult);
      setReviewMsg(`Review recorded: ${decision}`);
    } catch (e) {
      setReviewMsg(e instanceof Error ? e.message : "Review failed");
    }
  }

  const optionCards = useMemo(() => result?.managementSupport.options ?? [], [result]);

  return (
    <div className="shell">
      <header>
        <p className="section-label">
          <Link href="/">← AAE clinical agent</Link>
        </p>
        <h1 className="page-title">ECR CBCT analysis</h1>
        <p className="page-lede">
          Patel three-dimensional external cervical resorption classification from
          CBCT-derived measurements. Research/educational decision support —
          specialist confirmation required. Not a definitive treatment plan.
        </p>
      </header>

      <section className="input-category">
        <h3>Target & scan quality</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Tooth label (imaging)</span>
            <input
              value={form.toothLabel}
              onChange={(e) => setForm({ ...form, toothLabel: e.target.value })}
            />
          </label>
          <label className="input-field">
            <span>Quality status</span>
            <select
              value={form.qualityStatus}
              onChange={(e) =>
                setForm({
                  ...form,
                  qualityStatus: e.target.value as FormState["qualityStatus"],
                })
              }
            >
              <option value="pass">pass</option>
              <option value="conditional">conditional</option>
              <option value="fail">fail</option>
            </select>
          </label>
          <label className="input-field">
            <span>ECR differential</span>
            <select
              value={form.ecrDifferential}
              onChange={(e) =>
                setForm({
                  ...form,
                  ecrDifferential: e.target
                    .value as FormState["ecrDifferential"],
                })
              }
            >
              {ECR_DIFFERENTIAL_STATUSES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="input-field">
            <span>Mask review</span>
            <select
              value={form.maskReviewStatus}
              onChange={(e) =>
                setForm({
                  ...form,
                  maskReviewStatus: e.target
                    .value as FormState["maskReviewStatus"],
                })
              }
            >
              <option value="clinician_reviewed">clinician_reviewed</option>
              <option value="clinician_corrected">clinician_corrected</option>
              <option value="proposed">proposed</option>
            </select>
          </label>
        </div>
      </section>

      <section className="input-category">
        <h3>Patel measurements (tooth-aligned)</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>Most apical extent from CEJ (mm)</span>
            <input
              value={form.mostApicalExtentMmFromCEJ}
              onChange={(e) =>
                setForm({ ...form, mostApicalExtentMmFromCEJ: e.target.value })
              }
            />
          </label>
          <label className="input-field">
            <span>Root length CEJ→apex (mm)</span>
            <input
              value={form.rootLengthCejToApexMm}
              onChange={(e) =>
                setForm({ ...form, rootLengthCejToApexMm: e.target.value })
              }
            />
          </label>
          <label className="input-field">
            <span>Local crest distance from CEJ (mm)</span>
            <input
              value={form.localCrestDistanceMmFromCEJ}
              onChange={(e) =>
                setForm({
                  ...form,
                  localCrestDistanceMmFromCEJ: e.target.value,
                })
              }
            />
          </label>
          <label className="input-field">
            <span>Max circumference angle (°)</span>
            <input
              value={form.maximumCircumferenceDegrees}
              onChange={(e) =>
                setForm({
                  ...form,
                  maximumCircumferenceDegrees: e.target.value,
                })
              }
            />
          </label>
          <label className="input-field">
            <span>Min lesion–canal separation (mm)</span>
            <input
              value={form.minimumLesionCanalSeparationMm}
              onChange={(e) =>
                setForm({
                  ...form,
                  minimumLesionCanalSeparationMm: e.target.value,
                })
              }
            />
          </label>
        </div>
        <label className="input-field">
          <span>
            <input
              type="checkbox"
              checked={form.lesionCanalContactOrIntersection}
              onChange={(e) =>
                setForm({
                  ...form,
                  lesionCanalContactOrIntersection: e.target.checked,
                })
              }
            />{" "}
            Lesion contacts/intersects canal
          </span>
        </label>
        <label className="input-field">
          <span>
            <input
              type="checkbox"
              checked={form.continuousDentineBarrierVisible}
              onChange={(e) =>
                setForm({
                  ...form,
                  continuousDentineBarrierVisible: e.target.checked,
                })
              }
            />{" "}
            Continuous dentine barrier visible
          </span>
        </label>
      </section>

      <section className="input-category">
        <h3>Treatment-relevant imaging proxies</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>External access proxy</span>
            <select
              value={form.externalAccessProxy}
              onChange={(e) =>
                setForm({
                  ...form,
                  externalAccessProxy: e.target
                    .value as FormState["externalAccessProxy"],
                })
              }
            >
              {ACCESS_PROXY_VALUES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="input-field">
            <span>Internal access proxy</span>
            <select
              value={form.internalAccessProxy}
              onChange={(e) =>
                setForm({
                  ...form,
                  internalAccessProxy: e.target
                    .value as FormState["internalAccessProxy"],
                })
              }
            >
              {ACCESS_PROXY_VALUES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="input-field">
            <span>Structural continuity</span>
            <select
              value={form.structuralContinuity}
              onChange={(e) =>
                setForm({
                  ...form,
                  structuralContinuity: e.target
                    .value as FormState["structuralContinuity"],
                })
              }
            >
              {STRUCTURAL_CONTINUITY_VALUES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <div className="toolbar">
        <button type="button" className="btn" onClick={runAnalysis} disabled={busy}>
          {busy ? "Calculating…" : "Calculate Patel + options"}
        </button>
        {analysisId ? (
          <a
            className="btn btn--ghost"
            href={`/api/v1/ecr/analyses/${analysisId}/report?format=text`}
            target="_blank"
            rel="noreferrer"
          >
            Open report
          </a>
        ) : null}
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {result ? (
        <>
          <section className="output-block output-block--final">
            <h3 className="output-title">Patel components</h3>
            <p className="body-text">{result.outputLabel}</p>
            <div className="diagnosis-row">
              <div>
                <div className="diagnosis-label">Height</div>
                <div className="diagnosis-value">
                  {result.patel.height.value ?? "—"}
                  {result.patel.height.borderline ? " (borderline)" : ""}
                </div>
              </div>
              <div>
                <div className="diagnosis-label">Circumference</div>
                <div className="diagnosis-value">
                  {result.patel.circumference.value ?? "—"}{" "}
                  {result.patel.circumference.maximumAngleDegrees != null
                    ? `(${result.patel.circumference.maximumAngleDegrees}°)`
                    : ""}
                </div>
              </div>
              <div>
                <div className="diagnosis-label">Canal proximity</div>
                <div className="diagnosis-value">
                  {result.patel.canalProximity.value ?? "—"}
                </div>
              </div>
              <div>
                <div className="diagnosis-label">Code</div>
                <div className="diagnosis-value">{result.patel.code ?? "—"}</div>
              </div>
            </div>
            <p className="body-text body-text--tight">
              Status: {result.patel.status} · Differential:{" "}
              {result.ecrAssessment.status} · Quality: {result.scan.qualityStatus}
            </p>
            {result.warnings.length ? (
              <ul className="warnings">
                {result.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            ) : null}
          </section>

          <section className="output-block">
            <h3 className="output-title">Conditional management options</h3>
            <p className="scan-disclaimer">
              Multilabel imaging options only. No single “recommended treatment”
              button. Ruleset {result.managementSupport.rulesetVersion}.
            </p>
            {optionCards.length === 0 ? (
              <p className="body-text">No options ranked (abstained or incomplete).</p>
            ) : (
              <div className="evidence-grid">
                {optionCards.map((o) => (
                  <article key={o.option} className="output-block">
                    <div className="diagnosis-label">{o.status}</div>
                    <div className="diagnosis-value">{o.option}</div>
                    <p className="body-text body-text--tight">
                      Rules: {o.supportingRuleIds.join(", ")}
                    </p>
                    <div className="diagnosis-label">Requires clinical confirmation</div>
                    <ul className="warnings">
                      {o.requiresClinicalConfirmation.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </article>
                ))}
              </div>
            )}
            <div className="diagnosis-label">Cannot determine from CBCT</div>
            <ul className="warnings">
              {result.managementSupport.cannotDetermineFromCbct.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </section>

          <section className="output-block">
            <h3 className="output-title">Specialist review</h3>
            <label className="input-field">
              <span>Reviewer ID</span>
              <input
                value={reviewerId}
                onChange={(e) => setReviewerId(e.target.value)}
              />
            </label>
            <label className="input-field">
              <span>Notes</span>
              <textarea
                rows={3}
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
              />
            </label>
            <div className="toolbar toolbar--tight">
              {REVIEW_DECISIONS.map((d) => (
                <button
                  key={d}
                  type="button"
                  className="btn btn--ghost"
                  disabled={reviewerId.trim().length < 2}
                  onClick={() => submitReview(d)}
                >
                  {d}
                </button>
              ))}
            </div>
            {reviewMsg ? <p className="body-text">{reviewMsg}</p> : null}
          </section>
        </>
      ) : null}
    </div>
  );
}
