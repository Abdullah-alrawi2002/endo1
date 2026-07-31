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
import type {
  PatelNetworkResult,
  ProvisionalPlan,
} from "@/lib/ecr/agents/schemas";

type FormState = {
  toothLabel: string;
  seriesLabel: string;
  dicomReference: string;
  mostApicalExtentMmFromCEJ: string;
  rootLengthCejToApexMm: string;
  localCrestDistanceMmFromCEJ: string;
  maximumCircumferenceDegrees: string;
  circumferenceUncertaintyDegrees: string;
  minimumLesionCanalSeparationMm: string;
  lesionCanalContactOrIntersection: boolean;
  continuousDentineBarrierVisible: boolean;
  ecrDifferential: (typeof ECR_DIFFERENTIAL_STATUSES)[number];
  maskReviewStatus: "clinician_reviewed" | "clinician_corrected" | "proposed";
  externalAccessProxy: (typeof ACCESS_PROXY_VALUES)[number];
  internalAccessProxy: (typeof ACCESS_PROXY_VALUES)[number];
  structuralContinuity: (typeof STRUCTURAL_CONTINUITY_VALUES)[number];
  qualityStatus: "pass" | "conditional" | "fail";
  portalSurface: string;
};

const initial: FormState = {
  toothLabel: "11",
  seriesLabel: "CBCT volume — ECR workup",
  dicomReference: "",
  mostApicalExtentMmFromCEJ: "4.8",
  rootLengthCejToApexMm: "16",
  localCrestDistanceMmFromCEJ: "2.0",
  maximumCircumferenceDegrees: "142",
  circumferenceUncertaintyDegrees: "",
  minimumLesionCanalSeparationMm: "0",
  lesionCanalContactOrIntersection: true,
  continuousDentineBarrierVisible: false,
  ecrDifferential: "appearance_consistent_with_ecr",
  maskReviewStatus: "clinician_reviewed",
  externalAccessProxy: "favorable",
  internalAccessProxy: "possible",
  structuralContinuity: "reduced",
  qualityStatus: "pass",
  portalSurface: "buccal",
};

function strategyLabel(s: string): string {
  return s.replace(/_/g, " ");
}

export default function EcrPage() {
  const [form, setForm] = useState<FormState>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EcrResult | null>(null);
  const [plan, setPlan] = useState<ProvisionalPlan | null>(null);
  const [network, setNetwork] = useState<PatelNetworkResult | null>(null);
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
        circumferenceUncertaintyDegrees: num(
          form.circumferenceUncertaintyDegrees,
        ),
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
          artifactWarnings: [] as string[],
          qualityGateFailures: [] as string[],
        },
        portalSurface: form.portalSurface || undefined,
        planning: {
          portalSurface: form.portalSurface || null,
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
          seriesLabel: form.seriesLabel || undefined,
          dicomReference: form.dicomReference || undefined,
          measurements,
          useAgentNetwork: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setAnalysisId(data.analysisId);
      setResult(data.result as EcrResult);
      setPlan((data.provisionalPlan as ProvisionalPlan) ?? null);
      setNetwork((data.patelNetwork as PatelNetworkResult) ?? null);
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

  const candidates = useMemo(() => plan?.candidates ?? [], [plan]);
  const alternatives = useMemo(() => plan?.alternativeStrategies ?? [], [plan]);

  return (
    <div className="shell shell--wide">
      <header>
        <p className="section-label">
          <Link href="/">← AAE clinical diagnosis</Link>
        </p>
        <h1 className="page-title">ECR CBCT treatment planning</h1>
        <p className="page-lede">
          Classify external cervical resorption on CBCT with the Patel system,
          then generate a provisional, multilabel treatment plan for that tooth.
          Measurements must be reviewed from the 3D volume. Automatic ECR
          segmentation is not yet enabled — this path is measurement-driven and
          specialist-confirmed. Not a definitive treatment plan.
        </p>
        <ol className="workflow-steps">
          <li>
            <strong>1. CBCT evidence</strong> — tooth, quality, differential,
            reviewed measurements
          </li>
          <li>
            <strong>2. Patel classification</strong> — height × circumference ×
            canal (d/p)
          </li>
          <li>
            <strong>3. Treatment plan</strong> — overlapping conditional options
            + clinician review
          </li>
        </ol>
      </header>

      <section className="input-category">
        <h3>1 · CBCT case & quality</h3>
        <p className="body-text body-text--tight">
          Enter findings after reviewing the CBCT volume for the target tooth.
          Unreviewed masks and non-ECR differentials abstain from classification
          and planning.
        </p>
        <div className="input-grid">
          <label className="input-field">
            <span>Tooth (imaging label)</span>
            <input
              value={form.toothLabel}
              onChange={(e) => setForm({ ...form, toothLabel: e.target.value })}
            />
          </label>
          <label className="input-field">
            <span>Series / study label</span>
            <input
              value={form.seriesLabel}
              onChange={(e) => setForm({ ...form, seriesLabel: e.target.value })}
            />
          </label>
          <label className="input-field">
            <span>DICOM / PACS reference (optional)</span>
            <input
              placeholder="Study UID or Orthanc ID"
              value={form.dicomReference}
              onChange={(e) =>
                setForm({ ...form, dicomReference: e.target.value })
              }
            />
          </label>
          <label className="input-field">
            <span>Scan quality</span>
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
            <span>Mask / measurement review</span>
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
              <option value="proposed">proposed (will abstain)</option>
            </select>
          </label>
        </div>
      </section>

      <section className="input-category">
        <h3>1 · Patel measurements from CBCT</h3>
        <p className="body-text body-text--tight">
          Tooth-aligned: most apical lesion extent vs CEJ and crest, root
          length, maximum circumferential angle, lesion–canal relation.
        </p>
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
            <span>Root length CEJ → apex (mm)</span>
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
            <span>Circumference uncertainty ± (°)</span>
            <input
              placeholder="optional"
              value={form.circumferenceUncertaintyDegrees}
              onChange={(e) =>
                setForm({
                  ...form,
                  circumferenceUncertaintyDegrees: e.target.value,
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
          <label className="input-field">
            <span>Portal surface</span>
            <input
              value={form.portalSurface}
              onChange={(e) =>
                setForm({ ...form, portalSurface: e.target.value })
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
            Lesion contacts / intersects canal (→ Patel p)
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
            Continuous dentine barrier visible (→ Patel d)
          </span>
        </label>
      </section>

      <section className="input-category">
        <h3>1 · Access & structure (planning proxies)</h3>
        <div className="input-grid">
          <label className="input-field">
            <span>External access (imaging)</span>
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
            <span>Internal access (imaging)</span>
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
          {busy
            ? "Classifying & planning…"
            : "Classify Patel + generate treatment plan"}
        </button>
        {analysisId ? (
          <a
            className="btn btn--ghost"
            href={`/api/v1/ecr/analyses/${analysisId}/report?format=text`}
            target="_blank"
            rel="noreferrer"
          >
            Open plan report
          </a>
        ) : null}
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {result ? (
        <>
          <section className="output-block output-block--final">
            <h3 className="output-title">2 · Patel classification</h3>
            <p className="body-text">{result.outputLabel}</p>
            <div className="patel-code-hero">
              <span className="patel-code-hero__label">Patel code</span>
              <span className="patel-code-hero__value">
                {result.patel.code ?? "—"}
              </span>
              {network?.requiresSpecialistReview ? (
                <span className="patel-code-hero__flag">
                  Needs specialist review
                </span>
              ) : null}
            </div>
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
                  {result.patel.circumference.value ?? "—"}
                  {result.patel.circumference.maximumAngleDegrees != null
                    ? ` (${result.patel.circumference.maximumAngleDegrees}°)`
                    : ""}
                </div>
              </div>
              <div>
                <div className="diagnosis-label">Canal proximity</div>
                <div className="diagnosis-value">
                  {result.patel.canalProximity.value ?? "—"}
                </div>
                <p className="body-text body-text--tight">
                  p = imaging canal involvement, not pulp necrosis
                </p>
              </div>
              <div>
                <div className="diagnosis-label">Network</div>
                <div className="diagnosis-value">
                  {network?.classificationStatus ?? result.patel.status}
                </div>
                <p className="body-text body-text--tight">
                  {network?.decisionSource ?? "deterministic"}
                </p>
              </div>
            </div>
            <p className="body-text body-text--tight">
              Differential: {result.ecrAssessment.status} · Quality:{" "}
              {result.scan.qualityStatus}
              {analysisId ? ` · ID ${analysisId.slice(0, 8)}…` : ""}
            </p>
            {result.warnings.length ? (
              <ul className="warnings">
                {result.warnings.slice(0, 12).map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            ) : null}
          </section>

          <section className="output-block output-block--plan">
            <h3 className="output-title">3 · Provisional treatment plan</h3>
            {plan ? (
              <>
                <p className="scan-disclaimer">
                  Status: <strong>{plan.planStatus}</strong> · Safety:{" "}
                  {plan.safetyStatus} · Critic: {plan.criticStatus}. Overlapping
                  alternatives are preserved (e.g. 2Bp may support both external
                  repair with RCT consideration and internal repair with RCT).
                  CBCT alone cannot finalize vitality, restorability, or patient
                  preference.
                </p>
                <div className="plan-primary">
                  <div className="diagnosis-label">Primary imaging strategy</div>
                  <div className="diagnosis-value">
                    {plan.primaryStrategy
                      ? strategyLabel(plan.primaryStrategy)
                      : "None — specialist review"}
                  </div>
                </div>

                {candidates.length ? (
                  <div className="evidence-grid">
                    {candidates.map((c) => (
                      <article key={c.strategy} className="plan-candidate">
                        <div className="diagnosis-label">{c.status}</div>
                        <div className="diagnosis-value">
                          {strategyLabel(c.strategy)}
                        </div>
                        <p className="body-text body-text--tight">
                          {c.treatmentObjective}
                        </p>
                        {c.proceduralSequence.length ? (
                          <>
                            <div className="diagnosis-label">Sequence</div>
                            <ol className="plan-sequence">
                              {c.proceduralSequence.map((step) => (
                                <li key={step}>{step}</li>
                              ))}
                            </ol>
                          </>
                        ) : null}
                        <div className="diagnosis-label">
                          Required clinical confirmations
                        </div>
                        <ul className="warnings">
                          {c.requiredClinicalConfirmations.map((x) => (
                            <li key={x}>{x}</li>
                          ))}
                        </ul>
                        {c.activateIf.length ? (
                          <>
                            <div className="diagnosis-label">Activate if</div>
                            <ul className="warnings">
                              {c.activateIf.map((x) => (
                                <li key={x}>{x}</li>
                              ))}
                            </ul>
                          </>
                        ) : null}
                        {c.rejectIf.length ? (
                          <>
                            <div className="diagnosis-label">Reject / stop if</div>
                            <ul className="warnings">
                              {c.rejectIf.map((x) => (
                                <li key={x}>{x}</li>
                              ))}
                            </ul>
                          </>
                        ) : null}
                        <p className="body-text body-text--tight">
                          Rules: {c.sourceRuleIds.join(", ") || "—"}
                        </p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="body-text">
                    No treatment candidates generated — check classification
                    status and specialist review flags.
                  </p>
                )}

                {alternatives.length > 0 &&
                alternatives.some(
                  (a) => a.strategy !== plan.primaryStrategy,
                ) ? (
                  <>
                    <div className="diagnosis-label">
                      Alternative strategies retained
                    </div>
                    <ul className="warnings">
                      {alternatives.map((a) => (
                        <li key={a.strategy}>
                          [{a.status}] {strategyLabel(a.strategy)}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : null}

                {plan.decisionCheckpoints.length ? (
                  <>
                    <div className="diagnosis-label">Decision checkpoints</div>
                    <ul className="warnings">
                      {plan.decisionCheckpoints.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </>
                ) : null}

                {plan.unsupportedVariables.length ? (
                  <>
                    <div className="diagnosis-label">
                      Cannot determine from CBCT alone
                    </div>
                    <ul className="warnings">
                      {plan.unsupportedVariables.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </>
                ) : null}
              </>
            ) : (
              <p className="body-text">
                Provisional plan not returned. Showing ESE option cards from the
                deterministic engine below.
              </p>
            )}
          </section>

          {!plan && result.managementSupport.options.length ? (
            <section className="output-block">
              <h3 className="output-title">Conditional management options</h3>
              <div className="evidence-grid">
                {result.managementSupport.options.map((o) => (
                  <article key={o.option} className="output-block">
                    <div className="diagnosis-label">{o.status}</div>
                    <div className="diagnosis-value">
                      {strategyLabel(o.option)}
                    </div>
                    <ul className="warnings">
                      {o.requiresClinicalConfirmation.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          <section className="output-block">
            <h3 className="output-title">Clinician review</h3>
            <p className="body-text body-text--tight">
              Confirm or modify after interpreting the entire CBCT volume and
              clinical findings (sensibility, probing, restorability).
            </p>
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
