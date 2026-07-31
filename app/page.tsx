"use client";

import { ClinicalCaseForm } from "@/components/ClinicalCaseForm";
import { CTAnalysisUpload } from "@/components/CTAnalysisUpload";
import { useEffect, useMemo, useState } from "react";
import {
  APICAL_DIAGNOSES,
  type ClinicalCase,
  defaultClinicalCase,
  ERROR_TYPES,
  type FinalDiagnosis,
  PULPAL_DIAGNOSES,
  TAXONOMY_VERSION,
  type DiagnosticStatus,
} from "@/lib/schemas/clinical-case";

type StreamEvent =
  | { type: "status"; message: string }
  | { type: "rag"; matchCount: number; contextBlock: string }
  | { type: "stage"; stage: 1 | 2 | 3 | 4; title: string; content: string }
  | {
      type: "evidence";
      evidenceFor: FinalDiagnosis["evidenceFor"];
      evidenceAgainst: FinalDiagnosis["evidenceAgainst"];
      conflicts: string[];
      missingRequiredData: string[];
      recommendedNextTests: string[];
    }
  | { type: "final"; result: FinalDiagnosis }
  | { type: "error"; message: string };

type FeatureFlags = {
  ctModuleEnabled: boolean;
  correctionRagEnabled: boolean;
  taxonomyVersion: string;
  mvpMode: string;
  ctUploadUrl: string | null;
};

export default function Home() {
  const [clinicalCase, setClinicalCase] =
    useState<ClinicalCase>(defaultClinicalCase);
  const [features, setFeatures] = useState<FeatureFlags | null>(null);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [corrStatus, setCorrStatus] = useState<DiagnosticStatus>("diagnosable");
  const [corrPulpal, setCorrPulpal] = useState<
    (typeof PULPAL_DIAGNOSES)[number] | ""
  >("");
  const [corrApical, setCorrApical] = useState<
    (typeof APICAL_DIAGNOSES)[number] | ""
  >("");
  const [corrReason, setCorrReason] = useState("");
  const [corrMisunderstood, setCorrMisunderstood] = useState("");
  const [corrErrorTypes, setCorrErrorTypes] = useState<string[]>([
    "invalid_test_interpretation",
  ]);
  const [specialistIdentity, setSpecialistIdentity] = useState("");
  const [corrSaving, setCorrSaving] = useState(false);
  const [corrMessage, setCorrMessage] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);

  useEffect(() => {
    fetch("/api/features")
      .then((r) => r.json())
      .then((data: FeatureFlags) => setFeatures(data))
      .catch(() =>
        setFeatures({
          ctModuleEnabled: false,
          correctionRagEnabled: true,
          taxonomyVersion: TAXONOMY_VERSION,
          mvpMode: "clinical_agentic",
          ctUploadUrl: null,
        }),
      );
  }, []);

  const finalResult = useMemo(
    () =>
      [...events].reverse().find((e) => e.type === "final") as
        | Extract<StreamEvent, { type: "final" }>
        | undefined,
    [events],
  );

  const evidenceEvent = useMemo(
    () =>
      [...events].reverse().find((e) => e.type === "evidence") as
        | Extract<StreamEvent, { type: "evidence" }>
        | undefined,
    [events],
  );

  async function runDiagnosis() {
    setLoading(true);
    setError(null);
    setEvents([]);
    try {
      const res = await fetch("/api/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...clinicalCase,
          ctSupport: undefined,
        }),
      });
      if (!res.ok || !res.body) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          try {
            const ev = JSON.parse(line) as StreamEvent;
            setEvents((prev) => [...prev, ev]);
            if (ev.type === "error") setError(ev.message);
          } catch {
            setError("Malformed stream line");
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function submitCorrection() {
    if (!finalResult || !features?.correctionRagEnabled) return;
    setCorrSaving(true);
    setCorrMessage(null);
    try {
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case: { ...clinicalCase, ctSupport: undefined },
          agentStatus: finalResult.result.status,
          agentPulpal: finalResult.result.pulpalDiagnosis,
          agentApical: finalResult.result.apicalDiagnosis,
          adjudicatedReferenceDiagnosis: {
            status: corrStatus,
            pulpal: corrPulpal || null,
            apical: corrApical || null,
          },
          correctionReasoning: corrReason,
          errorTypes: corrErrorTypes,
          specialistIdentity,
          taxonomyVersion: TAXONOMY_VERSION,
          approvalStatus: "pending",
          misunderstoodSummary: corrMisunderstood || undefined,
        }),
      });
      const data = (await res.json()) as { error?: string };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setCorrMessage("Correction saved to local RAG memory.");
      setShowCorrection(false);
      setCorrReason("");
      setCorrMisunderstood("");
    } catch (e) {
      setCorrMessage(e instanceof Error ? e.message : "Save failed");
    } finally {
      setCorrSaving(false);
    }
  }

  async function exportCorrections() {
    setExportBusy(true);
    try {
      const res = await fetch("/api/corrections?limit=200");
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `endo-corrections-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExportBusy(false);
    }
  }

  return (
    <div className="shell">
      <header>
        <h1 className="page-title">Endodontic diagnosis</h1>
        <p className="page-lede">
          Open this site in any browser — no install. Agentic multi-stage
          diagnosis · taxonomy {TAXONOMY_VERSION} · correction memory · can
          abstain · clinician confirmation required. Educational decision
          support only — not a medical device.
        </p>
        <p className="page-lede">
          Separate product:{" "}
          <a href="/ecr">ECR CBCT treatment planning</a> (Patel classification
          → provisional options for external cervical resorption).
        </p>
        {features ? (
          <p className="page-lede">
            Correction RAG: {features.correctionRagEnabled ? "on" : "off"} · CT
            module: {features.ctModuleEnabled ? "on" : "off"}
          </p>
        ) : null}
      </header>

      <p className="section-label">Case</p>
      <ClinicalCaseForm value={clinicalCase} onChange={setClinicalCase} />

      {features?.ctModuleEnabled ? (
        <CTAnalysisUpload
          analysisId={clinicalCase.ctAnalysisId}
          ctUploadUrl={features.ctUploadUrl}
          onChange={(ctAnalysisId) =>
            setClinicalCase((current) => ({ ...current, ctAnalysisId }))
          }
        />
      ) : (
        <p className="scan-disclaimer">
          CBCT support is off (`ENABLE_CT_MODULE=false`). Use the all-in-one
          Docker host (see SHARE.md) to run web + CT together on one link.
        </p>
      )}

      <div className="toolbar">
        <button
          type="button"
          className="btn"
          onClick={runDiagnosis}
          disabled={loading}
        >
          {loading ? "Running…" : "Run"}
        </button>
        {features?.correctionRagEnabled ? (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={exportCorrections}
            disabled={exportBusy}
          >
            {exportBusy ? "Exporting…" : "Export corrections"}
          </button>
        ) : null}
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {events.length > 0 ? (
        <section className="output-stack">
          {events.map((ev, i) => {
            if (ev.type === "status") {
              return (
                <p key={`${i}-status`} className="body-text body-text--tight">
                  {ev.message}
                </p>
              );
            }
            if (ev.type === "rag") {
              return (
                <article key={`${i}-rag`} className="output-block">
                  <h3 className="output-title">
                    Similar corrections ({ev.matchCount})
                  </h3>
                  {ev.matchCount === 0 || !ev.contextBlock ? (
                    <p className="body-text body-text--tight">None retrieved.</p>
                  ) : (
                    <pre className="output-pre">{ev.contextBlock}</pre>
                  )}
                </article>
              );
            }
            if (ev.type === "stage") {
              return (
                <article key={`${i}-s${ev.stage}`} className="output-block">
                  <h3 className="output-title">{ev.title}</h3>
                  <pre className="output-pre">{ev.content}</pre>
                </article>
              );
            }
            return null;
          })}
        </section>
      ) : null}

      {evidenceEvent ? (
        <section className="output-block">
          <h3 className="output-title">Evidence table</h3>
          <div className="evidence-grid">
            <div>
              <div className="diagnosis-label">Supporting</div>
              <ul className="warnings">
                {evidenceEvent.evidenceFor.map((item) => (
                  <li key={`for-${item.claim}`}>
                    [{item.source}] {item.claim}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <div className="diagnosis-label">Against / conflicts</div>
              <ul className="warnings">
                {evidenceEvent.evidenceAgainst.map((item) => (
                  <li key={`against-${item.claim}`}>
                    [{item.source}] {item.claim}
                  </li>
                ))}
                {evidenceEvent.conflicts.map((c) => (
                  <li key={`conflict-${c}`}>{c}</li>
                ))}
              </ul>
            </div>
          </div>
          {evidenceEvent.missingRequiredData.length ? (
            <>
              <div className="diagnosis-label">Missing required data</div>
              <ul className="warnings">
                {evidenceEvent.missingRequiredData.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </>
          ) : null}
          {evidenceEvent.recommendedNextTests.length ? (
            <>
              <div className="diagnosis-label">Recommended next tests</div>
              <ul className="warnings">
                {evidenceEvent.recommendedNextTests.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </>
          ) : null}
        </section>
      ) : null}

      {finalResult ? (
        <section className="output-block output-block--final">
          <h3 className="output-title">Final diagnosis</h3>
          <p className="body-text">
            Status: <strong>{finalResult.result.status}</strong> · Taxonomy{" "}
            {finalResult.result.taxonomyVersion}
          </p>
          {finalResult.result.status === "diagnosable" ? (
            <div className="diagnosis-row">
              <div>
                <div className="diagnosis-label">Pulpal</div>
                <div className="diagnosis-value">
                  {finalResult.result.pulpalDiagnosis}
                </div>
              </div>
              <div>
                <div className="diagnosis-label">Apical</div>
                <div className="diagnosis-value">
                  {finalResult.result.apicalDiagnosis}
                </div>
              </div>
            </div>
          ) : (
            <p className="body-text">
              No AAE enum assigned. Complete missing data, resolve conflicts, or
              refer.
            </p>
          )}
          {finalResult.result.finalDiagnosisLine ? (
            <p className="body-text">{finalResult.result.finalDiagnosisLine}</p>
          ) : null}
          {finalResult.result.biologicalJustification ? (
            <p className="body-text body-text--tight">
              {finalResult.result.biologicalJustification}
            </p>
          ) : null}
          {finalResult.result.warnings?.length ? (
            <ul className="warnings">
              {finalResult.result.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          ) : null}
          {features?.correctionRagEnabled ? (
            <div className="toolbar toolbar--tight">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => {
                  setCorrStatus(finalResult.result.status);
                  setCorrPulpal(finalResult.result.pulpalDiagnosis ?? "");
                  setCorrApical(finalResult.result.apicalDiagnosis ?? "");
                  setShowCorrection(true);
                }}
              >
                Submit correction
              </button>
            </div>
          ) : null}
        </section>
      ) : null}

      {showCorrection ? (
        <div
          className="modal-backdrop"
          role="dialog"
          aria-modal
          onClick={() => setShowCorrection(false)}
        >
          <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
            <h2 className="modal-title">Save correction</h2>
            <p className="modal-lede">
              No patient identifiers. Explain the error so similar cases retrieve
              this lesson in RAG.
            </p>
            <label className="modal-field">
              <span>Specialist identity</span>
              <input
                value={specialistIdentity}
                onChange={(e) => setSpecialistIdentity(e.target.value)}
              />
            </label>
            <label className="modal-field">
              <span>Correct status</span>
              <select
                value={corrStatus}
                onChange={(e) =>
                  setCorrStatus(e.target.value as DiagnosticStatus)
                }
              >
                <option value="diagnosable">diagnosable</option>
                <option value="insufficient_data">insufficient_data</option>
                <option value="conflicting_data">conflicting_data</option>
                <option value="out_of_scope">out_of_scope</option>
              </select>
            </label>
            <label className="modal-field">
              <span>Correct pulpal</span>
              <select
                value={corrPulpal}
                onChange={(e) =>
                  setCorrPulpal(
                    e.target.value as (typeof PULPAL_DIAGNOSES)[number] | "",
                  )
                }
              >
                <option value="">null</option>
                {PULPAL_DIAGNOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>Correct apical</span>
              <select
                value={corrApical}
                onChange={(e) =>
                  setCorrApical(
                    e.target.value as (typeof APICAL_DIAGNOSES)[number] | "",
                  )
                }
              >
                <option value="">null</option>
                {APICAL_DIAGNOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>Error type</span>
              <select
                value={corrErrorTypes[0]}
                onChange={(e) => setCorrErrorTypes([e.target.value])}
              >
                {ERROR_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>Reasoning (required)</span>
              <textarea
                rows={5}
                value={corrReason}
                onChange={(e) => setCorrReason(e.target.value)}
              />
            </label>
            <label className="modal-field">
              <span>What did the agent misunderstand? (optional)</span>
              <textarea
                rows={3}
                value={corrMisunderstood}
                onChange={(e) => setCorrMisunderstood(e.target.value)}
              />
            </label>
            <div className="modal-actions">
              <button
                type="button"
                className="btn"
                onClick={submitCorrection}
                disabled={
                  corrSaving ||
                  corrReason.trim().length < 10 ||
                  specialistIdentity.trim().length < 2
                }
              >
                {corrSaving ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setShowCorrection(false)}
              >
                Cancel
              </button>
            </div>
            {corrMessage ? (
              <p className="modal-message">{corrMessage}</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
