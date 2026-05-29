"use client";

import { ClinicalCaseForm } from "@/components/ClinicalCaseForm";
import { useMemo, useState } from "react";
import {
  APICAL_DIAGNOSES,
  type ClinicalCase,
  defaultClinicalCase,
  type FinalDiagnosis,
  PULPAL_DIAGNOSES,
} from "@/lib/schemas/clinical-case";

type StreamEvent =
  | { type: "rag"; matchCount: number; contextBlock: string }
  | { type: "stage"; stage: 1 | 2 | 3 | 4; title: string; content: string }
  | { type: "final"; result: FinalDiagnosis }
  | { type: "error"; message: string };

const initialCase: ClinicalCase = defaultClinicalCase;

export default function Home() {
  const [clinicalCase, setClinicalCase] = useState<ClinicalCase>(initialCase);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [corrPulpal, setCorrPulpal] = useState<(typeof PULPAL_DIAGNOSES)[number]>(
    PULPAL_DIAGNOSES[0],
  );
  const [corrApical, setCorrApical] = useState<(typeof APICAL_DIAGNOSES)[number]>(
    APICAL_DIAGNOSES[0],
  );
  const [corrReason, setCorrReason] = useState("");
  const [corrMisunderstood, setCorrMisunderstood] = useState("");
  const [corrSaving, setCorrSaving] = useState(false);
  const [corrMessage, setCorrMessage] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);

  const finalResult = useMemo(
    () => [...events].reverse().find((e) => e.type === "final") as
      | Extract<StreamEvent, { type: "final" }>
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
        body: JSON.stringify(clinicalCase),
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
    if (!finalResult) return;
    setCorrSaving(true);
    setCorrMessage(null);
    try {
      const res = await fetch("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case: clinicalCase,
          agentPulpal: finalResult.result.pulpalDiagnosis,
          agentApical: finalResult.result.apicalDiagnosis,
          correctedPulpal: corrPulpal,
          correctedApical: corrApical,
          correctionReasoning: corrReason,
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
          Enter findings, run the agent, optionally save corrections for RAG.
          Educational use only.
        </p>
      </header>

      <p className="section-label">Case</p>
      <ClinicalCaseForm value={clinicalCase} onChange={setClinicalCase} />

      <div className="toolbar">
        <button
          type="button"
          className="btn"
          onClick={runDiagnosis}
          disabled={loading}
        >
          {loading ? "Running…" : "Run"}
        </button>
        <button
          type="button"
          className="btn btn--ghost"
          onClick={exportCorrections}
          disabled={exportBusy}
        >
          {exportBusy ? "Exporting…" : "Export corrections"}
        </button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {events.length > 0 ? (
        <section className="output-stack">
          {events.map((ev, i) => {
            if (ev.type === "rag") {
              return (
                <article key={`${i}-rag`} className="output-block">
                  <h3 className="output-title">
                    Similar corrections ({ev.matchCount})
                  </h3>
                  {ev.matchCount === 0 ? (
                    <p className="body-text--tight body-text--flush">
                      None retrieved.
                    </p>
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
            if (ev.type === "final") {
              return (
                <article
                  key={`${i}-final`}
                  className="output-block output-block--final"
                >
                  <h3 className="output-title">Diagnosis</h3>
                  <div className="diagnosis-row">
                    <div>
                      <div className="diagnosis-label">Pulpal</div>
                      <div className="diagnosis-value">
                        {ev.result.pulpalDiagnosis}
                      </div>
                    </div>
                    <div>
                      <div className="diagnosis-label">Apical</div>
                      <div className="diagnosis-value">
                        {ev.result.apicalDiagnosis}
                      </div>
                    </div>
                  </div>
                  <p className="body-text">{ev.result.finalDiagnosisLine}</p>
                  <p className="body-text body-text--tight">
                    {ev.result.biologicalJustification}
                  </p>
                  {ev.result.warnings?.length ? (
                    <ul className="warnings">
                      {ev.result.warnings.map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  ) : null}
                  <div className="toolbar toolbar--tight">
                    <button
                      type="button"
                      className="btn btn--ghost"
                      onClick={() => {
                        setCorrPulpal(ev.result.pulpalDiagnosis);
                        setCorrApical(ev.result.apicalDiagnosis);
                        setShowCorrection(true);
                      }}
                    >
                      Correct diagnosis
                    </button>
                  </div>
                </article>
              );
            }
            return null;
          })}
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
              No patient identifiers. Explain the error so similar cases pick
              this up in RAG.
            </p>
            <label className="modal-field">
              <span>Pulpal</span>
              <select
                value={corrPulpal}
                onChange={(e) =>
                  setCorrPulpal(e.target.value as (typeof PULPAL_DIAGNOSES)[number])
                }
              >
                {PULPAL_DIAGNOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>Apical</span>
              <select
                value={corrApical}
                onChange={(e) =>
                  setCorrApical(e.target.value as (typeof APICAL_DIAGNOSES)[number])
                }
              >
                {APICAL_DIAGNOSES.map((p) => (
                  <option key={p} value={p}>
                    {p}
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
              <span>Misunderstood (optional)</span>
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
                disabled={corrSaving || corrReason.trim().length < 10}
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
