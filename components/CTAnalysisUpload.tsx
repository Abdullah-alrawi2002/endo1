"use client";

import { useRef, useState } from "react";

type Props = {
  analysisId: string | undefined;
  onChange: (next: string | undefined) => void;
};

/**
 * Experimental CT upload. Enabled only when ENABLE_CT_MODULE=true.
 * Returns an opaque server-side analysisId — never trusts client CT metrics.
 */
export function CTAnalysisUpload({ analysisId, onChange }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [targetTooth, setTargetTooth] = useState("30");
  const [seedConfirmed, setSeedConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function analyze() {
    if (!files.length || !seedConfirmed) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.set("targetTooth", targetTooth);
      form.set("clinicianSeedProvided", "true");
      for (const file of files) form.append("files", file, file.name);
      const response = await fetch("/api/analyze-ct", {
        method: "POST",
        body: form,
      });
      const data = (await response.json()) as {
        analysisId?: string;
        ctSupportSummary?: {
          status: string;
          qualityGatePassed: boolean;
          candidateLowAttenuationRegion: boolean | null;
        };
        error?: string;
      };
      if (!response.ok || !data.analysisId) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }
      onChange(data.analysisId);
      setSummary(
        data.ctSupportSummary
          ? `Stored analysis ${data.analysisId} · status=${data.ctSupportSummary.status} · qualityGate=${data.ctSupportSummary.qualityGatePassed} · candidateLowAttenuation=${data.ctSupportSummary.candidateLowAttenuationRegion}`
          : data.analysisId,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "CT analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="input-category">
      <h3>CT support (experimental / non-diagnostic)</h3>
      <p className="scan-disclaimer">
        Upload a DICOM series only when clinically indicated. The sidecar stores
        an opaque analysis id server-side. Results are candidate features for
        review — they do not diagnose PARL, vitality, or canal length.
      </p>
      <div className="input-grid">
        <label className="input-field">
          <span>Target tooth (Universal)</span>
          <select
            value={targetTooth}
            onChange={(e) => setTargetTooth(e.target.value)}
          >
            {Array.from({ length: 32 }, (_, i) => String(i + 1)).map((n) => (
              <option key={n} value={n}>
                #{n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label className="input-field">
        <span>
          <input
            type="checkbox"
            checked={seedConfirmed}
            onChange={(e) => setSeedConfirmed(e.target.checked)}
          />{" "}
          I confirm the target tooth identity (clinician seed). Automated
          tooth numbering is disabled.
        </span>
      </label>
      <input
        ref={fileInputRef}
        type="file"
        accept=".dcm,.dicom,.zip,application/dicom,application/zip"
        multiple
        onChange={(e) => {
          setFiles(e.target.files ? Array.from(e.target.files) : []);
          onChange(undefined);
          setSummary(null);
        }}
      />
      <div className="toolbar toolbar--tight">
        <button
          type="button"
          className="btn"
          onClick={analyze}
          disabled={!files.length || !seedConfirmed || busy}
        >
          {busy ? "Processing…" : "Analyze CT (research)"}
        </button>
        {analysisId ? (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => {
              onChange(undefined);
              setSummary(null);
              setFiles([]);
              if (fileInputRef.current) fileInputRef.current.value = "";
            }}
          >
            Clear
          </button>
        ) : null}
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      {summary ? <p className="body-text body-text--tight">{summary}</p> : null}
    </div>
  );
}
