"""Execute a packaged notebook in a fresh directory; collect explicit outputs."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient

def _stage_label(conference):
    se = conference.get("suggested_extent")
    if se:
        return f"{se['extent_descriptor']} / {se['pulp_relationship']} / {se['access_descriptor']}"
    sp = conference.get("suggested_patel")   # legacy scheme only
    return f"legacy Patel {sp['code']}" if sp else "no extent summary"


p = argparse.ArgumentParser()
p.add_argument("--scan", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
root = Path(__file__).resolve().parent
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
scan = str(Path(args.scan).expanduser().resolve())
nb = nbformat.read(root / "ECR_CDS_Colab.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.source = c.source.replace("SCAN_PATH = None ", f"SCAN_PATH = {scan!r} ")
with tempfile.TemporaryDirectory(prefix="ecr_v23_") as run:
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": run}}).execute()
    generated = Path(run) / "ecr_cds" / "outputs_case"
    for f in generated.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
        elif f.is_dir():
            shutil.copytree(f, out / f.name, dirs_exist_ok=True)
    nbformat.write(nb, out / "ECR_CDS_executed.ipynb")
    image_count = 0
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                print(o.text, end="")
            if "image/png" in o.get("data", {}):
                image_count += 1
                (out / f"preview_{image_count}.png").write_bytes(
                    base64.b64decode(o["data"]["image/png"]))
    report = json.loads((out / "auto_report.json").read_text())
    conf_path = out / "conference" / "conference_report.json"
    conference = json.loads(conf_path.read_text()) if conf_path.exists() else None
    if conference is not None:
        assert conference["requires_confirmation"] is True and conference["clinical_validation"] is False
        assert conference["status"] in ("suggested_ecr_with_plan", "ecr_possible_conditional_plan",
                                        "no_ecr_suggested", "no_candidates")
    assert report["ecr_diagnosis"] is None
    assert report["patel"] is None and report["decision"] is None
    selected = [c for c in report["candidates"] if c["disposition"] != "rejected_by_screen"]
    rows = ["# Cervical resorption candidate review",
            "",
            "The revised notebook was executed on the uploaded 501-slice scan. "
            "This is an experimental software screen, not a diagnosis. "
            "The earlier rule-derived 2Bp interpretation is superseded.",
            "",
            "## Result",
            "",
            f"- **Status:** `{report['status']}`.",
            f"- **Review regions:** {len(selected)} on the automatically selected tooth.",
            "- **ECR diagnosis, Patel class and treatment:** withheld.",
            "- **Ground truth:** no expert ECR annotation or canal mask was supplied.",
            "",
            "| Component | Candidate volume | Centroid (z, y, x) | Disposition |",
            "|---|---:|---|---|"]
    for c in selected:
        pos = ", ".join(str(round(v, 1)) for v in c["centroid_volume_zyx"])
        rows.append(f"| {c['component_id']} | {c['volume_mm3']:.2f} mm³ | ({pos}) | {c['disposition']} |")
    rows += ["",
             "Volumes quantify threshold-derived review masks, not confirmed resorption.",
             "",
             "## Reasons for review",
             ""]
    for flag in sorted({f for c in selected for f in c["flags"]}):
        rows.append(f"- **{flag}:** retained in the JSON audit; it prevents an unqualified ECR finding.")
    if conference is not None:
        dx = conference.get("suggested_diagnosis") or {}
        rows += ["", "## Multi-agent case conference (rev 2.5)", "",
                 f"- **Status:** `{conference['status']}` (backend `{conference.get('backend')}`: {conference.get('backend_status')}, {conference.get('rounds')} debate rounds).",
                 f"- **Suggested diagnosis:** {dx.get('statement', 'none')}",
                 f"- **Confidence band:** {dx.get('band', 'n/a')} (uncalibrated; capped while anatomy is unreviewed).",
                 f"- **Suggested extent / plan:** " + (f"{_stage_label(conference)} -> {conference['suggested_plan']['rule_recommendation']['primary']} (conditional on {conference['suggested_plan']['conditional_on']})"
                                                     if conference.get("suggested_plan") else "withheld (ECR not suggested)."), ""]
        if dx.get("diagnostic_questions"):
            rows += ["| # | question | answer |", "|---|---|---|"] + [f"| {q['id']} | {q['question']} | {q['answer']} |" for q in dx["diagnostic_questions"]]
            rows += [f"| Q8 | Does the clinical picture fit? | {dx.get('clinical_correlation', 'not_assessed')} |", ""]
        if dx.get("differential"):
            rows += ["| hypothesis | uncalibrated posterior |", "|---|---:|"] + [f"| {h} | {v:.2f} |" for h, v in dx["differential"]] + [""]
        if dx.get("suggested_next_steps"):
            rows += ["Suggested work-up for the leading alternative:"] + [f"- {x}" for x in dx["suggested_next_steps"]] + [""]
        majors = [c["issue"] for c in dx.get("unresolved_challenges", []) if c["severity"] == "major"]
        if majors:
            rows += ["Major skeptic challenges:"] + [f"- {m}" for m in majors] + [""]
        rows += ["Full evidence trail: `conference/conference_report.json`; summary: `conference/conference_summary.md`; tiles: `conference/candidate_*_tiles.png`.", ""]
    rows += ["",
             "## Files",
             "",
             "- **auto_report.json:** settings, thresholds, component features and rejected regions.",
             "- **ecr_review_candidates.nii.gz:** component-ID review mask on the original grid.",
             "- **automatic_canal_proxy.nii.gz:** automatic filling/canal proxy, not an expert mask.",
             "- **auto_labels.nii.gz:** classical teeth/bone segmentation, not DentalSegmentator output.",
             "- **ECR_CDS_executed.ipynb:** recorded execution with previews.",
             "",
             "## Next clinical step",
             "",
             "An endodontist or oral radiologist should verify the tooth, CEJ, canal and "
             "candidate regions against the full scan and clinical examination. "
             "This screen cannot distinguish ECR from caries, restoration margins or artifacts, "
             "and it cannot exclude ECR when negative.",
             ""]
    (out / "ECR_candidate_review.md").write_text("\n".join(rows))
    print("\nVerified: no unconfirmed diagnosis, Patel code or treatment.")
