"""Rev 2.7 demo on the supplied 501-slice scan with a clearly-labelled MOCK Diagnocat context.

No real Diagnocat output exists for this scan (Partner API is partner-gated). The mock study only carries the one
condition our own anatomy agent already established on earlier runs (root-filled tooth), so it adds no information
that the imaging layer did not produce. Everything is rules-only (no LLM backend in this environment).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecr_cds.dicom_loader import load_dicom_series
from ecr_cds.diagnocat import DiagnocatStudy, DiagnocatTooth, DiagnocatCondition
from ecr_cds.diagnocat.pipeline import run_diagnocat_pipeline

SCAN = os.path.expanduser("~/.perplexity/attachments/9a3a9d7ecd8f4219a93dec92cebb5463/Archive-2.zip")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/workspace/ecr_diagnocat_v27"
VERIFY_MODE = sys.argv[2] if len(sys.argv) > 2 else "smooth"   # rev 2.8: "tta" for the multi-path critic re-inspection

t0 = time.time()
vol = load_dicom_series(SCAN, expected_slices=501)
print("loaded", vol.shape, vol.spacing, round(time.time() - t0, 1), "s", flush=True)
mock = DiagnocatStudy(
    study_id="MOCK_no_real_diagnocat_output",
    teeth=[DiagnocatTooth(iso_number=0, conditions=[DiagnocatCondition("endo_treated", "Endodontically treated tooth", 0.9, "ai")],
                          comment="MOCK context: tooth number not established; condition mirrors our own root-filling detection")],
    source="mock", notes=["MOCK study - no Diagnocat processing was performed on this scan"])
rep = run_diagnocat_pipeline(vol, mock, chart=None, max_teeth=1, backend=None, max_rounds=2, out_dir=OUT, render=True, verify=True, verify_mode=VERIFY_MODE)
print(json.dumps({"seconds": rep["seconds"], "teeth_examined": rep["teeth_examined"],
                  "per_tooth": [{k: v for k, v in t.items() if k in ("iso_number", "seed_zyx", "seed_source", "status")} for t in rep["per_tooth"]]}, indent=1))
for t in rep["per_tooth"]:
    dx = t.get("suggested_diagnosis") or {}
    print("leading:", dx.get("leading"), "P(ECR)=", dx.get("p_ecr_uncalibrated"), "| verification:",
          {k: (v["verdict"], v["reproducibility"]) for k, v in (t.get("verification") or {}).items()})
    print("coverage gaps:", [m["item"] for m in (t.get("coverage") or {}).get("missing_prerequisites", [])])
print("written to", OUT)
