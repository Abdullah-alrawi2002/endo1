# Diagnocat -> agentic ECR system (rev 2.7)

## What Diagnocat can and cannot contribute

Diagnocat is a cloud dental-AI service that segments CBCT volumes (teeth, pulp, jaws, canal), numbers the teeth and
reports per-tooth conditions with probabilities (0-0.99), exports STL meshes and produces a radiological report.
Its AI 4.0 condition list includes internal resorption, root caries, caries signs, abfraction, periapical
radiolucency, PDL widening, periodontal bone loss, furcation, endodontic treatment, fillings, crowns, overhangs,
open margins, impaction, orthodontic appliances and root fracture ([User manual Diagnocat AI 4.0](https://diagnocat.com/files/eu/User%20manual%20Diagnocat%20AI%204.0.pdf)).
**It has no external / cervical resorption class.** In an independent CBCT validation the 14 external-resorption
cases had no corresponding AI output ([Journal of Clinical Medicine 2026, PMC13302509](https://pmc.ncbi.nlm.nih.gov/articles/PMC13302509/)).
The FDA clearance covers periapical radiolucency detection ([Diagnocat 1.0 FDA user manual](https://diagnocat.com/files/Diagnocat%201.0%20User%20Manual_FDA.pdf)).

So the integration is:

```
DICOM  --->  Diagnocat (segmentation, FDI numbering, per-tooth conditions, STL, clinician-added flags/comments)
   |                      |
   |                      v
   |            DiagnocatStudy  (ecr_cds.diagnocat.schema; from Partner API JSON | UI JSON export | report PDF | STL)
   |                      |
   |                      v
   |            ToothContext per tooth (ecr_cds.diagnocat.context + PatientChart)
   |              prior shifts, clinical-modifier hints, canal hint, metal expected, triage priority
   v                      v
Perception layer (ecr_cds.perception) ----> agent layer (ecr_cds.agents) ----> diagnosis, measured extent, plan, CDT codes
   segment / propose / delineate / measure    context, QA, anatomy, screening, verifier, radiologist, skeptic,
   reformat / re-verify, provenance           clinical correlation, classifier, planner, coverage, knowledge, moderator
```

Diagnocat supplies **anatomy, numbering, context and triage**; resorption reasoning comes from this package's
screen and conference. Context shifts priors and fills unset clinical inputs; it never creates an imaging finding.
If our screen finds no candidate on a tooth, the tooth is reported as "no candidate" whatever Diagnocat says.

## Getting Diagnocat output into the system

| route | function | notes |
|---|---|---|
| Partner API | `DiagnocatClient(base_url, api_key, endpoints).process(dicom_zip)` | The API is partner-gated: the [Postman collection](https://www.postman.com/diagnocat-doc/diagnocat-partner-api/overview) is obtained from api-support@diagnocat.com. Endpoint paths and field names are therefore configurable (`DIAGNOCAT_BASE_URL`, `DIAGNOCAT_API_KEY`, `DIAGNOCAT_EP_UPLOAD/STATUS/REPORT`, `field_map`). Nothing here is verified against a live account. |
| JSON export | `load_diagnocat_json(path, field_map=None)` | Accepts our interchange schema or Partner-API-like payloads (tooth/number/ISONumber, conditions/findings, probability/confidence in 0-1 or percent, source ai/clinician). |
| Report PDF | `load_diagnocat_pdf(path)` | Best-effort text parse of the CBCT radiological report ("Tooth NN" blocks, condition names from the 4.0 manual, percentages when "Conditions details" is on). Verify against the UI. |
| STL | `stl_to_mask(path, shape, spacing, origin_xyz, axis_order, flip)` | Pure-numpy voxeliser (binary or ASCII STL). STL and DICOM coordinate handedness must be checked visually; the mask becomes the tooth mask via `MaskToothSegmenter`. |

Condition-name normalisation lives in `CONDITION_ALIASES`; unknown names are kept as `unknown:<name>` and never
affect priors. An "external resorption" label counts **only** when `source == "clinician"` (manual tag in Diagnocat).

## Context -> priors and inputs (`ecr_cds/diagnocat/context.py`)

| Diagnocat finding (p) | effect |
|---|---|
| internal resorption | log-odds +2.0p on internal_resorption; canal hint |
| root caries / caries signs | +1.2p caries |
| abfraction | +1.2p non-carious cervical lesion |
| crown / filling / overhang / open margin / post | +1.0p restoration-margin, +0.5p artefact; metal expected; `restoration_at_defect` hint |
| endodontically treated (p>=0.5) | `pulp_vital=False`; canal hint `hyperdense_filling` |
| periapical radiolucency / apical PDL widening | `pulp_vital=False` hint (confirm with sensibility tests) |
| severe bone loss / furcation | `perio_support_adequate=False` |
| impaction / orthodontic appliance | predisposing factor, +0.4p ECR |
| clinician-added external resorption | +1.0 ECR, top triage |
| molar (FDI x6-x8) | `multirooted=True`; tooth type for CDT code digit |

Patient chart (`ecr_cds/agents/chart.py`): chief-complaint teeth / quadrants (FDI; Universal via
`universal_to_iso`) raise triage priority; history events (orthodontics, trauma, bleaching, RCT, crown, perio,
bruxism) add predisposing factors, prior shifts and modifier hints; symptoms in the complaint focus set
`symptomatic` and pink-spot / bleeding hints for Q8. Priors are renormalised with shifts clipped to +-2.5.
Hints only fill inputs the clinician left unset.

## Patient-level run

```python
from ecr_cds.dicom_loader import load_dicom_series
from ecr_cds.diagnocat import load_diagnocat_json
from ecr_cds.diagnocat.pipeline import run_diagnocat_pipeline
from ecr_cds.agents.chart import PatientChart, HistoryEvent

vol = load_dicom_series("scan.zip", expected_slices=501)
study = load_diagnocat_json("diagnocat_export.json")
chart = PatientChart(age_years=34, complaint_quadrants=[2], symptoms=["pink spot"],
                     history=[HistoryEvent("orthodontics", None, 10)])
report = run_diagnocat_pipeline(vol, study, chart=chart, max_teeth=3, out_dir="diagnocat_out")
```

CLI: `python -m ecr_cds.cli diagnocat-run --dicom scan.zip --diagnocat-json export.json --chart chart.json --out out`.

For each prioritised tooth the pipeline seeds the segmentation from the Diagnocat STL (if given), a Diagnocat seed
voxel, or the next unused automatically segmented tooth ranked by radiolucent defect volume (our segmentation has no
FDI numbering, so this fallback is by rank and is labelled as such). The per-tooth conference runs with the
tooth's context; the patient report lists triage order, per-tooth outcome, CDT codes, the Diagnocat-scope
citations and every perception tool call.

## Knowledge agent and CDT codes

`ecr_cds/knowledge/guidelines.json` holds short cited paraphrases from the
[ESE 2023 position statement](https://onlinelibrary.wiley.com/doi/10.1111/iej.13916),
[Patel 2018](https://pubmed.ncbi.nlm.nih.gov/28746776/), the DMFR mimics review, the
[AAE CDT 2024 guide](https://www.aae.org/specialty/wp-content/uploads/sites/2/2023/12/CDTGuide_2024.pdf) and the
Diagnocat documentation. The KnowledgeAgent attaches them to the answered diagnostic questions (Q1-Q7 tags), the
plan (ESE aims, factors, prognosis; Patel management), the coverage gaps (clinical / staging / imaging) and the
Diagnocat scope. CDT codes are mapped per plan option and tooth type (D3471/2/3 surgical repair of root resorption,
D3310/20/30 endodontic therapy, D3333 internal perforation repair, D3470 intentional reimplantation, D3450/D3920
root amputation / hemisection, D7140/D7210 extraction, D7251 coronectomy as the closest descriptor for
decoronation, D0364 limited-FOV CBCT, D9310 consultation). Codes are for documentation; confirm with the payer.

## Honesty constraints (unchanged)

Every output is a suggestion with `requires_confirmation=True`; posteriors are uncalibrated; the confidence band is
capped while CEJ / apex / canal are automatic; a negative screen does not exclude ECR; Diagnocat does not detect ECR;
the Partner API schema is unverified until the collection is obtained.
