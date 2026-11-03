# VLM feature measurement and measurement-based extent summary (rev 2.9)

Rev 2.9 adds a **feature-measurement layer** between the perception toolkit and the multi-agent conference.
For every candidate the layer produces a typed `FeatureBundle` of

* **visual features** - categorical descriptions a local vision-language model (VLM) gives of the
  provenance-stamped evidence frame, and
* **physical measurements** - millimetre / fraction / boolean quantities computed from the voxel
  geometry (candidate mask, tooth / canal / bone masks, voxel spacing), each with confidence and provenance.

The conference agents consume the bundle, a **descriptive extent summary** replaces the staging classifier on
the default path, and the treatment planner keys its safe set off that summary with a documented,
measurement-indexed mapping. **No Patel (height / circumferential / canal-proximity) code is produced,
consumed or staged by anything on this path.** The rev <= 2.8 classifier survives only as an explicit legacy
opt-in (`staging_scheme="patel_3d"`) so frozen outputs can be reproduced.

Everything here is research software: outputs are uncalibrated suggestions for clinician review, a negative
reading never excludes ECR, and no irreversible action is ever selected by the software.

---

## 1. Components

| module | role |
|---|---|
| `ecr_cds/perception/feature_schema.py` | typed schema: `VISUAL_ENUMS`, `ESTIMATE_KINDS`, `MEASUREMENT_UNITS`, `VLM_FEATURE_RESPONSE_SCHEMA`, `validate_vlm_response`, `VisualFeatures`, `Measurement`, `FeatureBundle`, `validate_bundle_dict` |
| `ecr_cds/perception/vlm_features.py` | `measure_candidate_geometry` (voxel measurements), `query_vlm_features` (one structured VLM request), `reconcile` (VLM vs geometry), `measure_candidate_features` (bundle), `FEATURE_SYSTEM_PROMPT` |
| `ecr_cds/extent_summary.py` | `summarise_extent(bundle, descriptors) -> ExtentSummary`: continuous measurements plus three descriptors (`extent_descriptor`, `pulp_relationship`, `access_descriptor`) and a narrative |
| `ecr_cds/agents/feature_measurement.py` | `FeatureMeasurementAgent` (posts `feature_bundle` evidence), `ExtentSummaryAgent` (posts `extent_summary` evidence), `bundle_feature_notes` |
| `ecr_cds/decision_matrix.py` | `measurement_base_cell`, `recommend_from_measurements`, `measurement_matrix_rows`; `Recommendation.cell_key` / `cell_scheme` |
| `ecr_cds/rl_policy.py` | `ClinicalContext(extent=summary, ...)`; `Decision.cell_scheme` |
| `ecr_cds/agents/roles.py` | consumers: lesion-screening reconciliation, radiologist (capped feature log-odds), skeptic (unverified / disagreeing model values), clinical-correlation cross-check, treatment planner (measurement cell) |
| `ecr_cds/agents/orchestrator.py` | `CaseConference(staging_scheme="measurement", vlm_features=True, feature_backend=None)`; `suggested_extent` and `feature_bundles` in the consensus; markdown sections |
| `docs/measurement_matrix.csv` | the full 60-cell measurement-indexed safe-set mapping (generated from `measurement_matrix_rows()`) |

## 2. Flow for one candidate

```
candidate mask + tooth/canal/bone masks + spacing + screen geometry (axis, CEJ)
        |
        v
measure_candidate_geometry  ---->  13 Measurement objects (source = voxel_geometry, verified = True,
        |                            or source = unavailable when a landmark / mask is missing)
        v
query_vlm_features (optional)  -->  evidence frame + scale only -> JSON {visual_features, estimates, ...}
        |                            validated / coerced by validate_vlm_response
        v
reconcile  -->  per estimate: agrees | disagrees (geometry kept, flagged) | not_comparable (kept as
        |        vlm_unverified, low confidence, listed in unverified_vlm_values)
        v
FeatureBundle  -->  Evidence(kind="feature_bundle")  -->  consumers
        |
        v  (leading candidate only, when ECR is suggested / possible)
summarise_extent  -->  ExtentSummary  -->  Evidence(kind="extent_summary")  -->  recommend_from_measurements
```

Design rules:

1. **Geometry first.** Every measurement is computed from voxels before the model is asked anything.
2. **The model never sees the numbers.** The user payload contains only the candidate id, voxel size, crop
   width and panel list. Its estimates are therefore an independent reading that can be cross-checked rather
   than an echo.
3. **Geometry always wins.** A model value never overrides a geometric one. Disagreement is recorded on the
   measurement (`vlm_agreement="disagrees"`, flag `vlm_value_disagrees_with_geometry_geometry_kept`) and on the
   bundle (`vlm_geometry_disagreements`), and the skeptic raises it.
4. **Unverifiable values are flagged and excluded.** When geometry could not compute a quantity (no canal mask,
   no CEJ, no crest) the model's value is kept for display only: `source="vlm_unverified"`, `verified=False`,
   confidence <= 0.3, listed in `unverified_vlm_values`. `FeatureBundle.measurement_value()` hides such values
   by default, so the extent summary and the planner never key off them; the planner instead adds
   `canal_delineation` / `bone_crest_level` to `missing_inputs`.
5. **Abstention.** With `NemotronLocalBackend` several samples are drawn and clustered on the categorical
   fields (`feature_cluster_key`); high discrete semantic entropy -> `status="abstained"`, visual features are
   not used, geometry is unaffected.
6. **Nothing stages.** The prompt forbids staging, grading, naming a classification system, diagnosing and
   recommending treatment; unknown estimate keys (for example a smuggled `patel_height`) are dropped by the
   validator and only mentioned in the validation log.

## 3. Schema

### 3.1 Visual features (`VISUAL_ENUMS`)

| field | values |
|---|---|
| `radiolucency_pattern` | homogeneous, heterogeneous, mottled, not_visible, uncertain |
| `border_definition` | well_defined, irregular, smooth, ill_defined, uncertain |
| `fibro_osseous_appearance` | yes, no, uncertain (radiopaque islands inside the defect) |
| `pericanal_radiopaque_line` | intact, partially_interrupted, interrupted, not_assessable |
| `portal_of_entry_visible` | yes, no, uncertain |
| `portal_surface` | buccal, lingual_palatal, mesial, distal, uncertain |
| `relation_to_restoration_margin` | no_restoration_visible, separate_from_margin, adjacent_to_margin, continuous_with_margin, uncertain |
| `burnout_artifact_suspected` | yes, no, uncertain |
| `metal_beam_hardening` | none, mild, marked, uncertain |

plus `lesion_visible_in_image` (yes / no / uncertain / no_image), `panels_used`, `description` (<= 60 words).
Out-of-enumeration answers fall back to the last ("uncertain"-type) value and are recorded in
`vlm.validation_errors`.

### 3.2 Measurements (`MEASUREMENT_UNITS`)

All are measured along / around the tooth long axis of the screen geometry (apical positive), on the
candidate mask of the cervical screen.

| name | unit | definition | needs |
|---|---|---|---|
| `long_axis_extent_mm` | mm | extent of the candidate along the tooth axis (+ one voxel) | mask |
| `coronal_limit_from_cej_mm` | mm | most coronal voxel relative to the CEJ plane (negative = coronal to CEJ) | CEJ |
| `apical_limit_from_cej_mm` | mm | most apical voxel relative to the CEJ plane | CEJ |
| `coronal_limit_from_crest_mm` | mm | portal level relative to the bone crest (positive = below crest) | bone mask |
| `apical_limit_from_crest_mm` | mm | apical limit relative to the bone crest | bone mask |
| `mesio_distal_extent_mm` | mm | in-plane extent along the MD axis | mask (+ buccal direction for a patient frame) |
| `bucco_lingual_extent_mm` | mm | in-plane extent along the BL axis | mask (+ buccal direction) |
| `max_depth_from_surface_mm` | mm | deepest candidate voxel below the convex tooth surface | tooth mask |
| `circumferential_extent_deg` | deg | occupied 6-degree bins around the local tooth-axis centroid (1-bin gaps closed) | mask |
| `dentine_volume_fraction_involved` | fraction | candidate ∩ (tooth − canal) / (tooth − canal) | tooth + canal masks |
| `min_distance_to_canal_mm` | mm | minimum distance from candidate voxels to the canal mask (0 when touching) | canal mask |
| `root_canal_communication` | bool | overlap with the canal mask or within one voxel of it | canal mask |
| `lesion_volume_mm3` | mm3 | candidate voxels × voxel volume | mask |

Bone crest: 2nd percentile along the axis of bone voxels within a 1 mm shell of the tooth surface (flag
`bone_crest_is_automatic_estimate`). Without a buccal direction vector (`--buccal z,y,x`) the BL axis is the
radial direction through the candidate centroid (a proxy valid for buccal / lingual defects) and the in-plane
extents are capped at confidence 0.5 with flag `bl_md_axes_not_patient_oriented:<definition>`.

Each `Measurement` carries: `value`, `unit`, `source` (`voxel_geometry` | `vlm_unverified` | `unavailable`),
`verified`, `confidence` (0-1, uncalibrated: mask stability, metal proximity, landmark source; never model
certainty), `provenance` (slice / row / column ranges in the volume, centroid, axis, spacing, evidence-frame
panels, measurement-specific notes), `vlm_value`, `vlm_agreement`
(`agrees` | `disagrees` | `not_reported` | `not_comparable`), `discrepancy`, `flags`.

Confidence rules: base 0.9 × `stable_core_fraction` (min 0.3); × 0.7 when `near_metal_fraction` > 0.3;
CEJ-referenced values <= 0.5 unless the CEJ was reviewed, <= 0.3 when the CEJ is the crest-offset assumption;
canal-referenced values <= 0.6 when the canal is an automatic proxy, <= 0.4 when it is a root filling; crest
values <= 0.6; dentine fraction <= 0.7.

### 3.3 Model estimates (`ESTIMATE_KINDS`) and tolerances

The model may estimate `long_axis_extent_mm`, `mesio_distal_extent_mm`, `bucco_lingual_extent_mm`,
`coronal_limit_from_cej_mm`, `apical_limit_from_crest_mm`, `dentine_volume_fraction_involved`,
`min_distance_to_canal_mm` and `root_canal_communication` (yes / no / uncertain). Agreement tolerance:
`max(2 × voxel pitch, 20 % of the geometric value, 0.5 mm)` for lengths, 0.15 for fractions, exact for booleans.

### 3.4 Bundle

```json
{
  "schema_version": "vlm_features_v1",
  "candidate_id": 2, "source_id": "case", "spacing_zyx_mm": [0.1, 0.1, 0.1],
  "visual": {"assessed": true, "radiolucency_pattern": "heterogeneous", "...": "...", "lesion_visible_in_image": "yes"},
  "measurements": {"long_axis_extent_mm": {"value": 3.3, "unit": "mm", "source": "voxel_geometry", "verified": true,
                                            "confidence": 0.77, "provenance": {"...": "..."}, "vlm_value": 3.8,
                                            "vlm_agreement": "agrees", "discrepancy": 0.5, "flags": ["..."]}, "...": "..."},
  "vlm": {"status": "ok | abstained | failed | not_configured | no_image", "backend": "...", "dse": {"...": "..."},
          "validation_errors": [], "raw_estimates": {"...": "..."}},
  "unverified_vlm_values": [], "vlm_geometry_disagreements": [], "caveats": ["..."], "disclaimer": "..."
}
```

`validate_bundle_dict()` checks a serialised bundle structurally (schema version, enums, units, sources,
verified ⇒ voxel_geometry, confidence range, value types).

## 4. Extent summary (descriptive, not a staging code)

`summarise_extent` uses **verified measurements only** and produces three descriptors with the
measurements that drove them (`ExtentSummary.to_dict()` includes `is_staging_code: false`).

Extent score (0-8; each rule gives 0 / 1 / 2 points):

| measurement | 1 point at | 2 points at |
|---|---|---|
| `long_axis_extent_mm` | >= 3.0 | >= 6.0 |
| circumferential fraction (deg / 360) | >= 1/3 | >= 2/3 |
| `dentine_volume_fraction_involved` | >= 0.05 | >= 0.15 |
| `lesion_volume_mm3` | >= 20 | >= 60 |

`extent_descriptor`: score <= 1 **limited**, 2-4 **moderate**, >= 5 **extensive**.

`pulp_relationship`: **canal_communication** when `root_canal_communication` is true;
**thin_dentine_barrier** when `min_distance_to_canal_mm` < 0.5 mm; **dentine_barrier_preserved** otherwise;
**not_assessable** when the canal measurements are unavailable or unverified.

`access_descriptor` (portal level = coronal limit relative to the crest): **supracrestal_portal** (<= 0 mm),
**shallow_subcrestal_portal** (<= 3 mm), **deep_subcrestal_portal** (> 3 mm), **no_surface_portal** (candidate has
no external surface connection), **crest_unknown**.

Also: `restorability_concern` (dentine fraction >= 0.30), `repair_phase_suspected` (radiopaque inclusion
fraction >= 0.5), `near_boundary` (score within 1 of a cut, canal distance within one voxel of 0.5 mm, portal
level within one voxel of a cut) which the planner reports as `descriptor_near_band_edge`, and the policy
treats as borderline.

The thresholds are engineering bands chosen so the planner's cells line up with the ESE 2023 / management
literature statements already cited in the knowledge base; they are not a clinical staging system and carry
no validation claim.

## 5. Measurement-indexed safe-set mapping

`recommend_from_measurements(summary, modifiers)` looks up the cell
`extent=<...>|pulp=<...>|access=<...>` in `measurement_base_cell` (two 3 × 3 tables: externally reachable portal
vs deep subcrestal portal; `not_assessable` is handled conservatively as a thin barrier and recorded as a note;
`crest_unknown` as shallow subcrestal; `no_surface_portal` returns periodic review with a note that ECR is
unlikely from the mask and internal resorption / a closed cavity should be reconsidered). The shared clinical
modifier rules R1-R11 (accessibility, pulpal involvement, haemorrhage, multirooted, restorability, growing
patient, ...) then run exactly as in the legacy matrix, parameterised by `deep` (extensive or deep subcrestal),
`pulp_involved` (canal communication) and the end-stage-like condition (extensive + communication or deep
portal). The full 60-row table is in `docs/measurement_matrix.csv`; regenerate with

```python
from ecr_cds.decision_matrix import measurement_matrix_rows
```

Every `Recommendation` carries `cell_key` and `cell_scheme` (`measurement_v1` here, `patel_3d` for the legacy
matrix). `patel_code` appears in `to_dict()` only for the legacy scheme. `ConstrainedBanditPolicy` is keyed by
`cell_key` and `ClinicalContext(extent=summary)` vectorises the descriptors and continuous measurements.
Irreversible options (extraction, decoronation, intentional replantation, hemisection) are offered in the safe
set when the cell warrants it, always with `requires_confirmation=True`, and the conference adds an explicit
clinician action; they are never selected automatically.

## 6. How the agents use the bundle

* `FeatureMeasurementAgent` runs for every candidate after the verifier, posts `feature_bundle` evidence
  (basis `vlm+measurement` or `measurement`) and transcript lines listing unverified / disagreeing values.
* `LesionScreeningAgent.reconcile` posts `screen_reconciliation` (screen descriptor vs measured value
  differences), never changing dispositions.
* `RadiologistAgent` adds a per-hypothesis log-odds nudge from the assessed visual features
  (`FEATURE_LOGODDS`, capped at ±0.6, zero on abstention / geometry-only), includes the bundle in the model
  prompt, and records `feature_logodds`, `feature_notes`, `feature_bundle_used`. If the model's differential
  opinion on the same frame abstains (high semantic entropy), the feature nudge is dropped as well
  (`feature_nudge_dropped`), so the rev 2.8 invariant "abstention -> rule posterior kept" still holds.
* `SkepticAgent` raises challenges for unverified model values (minor), VLM / geometry disagreements
  (moderate), model-flagged confounders (restoration margin, burnout, beam hardening), abstention and
  screen / model portal disagreements. Model challenges are never "major".
* `ClinicalCorrelationAgent` cross-checks visual features against entered findings (`feature_crosscheck`).
* `ExtentSummaryAgent` summarises the leading candidate; `TreatmentPlannerAgent` plans from the cell and adds
  `escalation_considerations` (plain-language, measurement-indexed replacement for the end-stage table).
* The moderator returns `suggested_extent` (default) or `suggested_patel` (legacy opt-in only), plus
  `feature_bundles` and `screen_reconciliation`; `consensus_markdown` renders "Measured extent summary" and
  "Feature measurements" sections.

## 7. Running against a local VLM

Any OpenAI-compatible vision server works (llama.cpp `llama-server`, Ollama, vLLM); see
`docs/NEMOTRON_LOCAL.md` for serving recipes.

```bash
export NEMOTRON_BASE_URL=http://127.0.0.1:8080/v1          # local server, no data leaves the machine
python -m ecr_cds.cli conference --dicom scan.zip --out out --llm nemotron \
       --buccal 0,1,0                                       # optional patient frame for BL / MD extents
python -m ecr_cds.cli conference --dicom scan.zip --out out --llm none           # geometry-only bundles
python -m ecr_cds.cli conference --dicom scan.zip --out out --no-vlm-features    # VLM for the differential only
python -m ecr_cds.cli conference --dicom scan.zip --out out --staging-scheme patel_3d   # legacy rev <= 2.8 path
python -m ecr_cds.cli diagnocat-run --dicom scan.zip --diagnocat-json export.json --out out --llm nemotron
```

Python:

```python
from ecr_cds.agents import CaseConference, auto_backend
conf = CaseConference(backend=auto_backend("nemotron"))            # feature_backend defaults to the same backend
report = conf.run(volume, segmenter, seed, "clinician", buccal_direction=(0, 1, 0))
report["suggested_extent"]            # descriptors + measurements + narrative
report["feature_bundles"][cid]        # per-candidate bundle (dict)
```

A separate feature model can be passed with `CaseConference(feature_backend=...)`. Outputs land in
`conference_report.json` / `conference_summary.md`; evidence frames (`candidate_<id>_evidence_frame.png`) are
the images the model saw.

## 8. Tests

`tests/test_vlm_features.py` (fake OpenAI-compatible server from `tests/fake_vlm_server.py`, feature modes
`agree`, `disagree_samples`, `invalid`, `no_features`, `stage`):

* schema validation and coercion, prompt guard (no staging / diagnosis / treatment);
* geometric measurements against planted phantom truth (extent, CEJ / crest limits, circumferential extent,
  canal distance / communication, confidence rules, unavailable landmarks, in-plane frame);
* geometry-only bundle, agreeing VLM values verified, disagreeing values flagged with geometry kept;
* unverifiable VLM values (no canal mask / no CEJ) flagged, hidden from the extent summary, planner asks for
  canal delineation;
* abstention, invalid answers, structurally unusable answers, no image / text-only backend;
* extent summary + mapping (phantom cells, modifiers, policy context, 60-row table, no staging codes);
* conference end to end with the fake server (evidence kinds, bundle validity, extent summary, plan keyed by
  the measurement cell, no `suggested_patel` / `patel_code` / `[1-4][A-D][dp]` strings in the report,
  requires_confirmation, irreversible options flagged, skeptic raises disagreement) and the legacy opt-in.

`tests/test_agents.py` was updated to the measurement default and keeps a legacy `patel_3d` reproduction test.

## 9. Limitations

* **Not clinically validated.** All thresholds are engineering bands on synthetic phantoms; measurement
  accuracy on patient data depends on the automatic masks, the axis estimate and the CEJ / crest landmarks.
* The candidate mask comes from the cervical screen; a wrong or truncated mask produces wrong but confidently
  provenance-stamped measurements (flags `candidate_threshold_sensitive`,
  `candidate_touches_crop_boundary_extent_may_be_truncated`, `near_metal_streaks_may_distort_mask`).
* Without reviewed CEJ / apex points, CEJ-referenced measurements are capped at 0.5 (0.3 for the crest-offset
  assumption); without a canal mask the pulp relationship is `not_assessable` and no model value substitutes.
* MD / BL labels are only patient-oriented when `--buccal` is given.
* The VLM reads one 2-D evidence frame; its categorical features are opinions about an image and get a capped
  nudge only. The build sandbox had no GPU: the VLM path is exercised against the scripted fake server only, so
  prompt behaviour on a real Nemotron / other local model (literalness, abstention rate, estimate quality) has
  not been measured.
* The measurement-indexed mapping reuses the evidence labels of the rev 2.8 matrix cells it was aligned with;
  it is a documented engineering mapping, not a guideline.
