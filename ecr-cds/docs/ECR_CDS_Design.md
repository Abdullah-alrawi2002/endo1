# Clinical Decision Support Algorithm for External Cervical Resorption (ECR) Treatment Planning

**Input:** CBCT DICOM series (≈500 axial slices) · **Classification:** Patel 3D (height 1–4, circumferential spread A–D, canal proximity d/p) · **Output:** ranked endodontic treatment recommendation with rationale, safe-set and confirmation flags · **Learning:** constrained contextual-bandit RL with hybrid reward and off-policy promotion gate.

> Advisory system. Every irreversible option (extraction, decoronation, intentional replantation, hemisection) is a proposal that requires explicit clinician confirmation. The reference implementation lives in `ecr_cds/` and is exercised by `tests/test_pipeline.py` on a synthetic 500-slice DICOM phantom.

---

## 1. System overview

```
 DICOM folder (500 slices)
        │  pydicom: sort by slice normal, uniform-spacing check, rescale, geometry, provenance
        ▼
 CBCTVolume (Z,Y,X float32, spacing, origin, direction, metadata)  ── QA gate (voxel ≤0.2 mm, isotropic)
        │
        ▼
 Tooth / canal / bone segmentation (DentalSegmentator – nnU-Net)  ── hook: ToothSegmenter
        │
        ▼
 Preprocess inside tooth ROI: clip 5–95 pct → [0,1] → Gaussian σ=1 vx   (Ravikumar 2025)
        │
        ▼
 ECR lesion probability map (texture SVM on LGRE/HGRE  or  nnU-Net) ── hook: LesionSegmenter
        │  threshold (site-tuned, Youden) → erode → largest component → dilate
        ▼
 Patel 3D parameter extraction along the tooth long axis (PCA)
        │  height 1–4 · spread A–D (angular coverage) · proximity d/p (EDT to canal)
        │  + continuous evidence: volume mm³, min canal distance, calcified fraction, portal angle, margins
        ▼
 Rule-based decision matrix (32 cells, ESE-2023 / Patel-2018 anchored) + clinical modifiers R1–R11
        │  → SAFE SET (allowed actions) + default ranking + prognosis prior
        ▼
 RL layer: constrained Thompson-sampling bandit re-ranks INSIDE the safe set
        │  → chosen action, propensities, reinspection request if borderline
        ▼
 Report (JSON) → clinician review → clinician decision + 6/12/24-month outcome
        │
        └──► hybrid reward R = R_rubric + R_format + R_diag → posterior update
             → offline evaluation (doubly-robust) → promotion gate → new policy version
```

---

## 2. Stage 1 – DICOM ingestion (`dicom_loader.py`)

| Step | Design decision | Why |
|---|---|---|
| Series selection | Pick the `SeriesInstanceUID` with the most slices unless specified | CBCT exports often bundle scout/panoramic reformats |
| Ordering | Sort by projection of `ImagePositionPatient` on the slice normal (cross product of `ImageOrientationPatient`) | `InstanceNumber` is unreliable across vendors |
| Spacing validation | Reject if any inter-slice gap deviates >25 % from the median | Missing slices silently bias height measurements |
| Intensity | Apply `RescaleSlope/Intercept`, keep float32 | CBCT grey values are **not** Hounsfield units, so no absolute HU thresholds anywhere downstream (Spin-Neto et al. 2011) |
| Provenance | Record kVp, mA, exposure, FOV, voxel size, manufacturer, isotropy flag | Spin-Neto showed windowing, plane definition and slice thickness change measurements; provenance is required for auditability and RL state |
| Memory | Two-pass read (headers, then pixels) | 500 × 512² × 4 B ≈ 0.5 GB; predictable on a workstation |

**QA gate:** voxel > 0.2 mm → flag that d/p calls are unreliable and recommend a small-FOV high-resolution rescan (≤0.1 mm; the ECR series in the attached papers used 76–85 µm).

---

## 3. Stage 2 – Segmentation (`segmentation.py`)

Two pluggable backends with a shared contract (return probability maps so thresholds can be re-tuned per site):

1. **Tooth / canal / bone:** DentalSegmentator (nnU-Net in 3D Slicer) as used by Ravikumar et al.; crop to the target tooth, allow manual correction.
2. **ECR lesion:** reference implementation follows Ravikumar et al. (arXiv 2501.05236): voxel-wise texture features → linear SVM → post-processing (erosion → largest connected component → dilation, kernel 6 vx). LGRE and HGRE were the most discriminative features; neighbourhood radius 5 was optimal (Dice 0.59 ± 0.19 on 6 scans, with failures when pulp was absorbed into the lesion). The implementation therefore **explicitly excludes the canal mask from the lesion** and uses cropped training regions to control class imbalance.

**Threshold policy.** Al Fartousi et al. (JCM 2026) showed on 358 teeth that a manufacturer-fixed 0.50 probability threshold is rarely optimal on CBCT (re-tuning raised missed-canal sensitivity from 69.6 % to higher levels at a specificity cost) and that sub-millimetre lesions had only 33.6 % sensitivity. The CDS therefore (a) stores the threshold as a site-calibrated parameter, (b) reports calibration (Brier, intercept/slope) and (c) emits an explicit "no lesion detected – low sensitivity for sub-mm lesions" message rather than a silent negative.

---

## 4. Stage 3 – Patel 3D parameter extraction (`patel_classifier.py`)

All geometry is computed along the **tooth long axis** (PCA of the tooth mask, oriented so the narrow end = apex), making the result independent of acquisition plane.

| Parameter | Computation | Category rule (Patel et al. 2018, IEJ 51:206) |
|---|---|---|
| Height | Project lesion voxels on axis; apical extent relative to CEJ (enamel-shell estimate or supplied); bone crest = 2nd percentile of peri-radicular bone level; root length = apex − CEJ | 1: not apical to bone crest (supracrestal) · 2: apical to crest, within coronal third · 3: into middle third · 4: into apical third |
| Circumferential spread | Angle of each lesion voxel around the axis → 1° bins → circular closing (6°) → covered degrees | A ≤ 90° · B ≤ 180° · C ≤ 270° · D > 270° |
| Proximity | Euclidean distance transform of canal mask (mm) evaluated at lesion voxels; perforation if overlap > 0 | p if d_min ≤ max(2 voxels, 0.3 mm) else d |

Additional continuous evidence feeds the RL state and the report: lesion volume (mm³), min canal distance, **calcified fraction** (fraction of lesion voxels with HGRE above the dentine median – the repair-phase proxy Ravikumar et al. found by k-means), **portal-of-entry angle** (accessibility proxy: within ±45° of buccal/lingual = surgically accessible), and **margins to the nearest category boundary**. A case is flagged `borderline` if height margin < 0.5 mm, spread margin < 10° or proximity margin < 1 voxel – this is the trigger for the reinspection action in the RL layer.

Phantom validation (0.1 mm isotropic, 500 slices): all 8 planted codes (1Ad, 2Ad, 2Bp, 2Cp, 3Bp, 3Dd, 4Ap, 4Dp) were recovered; apical extent within 0.2–0.6 mm of truth.

---

## 5. Stage 4 – Decision matrix (`decision_matrix.py`)

### 5.1 Treatment action space

| Action | Definition | Irreversible |
|---|---|---|
| `periodic_review` | No active treatment; annual clinical review, CBCT on change | no |
| `external_repair` | Periodontal flap surgery, curettage ± 90 % TCA, bioceramic/GIC repair, no RCT | no |
| `external_repair_vpt` | External repair + vital pulp therapy (bioceramic capping) | no |
| `external_repair_rct` | External repair + root canal treatment | no |
| `internal_repair_rct` | RCT + repair from within the canal | no |
| `intentional_replantation` | Extraction, extra-oral repair, replantation, 2-week flexible splint | yes |
| `root_resection_hemisection` | Multirooted teeth with lesion confined to one root | yes |
| `decoronation` | Ridge preservation (growing patients / future implant site) | yes |
| `extraction` | Atraumatic/surgical extraction + replacement planning | yes |

### 5.2 Base matrix (32 cells)

Evidence anchors: ESE Position Statement 2023 class table (external repair only 1–2Ad, 2Bd; external repair + RCT 1–2Ap, 2Bp, 2Cp; internal repair + RCT 2Bp, 2Cp, 2Dp, 3Cp, 3Dp; intentional replantation 3Ad/p, 3Bd/p; decoronation 3–4C/D; periodic review 2–4C/Dp; extraction for untreatable ECR) and Patel 2018 Part 2 management rules. Cells not covered by either are marked `expert-extrapolation`. Prognosis priors are anchored on Heithersay's outcomes (class 1–2: 100 %, class 3: 77.8 %, class 4: 12.5 %) and the Irinakis 2022 failure gradient.

| Height | Spread | **d** – confined to dentine | **p** – probable pulpal involvement |
|---|---|---|---|
| **1** supracrestal | A ≤90° | external_repair (0.95) | external_repair_vpt → external_repair_rct (0.90) |
| | B ≤180° | external_repair (0.92) | external_repair_rct (0.85) |
| | C ≤270° | external_repair → ext+RCT (0.80) | external_repair_rct → internal (0.70) |
| | D >270° | external_repair_rct → monitor/extraction (0.60)* | external_repair_rct → internal/monitor/extraction (0.50)* |
| **2** coronal ⅓, subcrestal | A | external_repair (0.92) | external_repair_rct → VPT/internal (0.85) |
| | B | external_repair (0.88) | external_repair_rct → internal/hemisection/extraction (0.78) |
| | C | external_repair_rct → ext/internal (0.70) | **internal_repair_rct** → ext+RCT/monitor (0.65) |
| | D | periodic_review → internal/decoronation/extraction (0.45)* | **internal_repair_rct** → monitor/decoronation/extraction (0.45) |
| **3** middle ⅓ | A | intentional_replantation → ext repair (0.78) | intentional_replantation → internal (0.72) |
| | B | intentional_replantation → ext+RCT/internal (0.72) | intentional_replantation → internal/hemisection (0.65) |
| | C | periodic_review → internal/decoronation/extraction (0.45) | internal_repair_rct → monitor/decoronation/extraction (0.45) |
| | D | periodic_review → decoronation/extraction (0.30) | internal_repair_rct → monitor/decoronation/extraction (0.30) |
| **4** apical ⅓ | A | intentional_replantation → monitor/extraction (0.40) | intentional_replantation → internal/extraction (0.35) |
| | B | periodic_review → IR/extraction (0.30) | extraction → IR/hemisection/monitor (0.25) |
| | C | periodic_review → decoronation/extraction (0.15) | extraction → decoronation/monitor (0.12) |
| | D | extraction → decoronation/monitor (0.10) | extraction → decoronation/monitor (0.08) |

\* expert extrapolation (no guideline cell). Bracketed numbers are the prognosis priors. Full machine-readable version: `docs/decision_matrix_32.csv`.

### 5.3 Clinical modifier rules (applied in order)

| Rule | Trigger | Effect | Source |
|---|---|---|---|
| R1 | External access not feasible (interproximal portal, deep subcrestal) | Promote internal_repair_rct (height ≤2) or intentional_replantation (height ≥3) | Patel 2018 pt 2 |
| R2 | p-lesion, symptoms, or non-vital pulp | Remove repair-without-RCT; remove VPT if non-vital | Patel 2018 pt 2 |
| R3 | Persistent intracanal haemorrhage after 2 × Ca(OH)₂ | Demote internal repair; promote hemisection (multirooted) or extraction/decoronation | Talpoş-Niculescu 2025 cases 1–3 |
| R4 | Multirooted, lesion on a single root | Hemisection ranked above extraction | Talpoş-Niculescu 2025 case 2 (#17 DB root) |
| R5 | Non-restorable or inadequate periodontal support | Remove all repair/replantation options; flag `untreatable` | ESE 2023 |
| R6 | Untreatable + asymptomatic | Periodic review first (patient-elected, informed of risks) | Patel 2018 pt 2 |
| R7 | Untreatable + symptomatic | Extraction (adult) or decoronation (growing) first | Patel 2018 pt 2 |
| R8 | Growing patient | Decoronation ranked above extraction | ESE 2023 |
| R9 | Repair-phase lesion (calcified fraction ≥ 0.5) and no symptoms, height ≥3 | Promote periodic review before irreversible treatment | Mavridou (bone ingrowth + no symptoms → monitor); Ravikumar calcification cluster |
| R10 | High root-fracture risk / periodontal disease | Remove intentional replantation | Patel 2018 pt 2 contraindications |
| R11 | Code ∈ end-stage set | Attach end-stage matrix (§6) | – |

The output of this stage is the **safe set** (ordered list of permitted actions). The RL layer may re-rank but never add an action.

---

## 6. End-stage lesion management matrix

End-stage set: `2Dd, 2Dp, 3Cd, 3Cp, 3Dd, 3Dp, 4Ad, 4Ap, 4Bd, 4Bp, 4Cd, 4Cp, 4Dd, 4Dp`.

| Symptoms | Structure / perio | Patient | 1st line | 2nd line | 3rd line |
|---|---|---|---|---|---|
| Asymptomatic | Restorable, perio adequate | Adult | internal_repair_rct (2Dp/3Cp/3Dp) or periodic_review (d-lesions) | intentional_replantation (4A/B, favourable root form) | extraction |
| Asymptomatic | Restorable, perio adequate | Growing | periodic_review | internal_repair_rct | decoronation |
| Asymptomatic | Non-restorable / perio compromised | Adult | periodic_review (patient-elected, informed of fracture/discoloration/pain risk) | extraction + replacement (implant / bridge / RBB) | decoronation if site preservation wanted |
| Asymptomatic | Non-restorable / perio compromised | Growing | periodic_review | decoronation | extraction + space management (MDT) |
| Symptomatic | Restorable, perio adequate | Adult | internal_repair_rct if bleeding controllable | hemisection (multirooted, single-root lesion) | extraction + replacement |
| Symptomatic | Restorable, perio adequate | Growing | internal_repair_rct | decoronation | extraction |
| Symptomatic | Non-restorable / perio compromised | Adult | extraction (plan surgical; cervical fracture risk) | decoronation | – |
| Symptomatic | Non-restorable / perio compromised | Growing | decoronation | extraction + space management | – |
| Any | Multirooted, lesion confined to one root | Any | root_resection_hemisection + RCT of remaining roots | extraction | – |
| Any | Repair-phase lesion, no symptoms | Any | periodic_review | – | – |

Recall protocol for monitored teeth (Patel 2018 pt 2): annual review – symptoms, periodontal attachment, crown colour, pulp status, radiographic progression; CBCT only when change is suspected. Machine-readable: `docs/end_stage_matrix.csv`.

---

## 7. Stage 5 – Reinforcement-learning layer (`rl_policy.py`)

### 7.1 Formulation

ECR treatment is a one-shot decision per tooth with a delayed outcome, and ECR datasets are small (6 scans in Ravikumar et al.; 42 teeth in Jeng et al. 2024). A **constrained contextual bandit** is therefore the right first-stage RL formulation; a full MDP/agentic policy (§7.5) is the planned second stage.

* **State** `x` (21 features): Patel height/spread/proximity, lesion volume, min canal distance, spread degrees, calcified fraction, borderline flag, clinical modifiers (symptoms, vitality, restorability, perio, multirooted, accessibility, bleeding, growth, aesthetic zone), tooth type, age.
* **Action space:** the 9 treatments, **masked to the rule-derived safe set**.
* **Policy:** Thompson sampling with  
  `score(a | x, cell) = w·Beta(α_cell,a, β_cell,a) + (1 − w)·(0.5 + θ_aᵀx)`  
  – a tabular Beta posterior per (Patel cell, action) seeded from the matrix (primary gets mean = prognosis prior, alternatives decay by rank, prior strength n₀ = 20) plus a Bayesian linear model per action for generalisation across cells. The system therefore **starts as the guideline** and only drifts with evidence.
* **Propensities** (action probabilities over 512 posterior samples) are logged with every recommendation to enable unbiased off-policy evaluation later.

### 7.2 Hybrid reward (adapted from OralGPT-Plus, arXiv 2603.06366)

`R(τ) = R_rubric + R_format + R_diag`

| Term | Definition | Purpose |
|---|---|---|
| `R_rubric ∈ [0,1]` | Partial-credit agreement between recommendation and the clinician's final choice, graded by invasiveness-tier distance (same tier 0.75, one tier 0.5, two 0.25, three 0.1, four 0.0); once follow-up exists it becomes 0.5·agreement + 0.5·outcome (success at ≥12 months weighted 1.0, <12 months 0.5), −0.25 if the tooth was lost after an interventional recommendation | Dense, continuous supervision instead of sparse 0/1 (the paper's rubric reward) |
| `R_format` | +0.1 if action ∈ safe set, rationale present and irreversible actions flagged for confirmation; **−1.0** on any guardrail violation | Format/guardrail adherence |
| `R_diag` | `𝟙[conf > η] · 𝟙[borderline] · α · (H − n_reinspections)₊ · (1.0 if reinspection changed the code else 0.5)`, −0.1 if reinspection requested on a non-borderline case | Conditioned reinspection reward: rewards re-measurement **only** when the hypothesis is already credible and the case is near a category boundary – the mechanism the paper used to stop tool-use reward hacking (collapse–rebound) |

Reinspection actions available to the system: re-slice along the root axis, re-threshold the lesion map, re-estimate the CEJ/bone crest, recompute canal proximity at higher resolution.

### 7.3 Update and promotion gate

1. Reward is mapped to [0,1] and used to update the Beta posterior of the chosen (cell, action) and the linear model; the clinician's divergent choice (if in the safe set) receives a 0.5 supervised credit.
2. Every policy version is **evaluated offline** on the logged data with a doubly-robust estimator (IPS weights from the logged propensities + per-action reward model) and cluster-bootstrap 95 % CI.
3. **Promotion rule:** deploy the candidate only if its DR lower bound ≥ the deployed policy's value and n ≥ 50 logged cases.
4. Drift monitoring: per-cell posterior means, acceptance rate, override rate by treatment tier, calibration of the lesion detector (Brier, slope/intercept), re-tuned Youden thresholds – all following the validation design of Al Fartousi et al.

Simulation (test suite): for a 2Cp context where clinicians consistently chose external repair + RCT with 90 % success, the bandit moved from the guideline default (internal repair) to external repair + RCT within 80 cases (P = 0.84), never left the safe set, and the DR estimate (1.03, CI 1.01–1.05) beat the behaviour value (0.89), passing the gate.

### 7.4 Safety constraints

* Action masking – the policy cannot propose outside the rule-derived safe set.
* Irreversible actions are never auto-finalised (`requires_confirmation = True`).
* Guardrail violations cost −1.0 reward and are logged as incidents.
* Learning never changes the classification stage – only the ranking inside permitted treatments.
* Human expert remains the reference: the Akyüz et al. (JOE 2026) comparison showed VLMs reached 94 % accuracy on final treatment selection but were weak on complications such as perforation, so perforation (d/p) remains a geometric measurement, not an LLM judgement.

### 7.5 Phase 2 – agentic re-inspection policy (GRPO)

When ≥ 1 000 labelled cases exist, replace the bandit's fixed reinspection heuristic with an agentic policy trained with GRPO exactly as in OralGPT-Plus: thought → action (zoom-in on portal of entry, re-slice along root axis, mirror-in on the contralateral tooth, measure) → observation loop, K ≤ 6 steps, trajectory reward as in §7.2, rubric evaluator aligned to the Patel definitions. Supervised warm-start on expert trajectories (dentist sampling of CBCT inspection sequences), then RL.

---

## 8. Validation plan

| Level | Metric | Reference method |
|---|---|---|
| Lesion detection | Sensitivity/specificity/AUC at fixed and Youden thresholds, Brier, calibration slope, decision-curve net benefit, stratified by lesion size and voxel size, cluster-bootstrap CIs (cluster = scan) | Al Fartousi et al. 2026 |
| Lesion segmentation | Dice, precision/recall, leave-one-patient-out | Ravikumar et al. 2025 |
| Patel classification | Exact-code agreement and per-dimension κ against two endodontists adjudicated by a third; margin-stratified error analysis | Akyüz et al. 2026 (κ = 0.95 between experts) |
| Recommendation | Agreement with multi-rater consensus by invasiveness tier; override analysis; prospective 6/12/24-month survival and success per code | Jeng 2024, Jebril 2020 |
| RL | Offline DR value vs. deployed policy, regret vs. guideline baseline, safe-set violation rate (must be 0) | §7.3 |

Phantom and histology-correlated cases (Talpoş-Niculescu 2025) serve as ground-truth anchors for perforation and repair-phase texture.

---

## 9. Repository layout

```
ecr_cds/
  dicom_loader.py      500-slice series → CBCTVolume (geometry, provenance, QA)
  preprocess.py        percentile clip, normalise, smooth, LGRE/HGRE texture maps
  segmentation.py      ToothSegmenter / LesionSegmenter hooks, texture-SVM reference, post-processing
  patel_classifier.py  axis-aligned height / spread / proximity + margins + calcified fraction
  decision_matrix.py   32-cell matrix, modifier rules R1–R11, end-stage matrix
  rl_policy.py         constrained Thompson-sampling bandit, hybrid reward, DR evaluation, gate
  pipeline.py          run_pipeline(dicom_dir, …) → JSON report
tests/test_pipeline.py synthetic 500-slice DICOM phantom; all stages tested
docs/                  this document, decision_matrix_32.csv, end_stage_matrix.csv
examples/              example_report_3Bp.json
```

Run: `pip install pydicom scikit-learn scipy numpy && python tests/test_pipeline.py`

---

## 10. Sources

* Patel S, Foschi F, Mannocci F, Patel K. External cervical resorption: a three-dimensional classification. Int Endod J 2018;51:206–214. https://onlinelibrary.wiley.com/doi/full/10.1111/iej.12824
* Patel S, Foschi F, Condon R, Pimentel T, Bhuva B. External cervical resorption: part 2 – management. Int Endod J 2018;51:1224–1238. https://onlinelibrary.wiley.com/doi/full/10.1111/iej.12946
* Patel S et al. ESE Position Statement on root resorption. Int Endod J 2023;56:792–801. https://onlinelibrary.wiley.com/doi/full/10.1111/iej.13916 (class table: https://helda.helsinki.fi/server/api/core/bitstreams/3298d985-5c82-495e-ba93-22a51a62cadf/content)
* Rossi-Fedele G et al. Prognostication of ECR based on Heithersay's and Patel et al. classifications: a narrative review. Int Endod J 2026. https://onlinelibrary.wiley.com/doi/full/10.1111/iej.70119
* Talpoş-Niculescu R-M et al. External root resorptions: CBCT assessment and histopathological characterization. Rom J Morphol Embryol 2025;66(2):391–398. doi:10.47162/RJME.66.2.13 (attached)
* Ravikumar S, Khan AA, Davis MC, Paniagua B. Automated ECR segmentation in CBCT using local texture features. arXiv:2501.05236 (attached)
* Fan Y et al. OralGPT-Plus: Learning to use visual tools via reinforcement learning for panoramic X-ray analysis. arXiv:2603.06366 (attached)
* Al Fartousi M, Buscot A, Gernhardt CR. Diagnostic performance of AI-based cloud software regarding endodontic findings on CBCT. J Clin Med 2026;15:4839 (attached)
* Akyüz İE, Kıvırcık BE, Aslan T. AI-assisted periapical radiographic assessment: lesion detection, complication analysis and treatment recommendations. J Endod 2026 (attached)
* Spin-Neto R et al. Exploring CBCT-based DICOM files: a systematic review. J Digit Imaging 2011;24:959–966 (attached)
