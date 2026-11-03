# Improving recognition accuracy: evidence review and what rev 2.8 adopted

Request (rev 2.8): "improve recognition accuracy; use the local model Nemotron 3.5 Bolt 74B instead of an LLM."

This note records the literature consulted, the changes adopted, those deliberately **not** adopted, and the
ablation against the frozen rev 2.7 baseline. The rule from rev 2.7 still applies: a proposed change is a hypothesis
until the ablation shows it helps on this task.

## 1. What "recognition accuracy" means here

Two separate things are measured (`ecr_cds.evaluation`):

* **Perception** - does the candidate proposer find the planted lesion (hit rate, Dice, false candidates per case,
  reproducibility under re-processing)? This is pixel work and no language model is involved.
* **Report** - given the candidates, is the suggested diagnosis right (top-1, ECR recall, false-positive rate,
  Brier score, AUROC of the uncalibrated P(ECR))?

A vision-language model can only influence the second. Claims that a VLM "improves recognition" of a lesion it did
not detect are not supported by the literature below; the perception layer stays the detector.

## 2. Evidence table

| Finding | Source | Consequence for rev 2.8 |
|---|---|---|
| VLMs on radiology images: 17.3% accuracy, 74.4% hallucination; clinical context raised accuracy 10.6 → 24.0% but models favour text over image | [PMC12842777](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842777/) | model is a down-weighted second opinion; counterfactual image test added; `vlm_limits` knowledge statements |
| Discrete semantic entropy over 15 samples at T = 1.0, reject DSE > 0.3 / 0.6: GPT-4o accepted-answer accuracy 51.7 → 76.3% at 47% coverage | [Kim et al., Eur Radiol 2026](https://link.springer.com/article/10.1007/s00330-026-12384-z) | `NemotronLocalBackend` samples n = 5, clusters structured answers, abstains above 0.3 |
| Closed-set VLM accuracy may not reflect visual understanding; expected calibration error 18.2 | [J Imaging Inform Med 2026](https://link.springer.com/article/10.1007/s10278-026-02217-2) | no calibration claimed; `p_ecr_uncalibrated` kept |
| CLEAR: counterfactual, phrase-level grounding audit of radiology report generators | [arXiv 2610.38389](https://arxiv.org/html/2610.38389) | candidate-filled counterfactual frame; weak grounding caps weight at 0.2 |
| Perception-reasoning governance with ROI priors; reasoning must not override perception | [MICCAI 2026 paper 0077](https://papers.miccai.org/miccai-2026/paper/0077_paper.pdf) | architecture unchanged (rev 2.7 split), model receives the perception frame, not the volume |
| Decision-aware uncertainty protocol: test-time augmentation was the strongest error-detection signal, but rankings are not universal across tasks | [Semantic Scholar record](https://www.semanticscholar.org/paper/d08372e4d2695f6d4ae03baf17659eb7b3fe7c1a) | verifier `mode="tta"` (gain, noise, threshold-jitter, smoothing paths; reports `dice_min`, `dice_by_path`); ablated, not assumed |
| 2-bit VLM quantization: about 15% average drop, OCRBench −16 to −24 | [arXiv 2511.08914](https://www.arxiv.org/pdf/2511.08914) | numbers always passed as text; 2-bit build treated as lossy; BF16 recipe documented |
| VLM quantization best practices: language tower most sensitive, keep vision encoder high precision | [arXiv 2601.15287](https://web3.arxiv.org/pdf/2601.15287) | documented in `NEMOTRON_LOCAL.md` |
| Gaussian sharpening raised CBCT root-resorption classifier accuracy 0.75 → 0.97 (EfficientNet-B1, OIERR) | [PMC12216651](https://pmc.ncbi.nlm.nih.gov/articles/PMC12216651/) | `sharpen_display` unsharp mask (display path only, recorded in provenance); ablation variant |
| CLAHE + VGG features + random forest, 81.9% accuracy for ERR on CBCT | [BMC Oral Health 2024](https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-024-03910-w) | CLAHE already display-only in rev 2.7; kept |
| Simulated resorption on CBCT: 3D rendering hurt AlexNet performance, especially on endodontically treated teeth without resorption | [Dent Traumatol 2025](https://onlinelibrary.wiley.com/doi/10.1111/edt.70105) | frames stay multiplanar slices, no volume rendering; root-filled teeth flagged as hard cases |
| Hierarchical segment-then-classify 3D U-Net for cervical invasive root resorption | [BMC Oral Health 2024](https://link.springer.com/article/10.1186/s12903-024-04718-4) | matches our proposer → verifier → classifier order |
| Self-supervised DINO features for ECR on periapical radiographs, 85.6% CV accuracy | [PubMed 37977219](https://www.ncbi.nlm.nih.gov/pubmed/37977219) | future fine-tuning target (FINETUNING.md); not implementable without labelled CBCT |
| ToothFairy3: 582 CBCT volumes, 77 classes incl. 32 pulp cavities; anatomy-aware label smoothing, tiny-structure weighted loss, left-right mirroring | [MICCAI 2026 paper 0853](https://papers.miccai.org/miccai-2026/1066-Paper0853.html) | recommended training set if the canal/CEJ segmenter is ever learned; listed as prerequisite, not done |
| DentVFM dental foundation model (DINOv2-style, 3D CBCT included) | [alphaXiv 2510.14532](https://www.alphaxiv.org/abs/2510.14532) | candidate backbone for a future learned proposer |
| AI for orthodontically induced root resorption: meta-analysis, heterogeneous and mostly 2D | [MDPI Dent J 2026](https://www.mdpi.com/2304-6767/14/2/79) | caution on transferring 2D results |
| CBCT grey values are not Hounsfield units | [PubMed 42279004](https://pubmed.ncbi.nlm.nih.gov/42279004/) | volume-relative reference levels retained; no universal threshold |
| Nemotron 3.5 Bolt: hybrid Mamba-2/Transformer MoE VLM, C-RADIO v4-H encoder, 9B active / 74B total; 2-bit build "very aggressive and lossy by design"; no medical evaluation | [arXiv 2610.33039](https://arxiv.org/html/2610.33039), [NVIDIA blog](https://blogs.nvidia.com/blog/nemotron-3.5-super-vl/) | adopted as the local second-opinion model, with the guards above |

## 3. Adopted in rev 2.8

| Change | Where | Default | Effect on frozen baseline |
|---|---|---|---|
| Local Nemotron backend (OpenAI-compatible, 1 image/request, n-sample DSE abstention, thinking toggle) | `agents/llm_backend.py` | off unless `NEMOTRON_BASE_URL` set or `--llm nemotron` | none when off |
| Entropy-weighted fusion, hard cap 0.5, abstention → rules only | `agents/roles.py` RadiologistAgent | always when a model is present | none for rules-only runs |
| Counterfactual grounding test, weak grounding cap 0.2 | `agents/roles.py`, `perception/evidence.py` (`counterfactual_fill`) | off (`--counterfactual`) | none when off |
| TTA verifier (6 processing paths, `dice_min`) | `perception/tools.py reproduce_candidate(mode="tta")` | off (`--verify-mode tta`) | none when off |
| Unsharp-mask display option, recorded in frame provenance | `perception/display.py unsharp2d` | off (`--sharpen-display`) | none when off |
| Review-budget metrics (budget-1 true-first by reproducibility / by P(ECR)), AUROC, LLM stats | `evaluation/__init__.py` | always | metrics only |
| Three `vlm_limits` knowledge statements, cited whenever a model opinion was fused | `knowledge/guidelines.json` | always | report only |
| CLI flags `--llm nemotron`, `--verify-mode`, `--counterfactual`, `--sharpen-display` | `cli.py` | off | none |

## 4. Considered and not adopted

* **Using the VLM to find lesions** (asking the model "where is the resorption?"). Rejected: hallucination rates and
  text-over-image bias in the evidence above; the architecture guidance from rev 2.7 forbids a knowledge or language
  component creating findings.
* **Volume renderings in the frame.** Rejected on the Dent Traumatol 2025 result.
* **A universal intensity threshold** for "resorption". Rejected (CBCT grey values are not HU).
* **Learned 3D proposer (nnU-Net / DentVFM) now.** Not possible without labelled CBCT; the fine-tuning guide lists the
  data prerequisites. The ToothFairy3 and DentVFM entries are the recommended starting points.
* **Lower DSE threshold or more samples** than the cited protocol. Left at published values; tuning on phantoms would
  only fit the phantoms.
* **Generative image enhancement** (super-resolution, denoising nets) in the evidence path. Excluded per the standing
  constraint; unsharp masking is a deterministic display transform and is recorded as such.

## 5. Ablation against the frozen baseline (phantoms, n = 6, seed 3)

Results are written by `python -m ecr_cds.evaluation --n-cases 6 --seed 3 --variants verify_on,tta_verify --out ...`
and summarised in the rev 2.8 case review archive (`ecr_ablation_v28/ablation_results.json`). See the section
"Rev 2.8 ablation" appended below once the run completed. Phantom numbers are software checks; they say nothing about
clinical accuracy.

The Nemotron variants (`--llm nemotron`) could not be run in the build environment (no GPU); the numbers for them
are absent, not implied.

### Rev 2.8 ablation (completed; phantoms, n = 6, seed 3, same cases as the rev 2.7 run)

| Variant | Top-1 | ECR recall | ECR FPR | Brier | Dice (lesion cases) | Hit rate | False cand./case | Reproducibility true / false | AUROC true vs false | Budget-1 true-first (by reproducibility) | s/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen_baseline (rev 2.6/2.7 rules, no verifier) | 0.833 | 0.50 | 0.00 | 0.117 | 0.816 | 1.00 | 1.0 | n/a | n/a | n/a | 53.4 |
| verify_on (smooth re-processing, rev 2.7) | 0.833 | 0.50 | 0.00 | 0.117 | 0.816 | 1.00 | 1.0 | 0.833 / 0.613 | 1.00 | 1.00 | 56.6 |
| tta_verify (6 processing paths, min Dice; rev 2.8) | 0.833 | 0.50 | 0.00 | 0.117 | 0.816 | 1.00 | 1.0 | 0.875 / 0.657 | 1.00 | 1.00 | 57.3 |

Reading: on these six phantoms the TTA verifier changed **nothing** in the diagnosis or the perception metrics; the
true/false reproducibility gap is the same as the smooth path (0.218 vs 0.220), both separate planted lesions from
false candidates perfectly (AUROC 1.0) and both put the true candidate first under a one-review budget. The one
missed ECR case (recall 0.5) is missed by every variant for the same reason as in rev 2.7: the candidate is found
(hit rate 1.0) but the rule posterior prefers a confounder. This is reported as **no measured gain**, not as an
improvement; the ablation is too small and too synthetic to show a difference of the size one would expect, and the
TTA mode stays off by default. The Nemotron variants were not run (no GPU in the build environment).
