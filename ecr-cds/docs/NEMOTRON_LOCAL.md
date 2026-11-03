# Local vision model: NVIDIA Nemotron 3.5 Bolt 74B-A9B (rev 2.8)

Rev 2.8 replaces the hosted text LLM in the radiologist role with a **local** vision-language model served on the
clinician's own hardware, so no image or chart text leaves the machine. The model is NVIDIA Nemotron 3.5 Bolt
74B-A9B, released 15 October 2026, a hybrid Mamba-2/Transformer mixture-of-experts VLM with 74 B total and 9 B active
parameters, distilled from Nemotron 3.5 Super VL (120B-A12B) and shipped as a 2-bit 24 GB build that runs on a single
RTX / DGX Spark class GPU under llama.cpp, Ollama, LM Studio or vLLM
([NVIDIA blog](https://blogs.nvidia.com/blog/nemotron-3.5-super-vl/),
[technical page](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/),
[technical report, arXiv 2610.33039](https://arxiv.org/html/2610.33039)).

**What it is in this system:** a second opinion on an evidence frame that the perception layer already produced.
It never detects, delineates or measures anything, its weight in the posterior is capped at 0.5 and is reduced to 0
(abstention) or 0.2 (weak image grounding) automatically. The rule-based posterior and all measurements are unchanged
by the model. Nemotron has **no medical or dental evaluation in its model card**; everything in this document about
its behaviour on CBCT is an assumption to be tested, not a result.

**Status in this delivery:** the sandbox used to build rev 2.8 has no GPU, so the backend was exercised only against a
scripted OpenAI-compatible stand-in (`tests/fake_vlm_server.py`, 30 software tests in `tests/test_nemotron_backend.py`).
The real model has not been run on any scan by us. Run the ablation below before trusting it.

---

## 1. Serve the model

All three options expose an OpenAI-compatible `/v1/chat/completions` endpoint, which is what `NemotronLocalBackend`
speaks. Facts below are taken from the model cards; check them against the current card before deploying.

### llama.cpp (recommended for the 24 GB GGUF)

From the [GGUF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/raw/main/README.md):

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF --local-dir bolt
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-LiLiCorr-GGUF --local-dir bolt-draft   # speculative draft, optional

llama-server --model bolt/<Bolt-24GB>.gguf --mmproj bolt/<mmproj>.gguf --jinja --flash-attn on \
             --temp 1.0 --top-p 0.95 -ngl 999 -c 65536 --cache-type-k q8_0 --cache-type-v q8_0 \
             --spec-type draft-dflash --spec-draft-model bolt-draft/<LiLiCorr>.gguf --port 8080
```

The card gives a 28 GB recipe (65 536 context, q8_0 KV cache) for 32 GB-plus GPUs and a 32 GB recipe (262 144
context) for 48 GB-plus GPUs. Reasoning mode is toggled per request with `chat_template_kwargs.enable_thinking`,
which the backend sends.

### Ollama

```bash
ollama run nemotron-3.5-bolt:74b-a9b          # ~25 GB download; vision + tools + thinking
export NEMOTRON_BASE_URL=http://localhost:11434/v1
```

The [Ollama library page](https://ollama.com/library/nemotron-3.5-bolt:74b-a9b) lists vision, tools and thinking
capabilities and notes that Apple Metal is not supported for this model.

### vLLM (BF16, datacentre GPUs)

From the [BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16):

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --trust-remote-code \
     --reasoning-parser ultra_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
     --limit-mm-per-prompt '{"image":1}' --port 8000
export NEMOTRON_BASE_URL=http://localhost:8000/v1
```

The card limits image inputs to one per prompt (PNG/JPEG, RGB) and targets A100 / H100 / Blackwell. Licence is
NVIDIA OpenMDW-1.1 (check the card for the exact terms before clinical deployment).

## 2. Point the system at it

```bash
export NEMOTRON_BASE_URL=http://localhost:8080/v1   # required; auto_backend() prefers nemotron when this is set
export NEMOTRON_MODEL=<id>                           # optional; otherwise the first id from GET /v1/models
export NEMOTRON_API_KEY=none                         # optional; only if the server enforces a key

python -m ecr_cds.cli conference   --dicom scan.zip --llm nemotron --verify-mode tta --counterfactual --out out
python -m ecr_cds.cli diagnocat-run --dicom scan.zip --diagnocat-json export.json --llm nemotron --counterfactual --out out
```

Python:

```python
from ecr_cds.agents import CaseConference
from ecr_cds.agents.llm_backend import NemotronLocalBackend
be = NemotronLocalBackend(n_samples=5, dse_threshold=0.3, enable_thinking=False)
conf = CaseConference(backend=be, max_rounds=1)
rep = conf.run(volume, segmenter, seed, "auto_radiolucency", verify=True, verify_mode="tta", counterfactual_check=True)
```

Notebook: Section 4d sets `LLM_BACKEND = "nemotron"` and the same knobs.

## 3. How a request is built

| Item | Behaviour | Why |
|---|---|---|
| Image | exactly one PNG per request: the provenance-stamped 8-panel evidence frame of the candidate (`frames_png`), never the raw volume or the 12 mm tiles | vLLM recipe allows one image; the frame carries orientation, spacing, source id and crop transform |
| Text | descriptors, 8 diagnostic question answers, fired criteria, skeptic challenges (JSON) | the model interprets established findings; it is told it may not invent new ones |
| Sampling | `n_samples` = 5 draws at temperature 1.0 / top_p 0.95 (model-card defaults); `n_samples=1` falls back to temperature 0 | needed for the entropy estimate below |
| Output | JSON (`response_format`), `<think>` blocks stripped, fenced/noisy JSON extracted | structured fusion |
| Reasoning | `chat_template_kwargs.enable_thinking` (default off) | thinking roughly doubles latency; test both |

## 4. Three guards on the model's opinion

1. **Discrete semantic entropy abstention.** The five structured answers are clustered on
   `leading diagnosis | sorted question answers` and the Shannon entropy of the cluster distribution is computed
   (`discrete_semantic_entropy`). Weight = 0.5 × (1 − DSE/threshold); at or above `dse_threshold` (default 0.3) the
   model **abstains** and the rule posterior is used untouched. This follows the selective-prediction protocol in
   [Kim et al., Eur Radiol 2026](https://link.springer.com/article/10.1007/s00330-026-12384-z), who used 15 samples at
   T = 1.0 and rejection at DSE > 0.3 / 0.6 and raised accepted-answer accuracy from 51.7% to 76.3% at 47% coverage.
   Our thresholds are their values, **not** tuned on dental data.
2. **Counterfactual grounding test** (`--counterfactual`). A second frame is rendered in which the candidate voxels are
   filled with the volume's own dentine (or enamel) reference level and the overlay removed; the panel titles say
   `COUNTERFACTUAL (candidate filled, not evidence)` and the PNG is stored under `cf_frames_png`, never in the evidence
   path. The model is asked again. If P(ECR) moves by less than 0.10 and the model still "sees" a lesion (or keeps
   P(ECR) ≥ 0.3), grounding is `weak` and the weight is capped at 0.2. This is a phrase-level adaptation of the
   perturbation audit in [CLEAR, arXiv 2610.38389](https://arxiv.org/html/2610.38389) and responds to the finding that
   VLMs weight text over image ([PMC12842777](https://pmc.ncbi.nlm.nih.gov/articles/PMC12842777/)).
3. **Hard cap.** `LLM_WEIGHT_MAX = 0.5`; the model can never outvote the rules and never changes a measurement, a
   stage code or a plan directly.

Everything is recorded per candidate in `candidates[*].llm_meta` (weight, DSE, clusters, grounding verdict) and
summarised in the consensus markdown under "Vision-language second opinion (local model)". The knowledge agent
attaches the `vlm_limits` statements whenever a model opinion was used.

## 5. Hardware and quantization caveats

* The 24 GB build is 2-bit; the technical report calls the compression "very aggressive and lossy by design". Across
  VLM benchmarks 2-bit quantization costs roughly 15% on average and 16 to 24 points on text-in-image reading
  ([arXiv 2511.08914](https://www.arxiv.org/pdf/2511.08914)); the language tower is the most sensitive part
  ([arXiv 2601.15287](https://web3.arxiv.org/pdf/2601.15287)). Expect the model to misread the small panel labels and
  scale bars on the evidence frame; the backend therefore also passes all numbers as text.
* BF16 under vLLM avoids that loss but needs ~150 GB of GPU memory.
* Latency: 5 samples × (1 + 1 counterfactual) requests per candidate. Budget around 10 to 60 s per candidate
  depending on hardware; the conference runs the perception layer first, so a slow model never blocks the screen.

## 6. Evaluate before trusting (required)

```bash
python -m ecr_cds.evaluation --n-cases 12 --seed 3 --llm nemotron --out ablation_nemotron.json
```

This adds `nemotron_frame`, `nemotron_frame_cf` and `nemotron_sharpen_cf` to the frozen rev 2.7 baseline and reports
perception metrics (Dice, hit rate, false candidates) separately from report metrics (top-1, ECR recall, FPR, Brier,
AUROC) plus `llm_abstain_rate`, mean weight and the share of `weak` grounding verdicts. Phantom results are software
checks; the clinical question is only answerable on expert-labelled CBCT.

## 7. What this is not

* Not a detector: a negative model opinion does not exclude ECR and a positive one is not a finding.
* Not calibrated: `p_ecr_uncalibrated` remains a ranking score, with or without the model.
* Not validated: no dental, radiological or medical evaluation of Nemotron 3.5 Bolt exists in its documentation,
  and we ran it only against a scripted stand-in.
