# Fine-tuning the second-opinion model (rev 2.6)

The conference's radiologist and skeptic agents can consult a language model. Out of the box that is
a general hosted model prompted with the descriptors, the answers to the eight diagnostic questions
and the fired criteria. `ecr_cds.finetune` turns the conference itself into a training-data
generator so a model can be specialised to this exact task and output contract.

## What is learned, honestly

| Learned from the shipped data | Not learned |
|---|---|
| Strict JSON in the two schemas the agents parse | Clinical truth - synthetic phantoms are not patients |
| Consistent use of the Q1-Q8 answers (external origin, canal independence, growth pattern, pericanalar dentine, morphology, multiplanar persistence, extent, clinical fit) | Real-world prevalence of mimics (caries, burnout, furcation) |
| Internal-vs-external resorption distinction (canal-centred balloon vs eccentric external defect) | Image reading - the shipped backends are text-only; tiles are only sent to multimodal hosted models |
| Flagging injected confounders: metal streaks, noise, partial volume, grey-level drift, restoration band | Calibration on patients: posteriors stay "uncalibrated" |
| Disagreeing with the rule engine where phantom truth says it is wrong (e.g. wide shallow NCCL scored as ECR) | Anything about the CEJ/apex when they are automatic estimates |

Clinician-confirmed cases (`from_confirmed_case`) are the only rows that carry clinical ground truth.
Keep them in their own split, weight them up (default x3), and report their metrics separately.

## Pipeline

```
ecr_cds/finetune/
  dataset.py        synthetic cases + confounders -> chat JSONL (RecordingBackend captures the live prompts)
  train_lora.py     LoRA / QLoRA SFT (transformers + peft + trl), Colab T4 defaults
  export_openai.py  hosted fine-tuning JSONL export (+ optional job submission)
  evaluate.py       score rules / hosted / fine-tuned backends on held-out prompts
agents/llm_backend.LocalHFBackend   in-process serving of the adapter; auto_backend picks it up from LLM_LOCAL_MODEL
```

1. **Generate** (CPU, about 1 min per case; resume-safe, stride over workers)
   `python -m ecr_cds.finetune.dataset --n-cases 72 --out finetune_data/raw_w0.jsonl --worker 0 --n-workers 2`
   Each case samples a morphology (ECR 40 %, open wedge 15 %, NCCL 20 %, internal resorption 15 %,
   intact 10 %), Patel-style height / spread / perforation / portal angle, and a confounder mix.
   The conference runs rule-only with a `RecordingBackend`; every prompt the radiologist and skeptic
   would have sent is paired with a target:
   * radiologist -> label-smoothed differential (truth 0.80; 0.60 for the genuinely ambiguous open
     wedge; incidental fragments that do not overlap the true lesion -> artefact/noise), key findings
     for/against taken from the Q1-Q7 evidence strings, uncertainties from the confounders;
   * skeptic -> the confounders that were really injected plus the geometry-based rule challenges.
   Which component carries the label is decided by overlap with the true lesion mask, not by size.
2. **Split by case** `ds.split(raw, out_dir, eval_fraction=0.2)` - never by example.
3. **Baseline** `python -m ecr_cds.finetune.evaluate --eval split/eval.jsonl --backend rules`
   (and `--backend perplexity|openai` for the un-tuned model).
4. **Train** `python -m ecr_cds.finetune.train_lora --data split/train.jsonl --eval split/eval.jsonl --base Qwen/Qwen2.5-7B-Instruct --qlora --out ft/adapter`
   (rank 16, alpha 32, lr 1e-4, 2 epochs, assistant-only loss, max length 4096; `--dry-run` checks
   token lengths). Hosted alternative: `export_openai.py --submit`.
5. **Evaluate** the adapter with `--backend local` (`LLM_LOCAL_MODEL=ft/adapter`) and compare the
   same metrics: JSON validity, top-1 agreement with truth, Brier / log-loss of P(ECR), ECR recall and
   false-positive rate, internal-vs-ECR confusions, over-confidence on ambiguous cases, confounder
   recall for the skeptic.
6. **Use** it: `auto_backend()` prefers `LLM_LOCAL_MODEL`, then `LLM_API_KEY` endpoints, then the
   Perplexity SDK. The blend stays 50/50 with the rule posterior and the model still cannot raise the
   confidence band by itself; the fine-tuned model is a better second opinion, not a new authority.

## Shipped data

`finetune_data/split/{train,eval}.jsonl` are generated from seed 0 (counts in `finetune_data/README.json`).
They are synthetic and contain no patient data. The Colab notebook (section 8b) concatenates them,
optionally adds fresh phantoms and confirmed cases, trains the adapter and evaluates it.

## Promotion rule

Do not switch the default backend to a fine-tuned model until it beats the baseline on (a) the
synthetic eval set **and** (b) a held-out set of clinician-confirmed cases that was never used for
training, with the same honesty flags (`requires_confirmation`, uncalibrated, negative != exclusion)
left in place. Log the adapter's `training_card.json` next to the conference reports it produced.
