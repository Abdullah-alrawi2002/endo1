"""LoRA / QLoRA supervised fine-tuning of an open instruction model on the conference prompts.

Runs on a single Colab GPU (T4 16 GB with 4-bit loading for 7-8B models; A100 for 14B).  Requires:
    pip install "transformers>=4.44" "peft>=0.12" "trl>=0.9" datasets accelerate bitsandbytes

Example
-------
    python -m ecr_cds.finetune.train_lora --data finetune_data/split/train.jsonl \
        --eval finetune_data/split/eval.jsonl --base Qwen/Qwen2.5-7B-Instruct --qlora --out ft/qwen7b-ecr

The adapter (and optionally a merged model) is written to --out.  Serve it with vLLM / Ollama or load it
in-process with ``ecr_cds.agents.llm_backend.LocalHFBackend`` and set ``LLM_LOCAL_MODEL=<out>``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def load_rows(path: str, tasks: List[str]) -> List[Dict]:
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task", "radiologist") in tasks:
            rows.append({"messages": r["messages"]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="ft/adapter")
    ap.add_argument("--tasks", default="radiologist,skeptic")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true", help="4-bit base weights (needs bitsandbytes + CUDA)")
    ap.add_argument("--merge", action="store_true", help="also write a merged full model (needs RAM/VRAM)")
    ap.add_argument("--dry-run", action="store_true", help="validate data and token lengths only")
    a = ap.parse_args(argv)

    tasks = a.tasks.split(",")
    train = load_rows(a.data, tasks)
    evals = load_rows(a.eval, tasks) if a.eval else None
    print(f"train rows: {len(train)}" + (f", eval rows: {len(evals)}" if evals else ""))
    if not train:
        raise SystemExit("no training rows")

    from transformers import AutoTokenizer  # imported late so --help works without torch
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lens = [len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in train[: min(200, len(train))]]
    print(f"token length (first {len(lens)}): median {sorted(lens)[len(lens)//2]}, max {max(lens)}; max-len {a.max_len}")
    if max(lens) > a.max_len:
        print("WARNING: some examples exceed --max-len and will be truncated")
    if a.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    quant = None
    if a.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=quant, device_map="auto",
                                                 torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                    warmup_ratio=0.05, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                    logging_steps=5, save_strategy="epoch", eval_strategy="epoch" if evals else "no",
                    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
                    max_length=a.max_len, packing=False, gradient_checkpointing=True,
                    assistant_only_loss=True, report_to=[])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(evals) if evals else None, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "training_card.json"), "w") as f:
        json.dump({"base": a.base, "tasks": tasks, "n_train": len(train), "n_eval": len(evals or []), "epochs": a.epochs,
                   "lr": a.lr, "rank": a.rank, "alpha": a.alpha, "qlora": a.qlora,
                   "data_note": "Synthetic phantom cases unless clinician-confirmed rows were included; "
                                "this adapter learns the reasoning format and confounder awareness, not clinical truth."}, f, indent=2)
    if a.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(os.path.join(a.out, "merged"))
        tok.save_pretrained(os.path.join(a.out, "merged"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
