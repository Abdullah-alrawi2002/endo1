"""Export the dataset for a hosted fine-tuning API (OpenAI-style chat JSONL) and optionally submit the job.

    python -m ecr_cds.finetune.export_openai --data finetune_data/split/train.jsonl --out finetune_data/openai_train.jsonl
    python -m ecr_cds.finetune.export_openai --data ... --out ... --submit --model gpt-4o-mini-2024-07-18

Hosted fine-tuning sends the prompts to the provider.  Synthetic phantom rows contain no patient data;
clinician-confirmed rows contain de-identified measurements only, but check your data-governance rules
before submitting them.
"""
from __future__ import annotations

import argparse
import json
import os


def export(data: str, out: str, tasks) -> int:
    n = 0
    with open(out, "w") as f:
        for line in open(data):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task", "radiologist") not in tasks:
                continue
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--tasks", default="radiologist,skeptic")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--suffix", default="ecr-conference")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",")
    n = export(a.data, a.out, tasks)
    print(f"wrote {n} rows -> {a.out}")
    eval_out = None
    if a.eval:
        eval_out = os.path.splitext(a.out)[0] + "_eval.jsonl"
        print(f"wrote {export(a.eval, eval_out, tasks)} eval rows -> {eval_out}")
    if a.submit:
        from openai import OpenAI
        client = OpenAI()  # OPENAI_API_KEY from the environment
        tf = client.files.create(file=open(a.out, "rb"), purpose="fine-tune")
        kw = {}
        if eval_out:
            kw["validation_file"] = client.files.create(file=open(eval_out, "rb"), purpose="fine-tune").id
        job = client.fine_tuning.jobs.create(training_file=tf.id, model=a.model, suffix=a.suffix,
                                             hyperparameters={"n_epochs": a.epochs}, **kw)
        print("submitted job", job.id, "status", job.status)
        print("when finished, set LLM_MODEL=<fine_tuned_model> and LLM_API_KEY to use it in the conference")


if __name__ == "__main__":
    main()
