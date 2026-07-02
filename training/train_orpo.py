#!/usr/bin/env python3
"""Stage 2 — ORPO preference pass with TRL, on the merged SFT checkpoint.

Usage:
    python train_orpo.py --config configs/orpo_plan_a.yaml
    python train_orpo.py --config configs/orpo_plan_a.yaml --dry-run

Input data contract (data/preference_pairs.jsonl), one JSON object per line:

    {
      "messages": [ ...conversation up to and including the last user/tool
                    turn, Hermes role/content format... ],
      "chosen":   "<preferred assistant completion text>",
      "rejected": "<dispreferred assistant completion text>"
    }

The conversation context is rendered with the same ChatML renderer used for
SFT (single source of truth for formatting), producing TRL's prompt/chosen/
rejected string columns. Pairs target the behaviors from Phase 1: plans-
before-acting vs dives-in, correct tool schema vs malformed, asks-clarifying
-question vs hallucinates-assumptions.

Design choices:
    * Runs on the MERGED bf16 SFT checkpoint with a FRESH small LoRA
      (r=32) — stages stay decoupled and independently debuggable.
    * ORPO needs no reference model (its odds-ratio penalty replaces the
      DPO reference), which is exactly why it was chosen for one GPU.
    * Loaded 4-bit by default: ORPO effectively doubles sequence memory
      (chosen + rejected both forward), and a small preference pass is
      far less sensitive to base precision than the SFT stage was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainkit.chat_format import IM_END, ChatFormatError, render_conversation
from trainkit.config import RunConfig, load_config
from trainkit.data import iter_jsonl
from trainkit.resume import find_resume_checkpoint
from trainkit.storage import require_free_gb

IM_START_ASSISTANT = "<|im_start|>assistant\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config and pairs, print stats, exit before model load")
    return ap.parse_args()


def render_pair(obj: dict) -> dict | None:
    """One raw pair -> TRL columns, or None if malformed."""
    messages = obj.get("messages")
    chosen = obj.get("chosen")
    rejected = obj.get("rejected")
    if not (isinstance(messages, list) and isinstance(chosen, str)
            and isinstance(rejected, str) and chosen.strip() and rejected.strip()):
        return None
    if chosen.strip() == rejected.strip():
        return None  # identical pair carries zero preference signal

    # Context must end on a non-assistant turn; the completions ARE the
    # assistant turn. Append a placeholder to satisfy the renderer, then
    # strip its body, leaving prompt = context + generation header.
    if messages and messages[-1].get("role") == "assistant":
        return None
    try:
        segs = render_conversation(messages + [{"role": "assistant", "content": ""}])
    except ChatFormatError:
        return None
    prompt = "".join(s.text for s in segs[:-1]) + IM_START_ASSISTANT

    def completion(text: str) -> str:
        text = text.rstrip()
        return text if text.endswith(IM_END) else f"{text}{IM_END}\n"

    return {"prompt": prompt, "chosen": completion(chosen), "rejected": completion(rejected)}


def load_pairs(cfg: RunConfig) -> list[dict]:
    path = cfg.orpo.pairs_path
    pairs, bad = [], 0
    for _, obj in iter_jsonl(path):
        if obj is None or not isinstance(obj, dict):
            bad += 1
            continue
        rendered = render_pair(obj)
        if rendered is None:
            bad += 1
            continue
        pairs.append(rendered)
    print(f"[pairs] {len(pairs)} usable, {bad} rejected from {path}")
    if not pairs:
        raise SystemExit(f"no usable preference pairs in {path}")
    if bad > len(pairs) * 0.2:
        print(f"[pairs] WARNING: {bad} rejected rows is high — inspect the pair "
              "generation pipeline before trusting this run")
    return pairs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.stage != "orpo":
        raise SystemExit(f"config stage is {cfg.stage!r}; this script trains 'orpo'")

    merged = Path(cfg.orpo.merged_model_path)
    print(f"[run] {cfg.run_name}: ORPO β={cfg.orpo.beta} on {merged}")
    if not (merged / "config.json").is_file():
        raise SystemExit(
            f"{merged} is not a merged checkpoint — run merge_and_export.py first"
        )

    require_free_gb(cfg.checkpointing.output_dir
                    if Path(cfg.checkpointing.output_dir).exists() else ".",
                    cfg.storage.min_free_gb_before_training, "pre-ORPO disk guard")

    pairs = load_pairs(cfg)
    lengths = sorted(len(p["prompt"]) + len(p["chosen"]) for p in pairs)
    print(f"[pairs] char-length p50={lengths[len(lengths)//2]} "
          f"p95={lengths[int(len(lengths)*0.95)]} max={lengths[-1]}")

    if args.dry_run:
        print("[dry-run] config and pairs validate. Exiting.")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import ORPOConfig, ORPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(merged), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if cfg.model.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        str(merged),
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
    )

    ds = Dataset.from_list(pairs)

    orpo_args = ORPOConfig(
        output_dir=cfg.checkpointing.output_dir,
        run_name=cfg.run_name,
        beta=cfg.orpo.beta,
        max_length=cfg.orpo.max_length,
        max_prompt_length=cfg.orpo.max_prompt_length,
        num_train_epochs=cfg.optim.num_epochs,
        per_device_train_batch_size=cfg.optim.per_device_batch_size,
        gradient_accumulation_steps=cfg.optim.gradient_accumulation,
        learning_rate=cfg.optim.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.optim.warmup_ratio,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        optim=cfg.optim.optimizer,
        bf16=(cfg.model.dtype == "bfloat16"),
        gradient_checkpointing=cfg.optim.gradient_checkpointing,
        logging_steps=cfg.optim.logging_steps,
        save_steps=cfg.checkpointing.save_steps,
        save_total_limit=cfg.checkpointing.save_total_limit,
        seed=cfg.optim.seed,
        report_to="none",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_ckpt = find_resume_checkpoint(
        cfg.checkpointing.output_dir, cfg.checkpointing.resume
    )
    if resume_ckpt:
        print(f"[resume] resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    adapter_dir = Path(cfg.checkpointing.output_dir) / "final_adapter"
    require_free_gb(cfg.checkpointing.output_dir, 5.0, "final adapter save")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (Path(adapter_dir) / "run_config.json").write_text(
        json.dumps({"config_file": args.config, "run_name": cfg.run_name}, indent=2)
    )
    print(f"[done] ORPO adapter saved to {adapter_dir}")
    print(f"[next] final merge: python merge_and_export.py --config {args.config} "
          f"--adapter {adapter_dir} --output artifacts/final_bf16 "
          "--gguf q4_k_m q6_k --llama-cpp-dir <path>")


if __name__ == "__main__":
    main()
