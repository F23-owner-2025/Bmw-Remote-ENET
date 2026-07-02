#!/usr/bin/env python3
"""Stage 1 — LoRA SFT of Qwen3.6-27B with Unsloth.

Usage:
    python train_sft.py --config configs/sft_plan_a.yaml
    python train_sft.py --config configs/sft_plan_a.yaml --dry-run

Pipeline inside this script:
    1. Load + validate config; disk-space guard.
    2. Tokenizer-only load; run the mask-verification gate on real data.
       (Hard fail before the 55GB model ever touches VRAM.)
    3. Tokenize, split, FFD-pack the dataset; print token/packing stats.
    4. Load the model via Unsloth, attach LoRA, report per-pattern target-
       module coverage, freeze the vision tower.
    5. Resolve resume checkpoint (integrity-checked), train, save adapter.

``--dry-run`` performs steps 1–3 and exits: it validates config, data,
masking and packing without a GPU, so you can run it on any machine
(including this repo's CI) before renting the training box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainkit.config import RunConfig, load_config
from trainkit.data import (
    BuildStats,
    build_packed_dataset,
    iter_jsonl,
    load_and_tokenize,
    rows_to_hf_columns,
    split_train_eval,
)
from trainkit.mask_check import verify_masking
from trainkit.resume import find_resume_checkpoint
from trainkit.storage import prune_checkpoints, require_free_gb


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a YAML run config")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/data/masking/packing and exit before loading the model",
    )
    return ap.parse_args()


def load_tokenizer_only(cfg: RunConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def run_mask_gate(cfg: RunConfig, tokenizer) -> None:
    if not cfg.mask_check.enabled:
        print("[mask-check] DISABLED by config — you are trusting the mask blind.")
        return
    conversations = []
    for _, obj in iter_jsonl(cfg.data.train_path):
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            conversations.append(obj["messages"])
        if len(conversations) >= cfg.mask_check.sample_size * 5:
            break  # plenty for a sample; no need to parse the full corpus twice
    report = verify_masking(
        conversations,
        tokenizer,
        sample_size=cfg.mask_check.sample_size,
        min_trainable_fraction=cfg.mask_check.min_trainable_fraction,
        max_trainable_fraction=cfg.mask_check.max_trainable_fraction,
        max_tokenization_drift=cfg.mask_check.max_tokenization_drift,
    )
    print(report.summary())
    if not report.passed:
        raise SystemExit(
            "mask verification FAILED — refusing to train. Fix the data or the "
            "chat rendering before spending GPU-hours."
        )


def build_datasets(cfg: RunConfig, tokenizer):
    from datasets import Dataset

    examples, stats = load_and_tokenize(
        cfg.data.train_path, tokenizer, cfg.model.max_seq_len
    )
    if not examples:
        raise SystemExit(f"no usable examples in {cfg.data.train_path}")

    if cfg.data.eval_path:
        eval_examples, _ = load_and_tokenize(
            cfg.data.eval_path, tokenizer, cfg.model.max_seq_len
        )
        train_examples = examples
    else:
        train_examples, eval_examples = split_train_eval(
            examples, cfg.data.eval_fraction, cfg.data.shuffle_seed
        )

    train_rows, stats = build_packed_dataset(
        train_examples,
        cfg.model.max_seq_len,
        mode=cfg.packing.mode,
        reset_position_ids=cfg.packing.reset_position_ids,
        shuffle_seed=cfg.data.shuffle_seed,
        stats=stats,
    )
    print("[data] " + "\n[data] ".join(stats.summary().splitlines()))

    train_ds = Dataset.from_dict(
        rows_to_hf_columns(train_rows, tokenizer.pad_token_id, cfg.model.max_seq_len)
    )
    eval_ds = None
    if eval_examples:
        eval_rows, _ = build_packed_dataset(
            eval_examples,
            cfg.model.max_seq_len,
            mode=cfg.packing.mode,
            reset_position_ids=cfg.packing.reset_position_ids,
            shuffle_seed=cfg.data.shuffle_seed,
        )
        eval_ds = Dataset.from_dict(
            rows_to_hf_columns(eval_rows, tokenizer.pad_token_id, cfg.model.max_seq_len)
        )
    return train_ds, eval_ds, stats


def load_model_and_lora(cfg: RunConfig):
    # Imported here so --dry-run works on GPU-less machines. Unsloth must be
    # imported before transformers-model instantiation to apply its patches.
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name_or_path,
        max_seq_length=cfg.model.max_seq_len,
        dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
        load_in_4bit=cfg.model.load_in_4bit,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        use_rslora=cfg.lora.use_rslora,
        use_gradient_checkpointing="unsloth" if cfg.optim.gradient_checkpointing else False,
        random_state=cfg.optim.seed,
    )

    report_lora_coverage(model, cfg)

    if cfg.model.freeze_vision_tower:
        frozen = 0
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ("visual", "vision_tower", "vision_model")):
                param.requires_grad_(False)
                frozen += 1
        print(f"[model] vision tower: froze {frozen} params "
              f"({'none found — text-only checkpoint?' if frozen == 0 else 'ok'})")
    return model, tokenizer


def report_lora_coverage(model, cfg: RunConfig) -> None:
    """Print how many modules each target pattern matched.

    Qwen3.6's hybrid Gated-DeltaNet layers may use different projection
    names than classic attention; a pattern that matches zero modules means
    part of the network is silently untrained, so we make it loud.
    """
    counts = {pat: 0 for pat in cfg.lora.target_modules}
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in counts and "lora" not in name.lower():
            counts[leaf] += 1
    print("[lora] target-module coverage:")
    zero = []
    for pat, n in counts.items():
        print(f"[lora]   {pat:<12} matched {n} modules")
        if n == 0:
            zero.append(pat)
    if zero:
        print(
            f"[lora] WARNING: {zero} matched nothing. Inspect model.named_modules() "
            "and extend lora.target_modules for the hybrid (DeltaNet) blocks."
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable params: {trainable/1e6:.1f}M / {total/1e9:.1f}B "
          f"({100.0*trainable/total:.3f}%)")


class StorageGuardCallback:
    """Trainer callback: pre-save disk guard + post-save retention backstop."""

    def __new__(cls, cfg: RunConfig):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                pruned = prune_checkpoints(
                    cfg.checkpointing.output_dir, cfg.checkpointing.save_total_limit
                )
                for p in pruned:
                    print(f"[storage] pruned stale checkpoint {p}")

            def on_step_begin(self, args, state, control, **kwargs):
                next_step = state.global_step + 1
                if next_step % cfg.checkpointing.save_steps == 0:
                    require_free_gb(
                        cfg.checkpointing.output_dir,
                        cfg.storage.min_free_gb_per_checkpoint,
                        f"checkpoint at step {next_step}",
                    )

        return _Impl()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.stage != "sft":
        raise SystemExit(f"config stage is {cfg.stage!r}; this script trains 'sft'")

    print(f"[run] {cfg.run_name}: SFT LoRA r={cfg.lora.r} α={cfg.lora.alpha} "
          f"on {cfg.model.name_or_path} @ seq {cfg.model.max_seq_len}")

    status = require_free_gb(
        cfg.checkpointing.output_dir if Path(cfg.checkpointing.output_dir).exists() else ".",
        cfg.storage.min_free_gb_before_training,
        "pre-training disk guard",
    )
    print(f"[storage] {status.free_gb:.1f} GiB free of {status.total_gb:.1f} GiB")

    tokenizer = load_tokenizer_only(cfg)
    run_mask_gate(cfg, tokenizer)
    train_ds, eval_ds, _ = build_datasets(cfg, tokenizer)
    print(f"[data] train rows: {len(train_ds)}"
          + (f", eval rows: {len(eval_ds)}" if eval_ds else ", no eval split"))

    steps_per_epoch = math.ceil(
        len(train_ds) / (cfg.optim.per_device_batch_size * cfg.optim.gradient_accumulation)
    )
    total_steps = math.ceil(steps_per_epoch * cfg.optim.num_epochs)
    print(f"[plan] {steps_per_epoch} optimizer steps/epoch, ~{total_steps} total")

    if args.dry_run:
        print("[dry-run] config, data, masking and packing all validate. Exiting.")
        return

    model, _unsloth_tok = load_model_and_lora(cfg)

    from transformers import Trainer, TrainingArguments, default_data_collator

    targs = TrainingArguments(
        output_dir=cfg.checkpointing.output_dir,
        run_name=cfg.run_name,
        num_train_epochs=cfg.optim.num_epochs,
        per_device_train_batch_size=cfg.optim.per_device_batch_size,
        gradient_accumulation_steps=cfg.optim.gradient_accumulation,
        learning_rate=cfg.optim.learning_rate,
        lr_scheduler_type=(
            "cosine_with_min_lr" if cfg.optim.scheduler == "cosine" else cfg.optim.scheduler
        ),
        lr_scheduler_kwargs=(
            {"min_lr_rate": cfg.optim.min_lr_ratio}
            if cfg.optim.scheduler in ("cosine", "cosine_with_min_lr") else {}
        ),
        warmup_ratio=cfg.optim.warmup_ratio,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        optim=cfg.optim.optimizer,
        bf16=(cfg.model.dtype == "bfloat16"),
        fp16=(cfg.model.dtype == "float16"),
        logging_steps=cfg.optim.logging_steps,
        save_steps=cfg.checkpointing.save_steps,
        save_total_limit=cfg.checkpointing.save_total_limit,
        save_strategy="steps",
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.checkpointing.save_steps if eval_ds else None,
        per_device_eval_batch_size=cfg.optim.per_device_batch_size,
        seed=cfg.optim.seed,
        report_to="none",
        remove_unused_columns=False,  # keep position_ids for packed rows
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[StorageGuardCallback(cfg)],
    )

    resume_ckpt = find_resume_checkpoint(
        cfg.checkpointing.output_dir, cfg.checkpointing.resume
    )
    if resume_ckpt:
        print(f"[resume] resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    adapter_dir = Path(cfg.checkpointing.output_dir) / "final_adapter"
    require_free_gb(cfg.checkpointing.output_dir, 5.0, "final adapter save")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "run_config.json").write_text(
        json.dumps({"config_file": args.config, "run_name": cfg.run_name}, indent=2)
    )
    print(f"[done] adapter saved to {adapter_dir}")
    print("[next] merge with: python merge_and_export.py --config "
          f"{args.config} --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
