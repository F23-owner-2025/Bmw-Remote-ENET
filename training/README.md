# Phase 4 — Training Infrastructure

LoRA SFT + ORPO fine-tuning of **Qwen3.6-27B** into a local agentic
engineering assistant, on a single RTX PRO 6000 (96GB VRAM / 180GB RAM /
200GB disk). Stack: **Unsloth** for SFT, **TRL** for ORPO, per the Phase 1/4
decisions.

## Layout

```
training/
├── train_sft.py               # Stage 1: bf16 LoRA SFT (Unsloth)
├── merge_and_export.py        # LoRA merge -> bf16; sequential GGUF export
├── train_orpo.py              # Stage 2: ORPO preference pass (TRL)
├── orchestrate_training.ipynb # runs the whole sequence on the training box
├── configs/
│   ├── sft_plan_a.yaml        # the default run (bf16 base, 16K seq)
│   ├── sft_qlora_fallback.yaml# 4-bit base, 32K seq — OOM/long-context fallback
│   └── orpo_plan_a.yaml       # preference pass on the merged SFT model
├── trainkit/                  # framework-free core (fully unit-tested)
│   ├── config.py              # strict YAML config validation
│   ├── chat_format.py         # Hermes conversations -> ChatML segments + loss flags
│   ├── masking.py             # segment-wise tokenization, response-only labels
│   ├── packing.py             # FFD packing with per-example position reset
│   ├── data.py                # train.jsonl -> tokenized/packed HF dataset
│   ├── mask_check.py          # pre-flight mask verification (hard gate)
│   ├── storage.py             # disk guards + checkpoint retention (200GB box)
│   └── resume.py              # integrity-checked auto-resume
├── tests/                     # 60 tests, run anywhere (no GPU, no downloads)
└── docs/hyperparameters.md    # why every number is what it is
```

## The run, end to end

```bash
pip install --upgrade unsloth && pip install -r requirements.txt

# 0. Gates that cost nothing (run these on ANY machine first)
python -m pytest
python train_sft.py --config configs/sft_plan_a.yaml --dry-run

# 1. SFT (~20-25 GPU-hours; interruptible, auto-resumes on rerun)
python train_sft.py --config configs/sft_plan_a.yaml

# 2. Merge, verify, reclaim the ~55GB base download
python merge_and_export.py --config configs/sft_plan_a.yaml \
    --adapter runs/sft_plan_a/final_adapter \
    --output artifacts/merged_sft_bf16 --delete-hf-cache-after

# 3. ORPO on the merged model (needs data/preference_pairs.jsonl)
python train_orpo.py --config configs/orpo_plan_a.yaml

# 4. Final merge + GGUFs, one quant at a time, offloaded as they finish
python merge_and_export.py --config configs/orpo_plan_a.yaml \
    --adapter runs/orpo_plan_a/final_adapter \
    --output artifacts/final_bf16 \
    --gguf q4_k_m q6_k --llama-cpp-dir ~/llama.cpp \
    --offload-cmd 'rclone move {path} remote:models/'
```

Or open `orchestrate_training.ipynb`, which runs the same commands with
commentary and sanity checks between steps.

## Design decisions worth knowing

**The mask gate is a hard stop.** `train_sft.py` refuses to train until
`trainkit/mask_check.py` verifies, on a sample of the real corpus, that loss
lands only on assistant tokens, the stop token is being taught, and
segment-wise tokenization matches joint tokenization. A wrong mask is the
classic way agentic SFT fails *silently* (e.g. training the model to
hallucinate tool output); this makes it fail loudly at minute zero instead.

**`--dry-run` costs nothing and validates everything.** Config, data,
masking, packing stats and the step plan — with only a tokenizer download,
no GPU. Run it before renting the box.

**Drop, never truncate.** Overlong examples are dropped; truncating an
agentic trajectory mid-tool-call teaches malformed tool calls.

**Storage-awareness is enforced, not advised.** Every writing stage
(checkpoint, merge, each GGUF quant) guards free disk *before* writing;
GGUFs export sequentially with offload-then-delete; the HF base cache is
deleted only after the merge is verified on disk. Running out of disk
mid-write corrupts artifacts — on a 200GB box this is the failure to
engineer against.

**Resume is integrity-checked.** `resume: auto` only picks a checkpoint
with complete trainer state, optimizer state and weights; a crash during a
save can't poison the next run.

**LoRA coverage is reported, not assumed.** Qwen3.6's hybrid Gated-DeltaNet
blocks may expose projection names beyond the classic seven. At model load,
per-pattern match counts print, with a loud warning on any zero-match
pattern — extend `lora.target_modules` in the config if the hybrid layers
need it.

## Fallbacks

- **OOM at 16K bf16, or want 32K context:** use
  `configs/sft_qlora_fallback.yaml` (NF4 base, same everything else).
- **Unsloth breaks on Qwen3.6 at your version:** Axolotl is the documented
  fallback — it has a Qwen3-Next guide (identical hybrid architecture) and
  covers SFT/LoRA/ORPO in one YAML-driven pipeline. The Phase 3 data format
  and this repo's configs translate directly; the mask-check gate can be
  reused as-is since it only needs a tokenizer.

## Before the real run (checklist)

1. Verify the exact HF repo id for Qwen3.6-27B and set `model.name_or_path`.
2. Run the Phase 3 pipeline; confirm `data/train.jsonl` exists and passed
   its validation gate.
3. `python -m pytest` — 60 tests, seconds.
4. `python train_sft.py --config configs/sft_plan_a.yaml --dry-run` with the
   real tokenizer; read the mask-check output and packing stats.
5. Check the LoRA coverage report right after model load in the real run.
