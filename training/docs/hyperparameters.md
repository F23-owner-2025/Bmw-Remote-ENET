# Hyperparameter Justification — Plan A

Every value in `configs/sft_plan_a.yaml` and `configs/orpo_plan_a.yaml`,
justified against the actual run: Qwen3.6-27B dense (hybrid Gated-DeltaNet +
Gated Attention, unified multimodal checkpoint), ~175M target tokens of
agentic-heavy SFT data, one RTX PRO 6000 (96GB VRAM), 180GB RAM, 200GB disk.

## Stage 1 — SFT LoRA

### Precision: bf16 base, LoRA in bf16 (`dtype: bfloat16`, `load_in_4bit: false`)

27B bf16 weights ≈ 54GB. Qwen3.5-27B (same size class, same hybrid
architecture) trains as bf16 LoRA in ~56GB under Unsloth, leaving ~40GB for
activations, gradients, and optimizer state on a 96GB card. bf16 over fp16
because the RTX PRO 6000 (Blackwell) has native bf16 and bf16's fp32-sized
exponent range eliminates the loss-scaling machinery fp16 needs — one fewer
failure mode on a long run. QLoRA is the *fallback*, not the default: the
~1–2% quality cost of a 4-bit base is a bad trade when bf16 fits.

### LoRA: r=64, α=128, dropout 0, all linear projections

- **r=64** is the sweet spot measured repeatedly for behavioral fine-tunes of
  20–35B models: r=16–32 underfits multi-skill mixes (agentic + STEM +
  code), r=128+ doubles adapter memory for consistently marginal gains. At
  r=64 the adapter is ~0.5–1% of model params — enough capacity to encode
  new *behavior* (tool-call discipline, plan-then-act) without enough to
  catastrophically overwrite base knowledge.
- **α=2r (128)** keeps the effective scaling α/r = 2 constant — the default
  that transfers across rank choices; changing it and LR together makes runs
  incomparable.
- **dropout 0**: LoRA dropout mainly protects tiny datasets. At ~175M tokens
  over ≤2 epochs, overfitting pressure is low, and dropout measurably slows
  convergence per step (and disables some of Unsloth's fast paths).
- **All linear projections** (q/k/v/o + gate/up/down): MLP-only or
  attention-only LoRA loses several points on reasoning-heavy evals.
  Qwen3.6's DeltaNet blocks may expose *additional* projection names —
  `train_sft.py` prints per-pattern match counts at load; a zero-match
  pattern is a loud warning, and the fix is extending the list in the
  config, not code.

### Sequence length: 16384

Phase 3's Plan A config targets trajectories that fit 16K; the dry run
reports the overlong-drop rate on the real corpus (gate: <10%). 16K covers
the p99 of OpenThoughts-Agent-style trajectories while keeping activation
memory inside the bf16 budget. If you need 32K (very long traces), that is
what the QLoRA fallback config is for — the 36GB freed by NF4 weights pays
for the doubled activations.

### Batch: per-device 1 × grad-accum 8 = ~131K tokens/optimizer step

One packed 16K row is the largest micro-batch that reliably fits beside
the bf16 base. Eight accumulation steps give ~131K tokens per update —
inside the 100K–250K band that large-scale SFT recipes converge on: small
enough for ~1,300+ optimizer steps over the corpus (schedulers behave
poorly under a few hundred steps), large enough for stable gradients on
heterogeneous packed data. FFD packing at >90% efficiency is what makes
"1 row" actually mean ~15K real tokens rather than padding.

### Learning rate: 1e-4, cosine to 10% floor, 3% warmup

- **1e-4** is the canonical LoRA-SFT rate for this α/r; full-FT intuitions
  (1e-5–2e-5) do not apply because only adapters train. Higher (2e-4+) risks
  degrading the base's already-excellent code ability — our data is mostly
  *behavioral*, so we want firm but not aggressive updates.
- **Cosine with `min_lr_ratio: 0.1`** (decay to 1e-5, not 0): the tail of a
  to-zero cosine wastes the final ~15% of steps making null updates; a 10%
  floor keeps late examples contributing. This is why the config maps
  `cosine` → HF's `cosine_with_min_lr`.
- **3% warmup** (~40 steps): adapters start at zero (LoRA-B init), so early
  gradients are well-conditioned; long warmups just delay training. 3% is
  enough to let the 8-bit Adam moments settle.

### Epochs: 2

One epoch under-exploits a carefully deduplicated corpus (Phase 3 removed
the near-dupes that make multi-epoch training risky); three epochs on SFT
data reliably shows memorization artifacts (verbatim trajectory replay).
Two is the standard compromise. The eval split (0.5%) exists precisely to
watch for the epoch-2 eval-loss upturn; if it appears, stop at the best
checkpoint — that is also why checkpoints are frequent.

### Optimizer: `adamw_8bit`, weight decay 0.01, grad clip 1.0

8-bit Adam quantizes the two moment tensors, saving ~6× optimizer memory
versus fp32 Adam with no measurable quality delta at LoRA scale — on a
single-GPU budget this is free VRAM. Weight decay 0.01 on adapter weights
is mild regularization with no downside at this scale. Clip at 1.0 caps
the occasional pathological batch (agentic data has weird outliers —
enormous JSON blobs, binary-ish tool output that survived filtering).

### Gradient checkpointing: on (Unsloth's variant)

At 16K sequences, activations dominate memory. Checkpointing trades ~25–30%
step time for the ~big activation savings that let 16K fit at all beside a
54GB base. Unsloth's implementation additionally offloads to system RAM
asynchronously — and this box has 180GB of it. This is not optional at this
config; it is what makes the memory math close.

### Attention kernels: whatever Unsloth selects

On the classic-transformer share of layers Unsloth uses Flash-Attention-
class fused kernels automatically; the Gated-DeltaNet layers use their own
fused linear-attention kernels. We deliberately do not force
`attn_implementation` flags on a hybrid architecture — the framework's
tested default is the safe choice.

### Checkpointing: every 100 steps, keep 3, auto-resume

100 steps ≈ 45–60 min: an interruption costs at most ~1 hour of compute.
Keep-3 bounds disk (~3× adapter+optimizer, ~2GB total — checkpoints are
adapter-sized, not model-sized, which is why we can afford frequency).
`resume: auto` only ever selects a checkpoint that passes the integrity
check in `trainkit/resume.py`, so a crash during a save can never poison
the resumed run.

## Stage 2 — ORPO

### On the merged model, fresh r=32 LoRA, 4-bit base

Merging first, then preference-tuning with a fresh adapter keeps the stages
decoupled: if ORPO degrades something, the SFT merge is untouched and the
pass is re-runnable in isolation. r=32 because preference nudging adjusts
*relative* behavior, needing less capacity than teaching new skills. The
4-bit base is acceptable here (unlike SFT) because ORPO's forward passes
double (chosen + rejected) and the pass is short — precision sensitivity is
much lower for a 1-epoch, 5–15k-pair run.

### LR 5e-6, β=0.1, 1 epoch, effective batch 16 pairs

Preference objectives are sharp: the LR that is right for SFT (1e-4) will
collapse outputs to sycophantic short answers under a preference loss.
5e-6–1e-5 is the established band; we take the low end because the pairs
are synthetic. β=0.1 is ORPO's standard odds-ratio weight — raise toward
0.5 only if reward margins stay flat in the logs. One epoch: preference
passes visibly overfit on the second epoch at this data size. 10% warmup
because the absolute step count is small (~300–900 steps); a 3% warmup
would be a dozen steps, too few for Adam moments.

### max_length 8192 / max_prompt_length 6144

Preference pairs target *decision points* (plan-vs-dive-in, clarify-vs-
assume), which live early in trajectories; they do not need 16K. Halving
the length quarters attention-activation cost per pair, which is what lets
per-device batch 2 fit even with doubled forwards.

## Budget check

~175M tokens × 2 epochs ≈ 350M token-passes at ~2,600 optimizer steps.
On this card with Unsloth's kernels, bf16 LoRA at 16K packed throughput
lands around 3.5–5K tokens/s → **~20–27 GPU-hours** for SFT, matching the
Phase 1 estimate. ORPO adds ~2–4 hours. Merge and GGUF export are CPU-bound
(~1–2 hours total).
