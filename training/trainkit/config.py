"""Run configuration: YAML -> validated dataclasses.

Validation is strict and collects *all* problems before raising, so a config
with three mistakes reports three errors instead of failing one at a time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when a config file is structurally or semantically invalid."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # HF repo id or local path. Verify the exact repo id on hf.co before a
    # real run; it is deliberately not hardcoded anywhere else.
    name_or_path: str = "Qwen/Qwen3.6-27B"
    max_seq_len: int = 16384
    load_in_4bit: bool = False          # QLoRA fallback flips this on
    dtype: str = "bfloat16"
    # Qwen3.6 ships a unified multimodal checkpoint; we train text-only and
    # keep the vision tower frozen.
    freeze_vision_tower: bool = True
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.0
    # Standard attention + MLP projections. Qwen3.6's hybrid layers
    # (Gated DeltaNet) may expose additional linear module names; train_sft.py
    # prints per-pattern match counts and warns on zero-coverage patterns so
    # a silent mismatch cannot happen.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_rslora: bool = False


@dataclass
class DataConfig:
    train_path: str = "data/train.jsonl"       # Phase 3 output (Hermes format)
    eval_path: Optional[str] = None             # optional held-out split
    eval_fraction: float = 0.005                # used only when eval_path unset
    shuffle_seed: int = 3407
    # Examples longer than max_seq_len are dropped, not truncated: cutting an
    # agentic trajectory mid-tool-call teaches malformed behavior.
    overlong_policy: str = "drop"               # "drop" | "truncate_drop_if_broken"


@dataclass
class PackingConfig:
    mode: str = "ffd"                           # "ffd" | "none"
    reset_position_ids: bool = True


@dataclass
class MaskCheckConfig:
    enabled: bool = True
    sample_size: int = 200
    min_trainable_fraction: float = 0.02        # per-corpus bounds; agentic
    max_trainable_fraction: float = 0.85        # traces are env-output heavy
    max_tokenization_drift: float = 0.02        # joint-vs-segmented mismatch rate


@dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    num_epochs: float = 2.0
    per_device_batch_size: int = 1
    gradient_accumulation: int = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_8bit"
    seed: int = 3407
    logging_steps: int = 5
    gradient_checkpointing: bool = True


@dataclass
class CheckpointConfig:
    output_dir: str = "runs/sft_plan_a"
    save_steps: int = 100
    save_total_limit: int = 3
    resume: str = "auto"                        # "auto" | "never" | explicit path


@dataclass
class StorageConfig:
    min_free_gb_before_training: float = 30.0
    min_free_gb_per_checkpoint: float = 10.0


@dataclass
class OrpoConfig:
    """Only read by train_orpo.py; harmless in SFT configs."""
    beta: float = 0.1
    max_length: int = 8192
    max_prompt_length: int = 6144
    pairs_path: str = "data/preference_pairs.jsonl"
    merged_model_path: str = "artifacts/merged_sft_bf16"


@dataclass
class RunConfig:
    run_name: str = "sft_plan_a"
    stage: str = "sft"                          # "sft" | "orpo"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    mask_check: MaskCheckConfig = field(default_factory=MaskCheckConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    orpo: OrpoConfig = field(default_factory=OrpoConfig)


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------

_SECTION_TYPES = {
    "model": ModelConfig,
    "lora": LoraConfig,
    "data": DataConfig,
    "packing": PackingConfig,
    "mask_check": MaskCheckConfig,
    "optim": OptimConfig,
    "checkpointing": CheckpointConfig,
    "storage": StorageConfig,
    "orpo": OrpoConfig,
}


def _build_section(cls, raw: Dict[str, Any], section: str, errors: List[str]):
    known = {f.name for f in dataclasses.fields(cls)}
    for key in raw:
        if key not in known:
            errors.append(f"[{section}] unknown key: {key!r}")
    kwargs = {k: v for k, v in raw.items() if k in known}
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    errors: List[str] = []
    top_known = {"run_name", "stage"} | set(_SECTION_TYPES)
    for key in raw:
        if key not in top_known:
            errors.append(f"unknown top-level key: {key!r}")

    sections = {}
    for name, cls in _SECTION_TYPES.items():
        sub = raw.get(name, {})
        if not isinstance(sub, dict):
            errors.append(f"[{name}] must be a mapping")
            sub = {}
        sections[name] = _build_section(cls, sub, name, errors)

    cfg = RunConfig(
        run_name=raw.get("run_name", "unnamed_run"),
        stage=raw.get("stage", "sft"),
        **sections,
    )
    errors.extend(validate_config(cfg))
    if errors:
        raise ConfigError(
            f"invalid config {path}:\n  - " + "\n  - ".join(errors)
        )
    return cfg


def validate_config(cfg: RunConfig) -> List[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    e: List[str] = []
    if cfg.stage not in ("sft", "orpo"):
        e.append(f"stage must be 'sft' or 'orpo', got {cfg.stage!r}")

    m = cfg.model
    if m.max_seq_len < 512:
        e.append(f"model.max_seq_len={m.max_seq_len} is implausibly small")
    if m.dtype not in ("bfloat16", "float16"):
        e.append(f"model.dtype must be bfloat16 or float16, got {m.dtype!r}")

    lo = cfg.lora
    if lo.r <= 0:
        e.append(f"lora.r must be positive, got {lo.r}")
    if lo.alpha <= 0:
        e.append(f"lora.alpha must be positive, got {lo.alpha}")
    if not (0.0 <= lo.dropout < 1.0):
        e.append(f"lora.dropout must be in [0,1), got {lo.dropout}")
    if not lo.target_modules:
        e.append("lora.target_modules must not be empty")

    d = cfg.data
    if not (0.0 <= d.eval_fraction < 0.5):
        e.append(f"data.eval_fraction must be in [0, 0.5), got {d.eval_fraction}")
    if d.overlong_policy not in ("drop", "truncate_drop_if_broken"):
        e.append(f"data.overlong_policy invalid: {d.overlong_policy!r}")

    p = cfg.packing
    if p.mode not in ("ffd", "none"):
        e.append(f"packing.mode must be 'ffd' or 'none', got {p.mode!r}")

    mc = cfg.mask_check
    if not (0.0 <= mc.min_trainable_fraction < mc.max_trainable_fraction <= 1.0):
        e.append(
            "mask_check trainable-fraction bounds must satisfy "
            f"0 <= min < max <= 1, got [{mc.min_trainable_fraction}, "
            f"{mc.max_trainable_fraction}]"
        )

    o = cfg.optim
    if o.learning_rate <= 0:
        e.append(f"optim.learning_rate must be positive, got {o.learning_rate}")
    if o.learning_rate > 1e-2:
        e.append(
            f"optim.learning_rate={o.learning_rate} is >1e-2; almost certainly "
            "a typo for LoRA SFT"
        )
    if o.per_device_batch_size < 1:
        e.append("optim.per_device_batch_size must be >= 1")
    if o.gradient_accumulation < 1:
        e.append("optim.gradient_accumulation must be >= 1")
    if not (0.0 <= o.warmup_ratio < 0.5):
        e.append(f"optim.warmup_ratio must be in [0, 0.5), got {o.warmup_ratio}")
    if o.num_epochs <= 0:
        e.append(f"optim.num_epochs must be positive, got {o.num_epochs}")
    if o.scheduler not in ("cosine", "linear", "constant", "cosine_with_min_lr"):
        e.append(f"optim.scheduler invalid: {o.scheduler!r}")

    ck = cfg.checkpointing
    if ck.save_total_limit < 1:
        e.append("checkpointing.save_total_limit must be >= 1")
    if ck.save_steps < 1:
        e.append("checkpointing.save_steps must be >= 1")

    if cfg.stage == "orpo":
        if cfg.orpo.beta <= 0:
            e.append(f"orpo.beta must be positive, got {cfg.orpo.beta}")
        if cfg.orpo.max_prompt_length >= cfg.orpo.max_length:
            e.append("orpo.max_prompt_length must be < orpo.max_length")
    return e
