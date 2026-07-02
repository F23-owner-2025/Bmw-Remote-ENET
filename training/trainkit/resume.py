"""Checkpoint resume: discovery plus integrity checking.

``resume: auto`` must never resume from a checkpoint that was truncated by
a crash or a full disk — HF will either fail cryptically or, worse, load a
partial optimizer state. So auto-resume only picks a checkpoint that passes
a completeness check, and falls back to older complete ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .storage import list_checkpoints

# A usable trainer checkpoint needs trainer bookkeeping, optimizer state and
# model weights. Weight files vary by setup (LoRA adapters vs full), hence a
# set of acceptable alternatives.
_REQUIRED = ("trainer_state.json",)
_OPTIMIZER_ALTERNATIVES = ("optimizer.pt", "optimizer.bin")
_WEIGHT_ALTERNATIVES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
)


def checkpoint_integrity(ckpt: str | Path) -> Tuple[bool, List[str]]:
    ckpt = Path(ckpt)
    problems: List[str] = []
    if not ckpt.is_dir():
        return False, [f"not a directory: {ckpt}"]

    for name in _REQUIRED:
        f = ckpt / name
        if not f.is_file() or f.stat().st_size == 0:
            problems.append(f"missing or empty: {name}")

    state = ckpt / "trainer_state.json"
    if state.is_file():
        try:
            with open(state, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "global_step" not in data:
                problems.append("trainer_state.json lacks global_step")
        except (json.JSONDecodeError, UnicodeDecodeError):
            problems.append("trainer_state.json is corrupt (interrupted write)")

    if not any((ckpt / n).is_file() and (ckpt / n).stat().st_size > 0
               for n in _OPTIMIZER_ALTERNATIVES):
        problems.append("no optimizer state file found")
    if not any((ckpt / n).is_file() and (ckpt / n).stat().st_size > 0
               for n in _WEIGHT_ALTERNATIVES):
        problems.append("no model/adapter weight file found")

    return not problems, problems


def find_resume_checkpoint(output_dir: str | Path, policy: str) -> Optional[Path]:
    """Resolve the checkpoint to resume from under the configured policy.

    "never"       -> None (fresh start)
    "auto"        -> newest checkpoint passing integrity check, else None
    explicit path -> that checkpoint, raising if it fails integrity
    """
    if policy == "never":
        return None

    if policy == "auto":
        for ckpt in reversed(list_checkpoints(output_dir)):
            ok, problems = checkpoint_integrity(ckpt)
            if ok:
                return ckpt
            print(f"[resume] skipping incomplete checkpoint {ckpt.name}: "
                  + "; ".join(problems))
        return None

    # Explicit path: user asked for this exact checkpoint, so a problem is
    # an error rather than a silent fresh start.
    ckpt = Path(policy)
    ok, problems = checkpoint_integrity(ckpt)
    if not ok:
        raise RuntimeError(
            f"requested resume checkpoint {ckpt} failed integrity check: "
            + "; ".join(problems)
        )
    return ckpt
