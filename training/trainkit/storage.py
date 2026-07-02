"""Disk-space guards for the 200GB-constrained training box.

The Phase 1 budget showed the full artifact chain (bf16 base + datasets +
checkpoints + merged model + GGUF exports) brushes right up against 200GB.
Running out of disk mid-checkpoint or mid-merge corrupts the artifact being
written, so every writing stage calls a guard *before* it starts writing,
and checkpoint retention is enforced as a backstop independent of the
trainer's own save_total_limit.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class InsufficientDiskError(RuntimeError):
    pass


@dataclass
class DiskStatus:
    path: str
    total_gb: float
    used_gb: float
    free_gb: float


def disk_status(path: str | Path = ".") -> DiskStatus:
    p = Path(path)
    probe = p if p.exists() else p.parent
    usage = shutil.disk_usage(probe)
    gib = 1024 ** 3
    return DiskStatus(
        path=str(p),
        total_gb=usage.total / gib,
        used_gb=usage.used / gib,
        free_gb=usage.free / gib,
    )


def require_free_gb(path: str | Path, needed_gb: float, purpose: str) -> DiskStatus:
    status = disk_status(path)
    if status.free_gb < needed_gb:
        raise InsufficientDiskError(
            f"{purpose}: need {needed_gb:.1f} GiB free at {status.path}, "
            f"have {status.free_gb:.1f} GiB. Free space (old checkpoints, "
            "already-offloaded GGUFs, the HF hub cache) and retry."
        )
    return status


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def list_checkpoints(output_dir: str | Path) -> List[Path]:
    """Trainer-style checkpoint dirs sorted by step ascending."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    found = []
    for child in out.iterdir():
        m = _CKPT_RE.match(child.name)
        if child.is_dir() and m:
            found.append((int(m.group(1)), child))
    return [p for _, p in sorted(found)]


def prune_checkpoints(output_dir: str | Path, keep: int) -> List[Path]:
    """Delete all but the newest ``keep`` checkpoints. Returns deleted paths.

    Backstop for HF's save_total_limit (which can be bypassed when a run is
    restarted with different settings, leaving stale checkpoints behind).
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    ckpts = list_checkpoints(output_dir)
    doomed = ckpts[:-keep] if len(ckpts) > keep else []
    for path in doomed:
        shutil.rmtree(path)
    return doomed


def estimate_dir_gb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    ckpts = list_checkpoints(output_dir)
    return ckpts[-1] if ckpts else None
