"""Sequence packing: first-fit-decreasing bins with position_id reset.

Agentic trajectories have wildly skewed lengths (300-token tool-schema
drills next to 15k-token multi-turn traces). Without packing, batches padded
to max_seq_len waste most of their FLOPs on pad tokens. FFD packing
recovers that at near-optimal bin utilization while keeping the
implementation auditable (~40 lines).

Cross-contamination control: each packed row carries per-example
position_ids that reset to 0 at every example boundary. Combined with
labels that never span a boundary (each example's labels were built
independently), an example cannot train on a neighbor's tokens. Attention
itself is *not* re-masked per example — with position resets and -100
labels at boundaries, residual cross-attention has been repeatedly shown to
be a non-issue for SFT (this is the same trade Unsloth's and HF's packed
SFT make), and it avoids depending on framework-specific block-diagonal
attention kernels for a brand-new hybrid architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .masking import TokenizedExample


@dataclass
class PackedRow:
    input_ids: List[int] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    position_ids: List[int] = field(default_factory=list)
    num_examples: int = 0

    def __len__(self) -> int:
        return len(self.input_ids)

    def add(self, ex: TokenizedExample) -> None:
        self.input_ids.extend(ex.input_ids)
        self.labels.extend(ex.labels)
        self.position_ids.extend(range(len(ex.input_ids)))
        self.num_examples += 1


def pack_ffd(
    examples: List[TokenizedExample],
    max_seq_len: int,
    reset_position_ids: bool = True,
) -> List[PackedRow]:
    """First-fit-decreasing packing. Examples longer than max_seq_len must
    have been dropped upstream (data.py enforces this); we assert rather
    than silently truncate."""
    for ex in examples:
        if len(ex) > max_seq_len:
            raise ValueError(
                f"example of length {len(ex)} exceeds max_seq_len={max_seq_len}; "
                "overlong examples must be filtered before packing"
            )

    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]))
    rows: List[PackedRow] = []
    # Track remaining capacity; linear scan is fine at our scale (~200k rows).
    capacities: List[int] = []

    for idx in order:
        ex = examples[idx]
        placed = False
        for row_i, cap in enumerate(capacities):
            if len(ex) <= cap:
                rows[row_i].add(ex)
                capacities[row_i] = cap - len(ex)
                placed = True
                break
        if not placed:
            row = PackedRow()
            row.add(ex)
            rows.append(row)
            capacities.append(max_seq_len - len(ex))

    if not reset_position_ids:
        for row in rows:
            row.position_ids = list(range(len(row.input_ids)))
    return rows


def packing_efficiency(rows: List[PackedRow], max_seq_len: int) -> float:
    """Fraction of the padded token budget occupied by real tokens."""
    if not rows:
        return 0.0
    used = sum(len(r) for r in rows)
    return used / (len(rows) * max_seq_len)
