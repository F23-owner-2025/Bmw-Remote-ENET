"""Dataset build: Phase 3 train.jsonl -> tokenized (optionally packed) rows.

The input contract is the Phase 3 output: one JSON object per line with a
``messages`` list in Hermes/ChatML-compatible role/content form, already
validated by the Phase 3 gate. We still treat every row defensively —
malformed rows are counted and skipped, never allowed to crash a multi-hour
run or silently truncate into garbage.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .chat_format import ChatFormatError
from .masking import TokenizedExample, tokenize_conversation
from .packing import PackedRow, pack_ffd, packing_efficiency


@dataclass
class BuildStats:
    rows_read: int = 0
    rows_malformed: int = 0
    rows_render_failed: int = 0
    rows_overlong_dropped: int = 0
    rows_no_trainable_tokens: int = 0
    examples_kept: int = 0
    total_tokens: int = 0
    trainable_tokens: int = 0
    packed_rows: int = 0
    packing_efficiency: float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rows read                 : {self.rows_read}",
            f"  malformed JSON/schema   : {self.rows_malformed}",
            f"  render failures         : {self.rows_render_failed}",
            f"  overlong dropped        : {self.rows_overlong_dropped}",
            f"  zero-trainable dropped  : {self.rows_no_trainable_tokens}",
            f"examples kept             : {self.examples_kept}",
            f"total tokens              : {self.total_tokens}",
            f"trainable tokens          : {self.trainable_tokens}"
            f" ({100.0 * self.trainable_tokens / max(1, self.total_tokens):.1f}%)",
            f"packed rows               : {self.packed_rows}",
            f"packing efficiency        : {self.packing_efficiency:.3f}",
        ]
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def iter_jsonl(path: str | Path) -> Iterator[Tuple[int, Optional[dict]]]:
    """Yield (line_number, parsed_or_None) for each non-empty line."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, None


def load_and_tokenize(
    path: str | Path,
    tokenizer,
    max_seq_len: int,
    stats: Optional[BuildStats] = None,
) -> Tuple[List[TokenizedExample], BuildStats]:
    stats = stats or BuildStats()
    examples: List[TokenizedExample] = []

    for lineno, obj in iter_jsonl(path):
        stats.rows_read += 1
        if obj is None or not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
            stats.rows_malformed += 1
            continue
        try:
            ex = tokenize_conversation(obj["messages"], tokenizer)
        except ChatFormatError:
            stats.rows_render_failed += 1
            continue

        if len(ex) > max_seq_len:
            # Drop, never truncate: a trajectory cut mid-tool-call is a
            # lesson in producing malformed tool calls.
            stats.rows_overlong_dropped += 1
            continue
        if ex.num_trainable == 0:
            stats.rows_no_trainable_tokens += 1
            continue

        examples.append(ex)
        stats.examples_kept += 1
        stats.total_tokens += len(ex)
        stats.trainable_tokens += ex.num_trainable

    if stats.rows_read and stats.rows_overlong_dropped / stats.rows_read > 0.10:
        stats.notes.append(
            f"{stats.rows_overlong_dropped}/{stats.rows_read} rows dropped as "
            f"overlong (> {max_seq_len} tokens) — consider raising max_seq_len "
            "or re-checking the Phase 3 length budget"
        )
    return examples, stats


def split_train_eval(
    examples: List[TokenizedExample],
    eval_fraction: float,
    seed: int,
) -> Tuple[List[TokenizedExample], List[TokenizedExample]]:
    if eval_fraction <= 0 or len(examples) < 2:
        return examples, []
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(examples) * eval_fraction))
    eval_set = {i for i in idx[:n_eval]}
    train = [ex for i, ex in enumerate(examples) if i not in eval_set]
    evals = [ex for i, ex in enumerate(examples) if i in eval_set]
    return train, evals


def build_packed_dataset(
    examples: List[TokenizedExample],
    max_seq_len: int,
    mode: str = "ffd",
    reset_position_ids: bool = True,
    shuffle_seed: int = 3407,
    stats: Optional[BuildStats] = None,
) -> Tuple[List[PackedRow], BuildStats]:
    stats = stats or BuildStats()
    if mode == "none":
        rows = []
        for ex in examples:
            row = PackedRow()
            row.add(ex)
            rows.append(row)
    elif mode == "ffd":
        rows = pack_ffd(examples, max_seq_len, reset_position_ids=reset_position_ids)
    else:
        raise ValueError(f"unknown packing mode: {mode!r}")

    # FFD emits rows sorted by content length; shuffle so early training
    # steps aren't systematically long-trajectory-heavy.
    rng = random.Random(shuffle_seed)
    rng.shuffle(rows)

    stats.packed_rows = len(rows)
    stats.packing_efficiency = packing_efficiency(rows, max_seq_len)
    return rows, stats


def rows_to_hf_columns(rows: List[PackedRow], pad_token_id: int, max_seq_len: int) -> Dict[str, list]:
    """Convert packed rows to fixed-length columns for a datasets.Dataset.

    Right-pads to max_seq_len; pad positions get attention_mask=0 and
    labels=-100, and position_ids continue monotonically (their values are
    irrelevant under attention_mask=0, but monotone padding avoids kernel
    edge cases with repeated zeros).
    """
    out: Dict[str, list] = {"input_ids": [], "labels": [], "position_ids": [], "attention_mask": []}
    for row in rows:
        n = len(row.input_ids)
        pad = max_seq_len - n
        if pad < 0:
            raise ValueError(f"packed row of length {n} exceeds max_seq_len={max_seq_len}")
        out["input_ids"].append(row.input_ids + [pad_token_id] * pad)
        out["labels"].append(row.labels + [-100] * pad)
        last = row.position_ids[-1] if row.position_ids else -1
        out["position_ids"].append(row.position_ids + list(range(last + 1, last + 1 + pad)))
        out["attention_mask"].append([1] * n + [0] * pad)
    return out
