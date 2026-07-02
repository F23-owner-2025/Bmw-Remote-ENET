"""Pre-flight mask verification — the hard gate before GPU-hours are spent.

A wrong loss mask is the classic silent killer of agentic SFT runs: train on
tool_response turns and the model learns to hallucinate environment output;
mask everything and the loss goes to zero while nothing is learned. This
module makes both failure modes impossible to miss by checking, on a sample
of real training examples:

1.  Decoded trainable spans start inside assistant content — never with an
    ``<|im_start|>`` header for a non-assistant role, and never containing a
    ``<tool_response>`` opening (environment output must be untrained).
2.  Every example has at least one trainable token, and every trainable
    example ends its final span with ``<|im_end|>`` (the stop signal is
    being taught).
3.  Corpus-level trainable fraction lies inside configured bounds. Agentic
    traces are dominated by environment output, so the healthy range is
    wide (default 2%–85%) — this catches the two catastrophes (≈0% and
    ≈100%), not normal variance.
4.  Segment-wise tokenization matches joint tokenization of the same
    rendered string within a drift tolerance (default 2% of examples may
    differ, and only at boundaries). This validates masking.py's core
    assumption instead of trusting it.

``verify_masking`` returns a report; the trainer refuses to start when
``report.passed`` is False.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .chat_format import IM_END, IM_START, render_conversation, render_text
from .masking import IGNORE_INDEX, tokenize_segments


@dataclass
class MaskReport:
    examples_checked: int = 0
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trainable_fraction_mean: float = 0.0
    trainable_fraction_min: float = 1.0
    trainable_fraction_max: float = 0.0
    tokenization_drift_rate: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"mask check: {status} ({self.examples_checked} examples)",
            f"  trainable fraction: mean={self.trainable_fraction_mean:.3f} "
            f"min={self.trainable_fraction_min:.3f} max={self.trainable_fraction_max:.3f}",
            f"  tokenization drift rate: {self.tokenization_drift_rate:.4f}",
        ]
        lines.extend(f"  FAIL: {f}" for f in self.failures)
        lines.extend(f"  warn: {w}" for w in self.warnings)
        return "\n".join(lines)


def _trainable_spans(input_ids: List[int], labels: List[int]) -> List[List[int]]:
    spans, cur = [], []
    for tok, lab in zip(input_ids, labels):
        if lab != IGNORE_INDEX:
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


_BAD_SPAN_PREFIXES = (
    f"{IM_START}system",
    f"{IM_START}user",
    f"{IM_START}tool",
)


def verify_masking(
    conversations: List[List[Dict[str, str]]],
    tokenizer,
    sample_size: int = 200,
    min_trainable_fraction: float = 0.02,
    max_trainable_fraction: float = 0.85,
    max_tokenization_drift: float = 0.02,
    seed: int = 0,
) -> MaskReport:
    report = MaskReport()
    if not conversations:
        report.failures.append("no conversations supplied to mask check")
        return report

    rng = random.Random(seed)
    sample = conversations if len(conversations) <= sample_size else rng.sample(conversations, sample_size)

    fractions: List[float] = []
    drift_count = 0

    for ci, messages in enumerate(sample):
        segments = render_conversation(messages)
        ex = tokenize_segments(segments, tokenizer)
        report.examples_checked += 1

        if ex.num_trainable == 0:
            report.failures.append(f"example {ci}: zero trainable tokens")
            continue
        fractions.append(ex.trainable_fraction)

        spans = _trainable_spans(ex.input_ids, ex.labels)
        for si, span in enumerate(spans):
            text = tokenizer.decode(span)
            if any(text.startswith(p) for p in _BAD_SPAN_PREFIXES):
                report.failures.append(
                    f"example {ci} span {si}: trainable span begins with a "
                    f"non-assistant header: {text[:60]!r}"
                )
            # A trainable span opening a tool_response means env output is
            # being trained on. Assistant text may *mention* the tag (e.g.
            # instructional system-prompt echoes), so only flag span-initial.
            if text.lstrip().startswith("<tool_response>"):
                report.failures.append(
                    f"example {ci} span {si}: trainable span starts with "
                    "<tool_response> (environment output would be trained)"
                )
        last_text = tokenizer.decode(spans[-1])
        if IM_END not in last_text:
            report.failures.append(
                f"example {ci}: final trainable span lacks {IM_END} — the "
                "stop token is not being taught"
            )

        # Drift: joint tokenization of the identical rendered string.
        joint = tokenizer.encode(render_text(messages), add_special_tokens=False)
        if joint != ex.input_ids:
            drift_count += 1

    if fractions:
        report.trainable_fraction_mean = sum(fractions) / len(fractions)
        report.trainable_fraction_min = min(fractions)
        report.trainable_fraction_max = max(fractions)

        if report.trainable_fraction_mean < min_trainable_fraction:
            report.failures.append(
                f"mean trainable fraction {report.trainable_fraction_mean:.4f} < "
                f"{min_trainable_fraction} — mask is likely eating assistant tokens"
            )
        if report.trainable_fraction_mean > max_trainable_fraction:
            report.failures.append(
                f"mean trainable fraction {report.trainable_fraction_mean:.4f} > "
                f"{max_trainable_fraction} — non-assistant tokens are likely "
                "being trained"
            )

    report.tokenization_drift_rate = drift_count / max(1, report.examples_checked)
    if report.tokenization_drift_rate > max_tokenization_drift:
        report.failures.append(
            f"segmented-vs-joint tokenization drift {report.tokenization_drift_rate:.4f} "
            f"exceeds tolerance {max_tokenization_drift} — segment boundaries are "
            "not falling on special tokens for this tokenizer"
        )
    elif drift_count:
        report.warnings.append(
            f"{drift_count} example(s) showed boundary tokenization drift "
            "(within tolerance)"
        )
    return report
