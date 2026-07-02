"""Segment-wise tokenization producing response-only labels.

Each conversation is rendered by chat_format.render_conversation() into
(text, train) segments. We tokenize every segment independently (with
``add_special_tokens=False`` so the tokenizer never injects BOS/EOS mid-
conversation), concatenate the ids, and label tokens -100 wherever the
segment is not trainable.

Why segment-wise instead of "tokenize the whole string, then locate the
assistant spans": locating spans after the fact relies on string-offset
bookkeeping that breaks subtly when a tokenizer merges characters across a
boundary. Tokenizing per segment makes the label assignment trivially
correct *by construction* — at the cost of possible drift where a merge
would have happened across a boundary. ChatML is designed so that segment
boundaries always fall on special tokens (<|im_start|>, <|im_end|>\\n),
where no merging occurs; mask_check.py still measures actual drift against
joint tokenization and hard-fails the run if it exceeds the configured
tolerance, so this assumption is verified rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .chat_format import Segment, render_conversation

IGNORE_INDEX = -100


@dataclass
class TokenizedExample:
    input_ids: List[int]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_trainable(self) -> int:
        return sum(1 for t in self.labels if t != IGNORE_INDEX)

    @property
    def trainable_fraction(self) -> float:
        return self.num_trainable / len(self.labels) if self.labels else 0.0


def tokenize_segments(segments: List[Segment], tokenizer) -> TokenizedExample:
    input_ids: List[int] = []
    labels: List[int] = []
    for seg in segments:
        ids = tokenizer.encode(seg.text, add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend(ids if seg.train else [IGNORE_INDEX] * len(ids))
    return TokenizedExample(input_ids=input_ids, labels=labels)


def tokenize_conversation(messages: List[Dict[str, str]], tokenizer) -> TokenizedExample:
    return tokenize_segments(render_conversation(messages), tokenizer)
