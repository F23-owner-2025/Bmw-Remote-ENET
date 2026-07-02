"""Shared test fixtures.

FakeTokenizer is a deterministic word-level tokenizer that treats ChatML
special tokens atomically, so joint and segmented tokenization agree exactly
when segment boundaries fall on special-token edges — the same property the
real Qwen tokenizer has, which is what mask_check verifies.

DriftyTokenizer deliberately violates that property (it merges a trailing
newline with a following <|im_start|> when encoding jointly) so the drift
detector can be shown to fire.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPECIALS = ["<|im_start|>", "<|im_end|>"]
_SPLIT = re.compile(
    "(" + "|".join(re.escape(s) for s in SPECIALS) + r"|\s+|[^\s<]+|<)"
)


class FakeTokenizer:
    def __init__(self):
        self._tok_to_id = {}
        self._id_to_tok = {}
        self.pad_token = "<|pad|>"
        self.pad_token_id = self._intern(self.pad_token)
        self.eos_token = "<|im_end|>"

    def _intern(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            idx = len(self._tok_to_id)
            self._tok_to_id[tok] = idx
            self._id_to_tok[idx] = tok
        return self._tok_to_id[tok]

    def _pieces(self, text: str) -> List[str]:
        return [p for p in _SPLIT.findall(text) if p]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [self._intern(p) for p in self._pieces(text)]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._id_to_tok[i] for i in ids)


class DriftyTokenizer(FakeTokenizer):
    """Merges '\\n' + '<|im_start|>' into one token — but only when both are
    present in the same encode() call, i.e. only under joint tokenization."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        pieces = self._pieces(text)
        merged: List[str] = []
        i = 0
        while i < len(pieces):
            if (i + 1 < len(pieces) and pieces[i].endswith("\n")
                    and pieces[i + 1] == "<|im_start|>"):
                merged.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        return [self._intern(p) for p in merged]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def drifty_tokenizer():
    return DriftyTokenizer()


TOOL_SYSTEM = (
    "You are a senior engineering assistant. You have access to the following "
    "tools, described within <tools></tools> XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name": "run_commands", "parameters": '
    '{"type": "object", "properties": {"commands": {"type": "array"}}}}}\n'
    "</tools>\nFor each call return a JSON object inside <tool_call></tool_call> tags."
)


def make_agentic_conversation(task: str = "list files") -> list:
    return [
        {"role": "system", "content": TOOL_SYSTEM},
        {"role": "user", "content": f"Please {task} in the project directory."},
        {
            "role": "assistant",
            "content": (
                "I will inspect the directory first.\n<tool_call>\n"
                '{"name": "run_commands", "arguments": {"commands": ["ls -la"]}}\n'
                "</tool_call>"
            ),
        },
        {"role": "tool", "content": "total 12\ndrwxr-xr-x src\n-rw-r--r-- main.py"},
        {
            "role": "assistant",
            "content": "The directory contains src/ and main.py. Task complete.",
        },
    ]


def make_chat_conversation(question: str = "What is torque?") -> list:
    return [
        {"role": "system", "content": "You are a helpful engineering tutor."},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "Torque is the rotational analogue of force, tau equals r cross F.",
        },
    ]
