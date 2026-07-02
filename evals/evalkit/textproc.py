"""Response text processing: thinking-block stripping, answer extraction,
code extraction, tool-call parsing.

Qwen3.6 may emit <think>...</think> reasoning blocks; graders must judge the
visible answer, so strip_think() runs on every response before grading
(transcripts keep the original).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER\s*[:\-]\s*(.+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")
_FRACTION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
_LETTER_RE = re.compile(r"\b([A-D])\b")


def strip_think(text: str) -> str:
    out = _THINK_RE.sub("", text)
    # An unclosed think block (truncated generation) hides everything after
    # it; drop from the opening tag onward.
    if "<think>" in out:
        out = out.split("<think>", 1)[0]
    return out.strip()


# ------------------------------------------------------------- tool calls

def extract_tool_calls(text: str) -> Tuple[List[Dict], List[str]]:
    """Return (parsed_calls, errors). A call is {"name": str, "arguments": dict}."""
    calls, errors = [], []
    for i, raw in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as err:
            errors.append(f"tool_call {i}: invalid JSON ({err})")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            errors.append(f"tool_call {i}: missing string 'name'")
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            errors.append(f"tool_call {i}: 'arguments' must be an object")
            continue
        calls.append({"name": obj["name"], "arguments": args})
    # Opening tag without a closing tag is itself a formatting failure.
    if text.count("<tool_call>") != text.count("</tool_call>"):
        errors.append("unbalanced <tool_call> tags")
    return calls, errors


# ------------------------------------------------------------- code

def extract_python_code(text: str, entry_point: Optional[str] = None) -> str:
    """Best code block from a response.

    Preference order: last fenced block containing the entry point, else the
    last fenced block, else the raw text (models sometimes reply with bare
    code).
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return text.strip()
    if entry_point:
        with_entry = [b for b in blocks if entry_point in b]
        if with_entry:
            return with_entry[-1].strip()
    return blocks[-1].strip()


# ------------------------------------------------------------- answers

def parse_number(text: str) -> Optional[float]:
    """First number in `text`, tolerating commas, sci notation, fractions."""
    frac = _FRACTION_RE.search(text)
    plain = _NUMBER_RE.search(text)
    # Prefer the fraction interpretation only when it starts where the
    # plain number starts (e.g. "1/9" vs "1.5 m/s").
    if frac and plain and frac.start() == plain.start():
        denom = float(frac.group(2))
        if denom != 0:
            return float(frac.group(1)) / denom
    if plain:
        return float(plain.group(0).replace(",", ""))
    return None


def extract_final_answer_text(text: str) -> Optional[str]:
    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def extract_numeric_answer(text: str) -> Optional[float]:
    seg = extract_final_answer_text(text)
    if seg is not None:
        num = parse_number(seg)
        if num is not None:
            return num
    # Fallback: the last number anywhere in the response.
    all_nums = _NUMBER_RE.findall(text)
    if all_nums:
        return float(all_nums[-1].replace(",", ""))
    return None


def extract_mcq_answer(text: str) -> Optional[str]:
    seg = extract_final_answer_text(text)
    if seg:
        m = _LETTER_RE.search(seg.upper())
        if m:
            return m.group(1)
    letters = _LETTER_RE.findall(text.upper())
    return letters[-1] if letters else None
