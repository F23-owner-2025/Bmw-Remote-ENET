"""Render Hermes-format conversations into ChatML segments with loss flags.

Phase 3 emits examples as ``{"messages": [{"role": ..., "content": ...}, ...]}``
where roles are ``system`` / ``user`` / ``assistant`` / ``tool``. Assistant
turns may embed ``<tool_call>...</tool_call>`` blocks; tool turns carry the
environment output (already wrapped in ``<tool_response>`` tags by the Phase 3
converter, or raw — we normalize either way).

We render to Qwen's ChatML dialect ourselves, as an ordered list of
(text, train) segments, instead of calling ``tokenizer.apply_chat_template``
and then heuristically re-discovering assistant spans. Building the template
string and the loss mask from the same source of truth is what makes
response-only masking verifiable: masking.py tokenizes exactly these segments,
and mask_check.py confirms the concatenation matches joint tokenization.

Loss policy:
    * assistant content + its closing ``<|im_end|>\\n``  -> train
    * everything else (system, user, tool turns, and every ``<|im_start|>``
      header including the assistant's own)               -> no train

Training on the closing ``<|im_end|>`` is required: it is how the model
learns to stop. Not training on the assistant header is standard — the
header is deterministically supplied by the inference template, so there is
no benefit to spending loss on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

VALID_ROLES = ("system", "user", "assistant", "tool")


class ChatFormatError(ValueError):
    """Raised for conversations that cannot be rendered safely."""


@dataclass(frozen=True)
class Segment:
    text: str
    train: bool


def _norm_tool_content(content: str) -> str:
    """Ensure tool output is wrapped in <tool_response> tags exactly once."""
    stripped = content.strip()
    if stripped.startswith("<tool_response>") and stripped.endswith("</tool_response>"):
        return stripped
    return f"<tool_response>\n{stripped}\n</tool_response>"


def render_conversation(messages: List[Dict[str, str]]) -> List[Segment]:
    """Render one conversation into ordered (text, train) segments.

    Raises ChatFormatError on structural problems rather than silently
    producing a degenerate training example.
    """
    if not messages:
        raise ChatFormatError("empty conversation")

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role not in VALID_ROLES:
            raise ChatFormatError(f"message {i}: invalid role {role!r}")
        if not isinstance(msg.get("content"), str):
            raise ChatFormatError(f"message {i}: content must be a string")

    if messages[-1]["role"] != "assistant":
        raise ChatFormatError(
            "conversation must end on an assistant turn (nothing to train on "
            "otherwise); Phase 3 should have trimmed trailing env/user turns"
        )
    if not any(m["role"] == "assistant" for m in messages):
        raise ChatFormatError("conversation has no assistant turns")

    segments: List[Segment] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]

        if role == "assistant":
            # Header is untrained; content + closing im_end is trained.
            segments.append(Segment(f"{IM_START}assistant\n", train=False))
            segments.append(Segment(f"{content}{IM_END}\n", train=True))
        elif role == "tool":
            # Qwen's template surfaces tool results as a user turn wrapping
            # the payload in <tool_response> tags.
            body = _norm_tool_content(content)
            segments.append(
                Segment(f"{IM_START}user\n{body}{IM_END}\n", train=False)
            )
        else:  # system / user
            segments.append(
                Segment(f"{IM_START}{role}\n{content}{IM_END}\n", train=False)
            )
    return segments


def render_text(messages: List[Dict[str, str]]) -> str:
    """Full rendered string (used for joint-tokenization drift checks)."""
    return "".join(seg.text for seg in render_conversation(messages))


def render_prompt_and_completion(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Split a conversation into (prompt, completion) at the final assistant
    turn — the shape TRL's ORPOTrainer expects for preference pairs.

    The prompt includes the final assistant header (generation prompt); the
    completion is the final assistant content + im_end.
    """
    segs = render_conversation(messages)
    # Final two segments are always [assistant header, assistant body].
    prompt = "".join(s.text for s in segs[:-1])
    completion = segs[-1].text
    return {"prompt": prompt, "completion": completion}
