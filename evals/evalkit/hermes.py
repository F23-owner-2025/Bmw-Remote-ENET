"""Hermes-format system prompts — the exact tool-calling convention the
model was trained on in Phases 3–4. Tool schemas go in <tools> tags in the
system prompt; the model answers with <tool_call> JSON in plain content.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .shell import COMMAND_DOC

RUN_COMMANDS_TOOL: Dict = {
    "type": "function",
    "function": {
        "name": "run_commands",
        "description": (
            "Execute a batch of shell commands in the workspace, in order. "
            "Returns their combined output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Shell commands to execute, in order.",
                }
            },
            "required": ["commands"],
        },
    },
}


def tool_system_prompt(tools: List[Dict], extra: str = "") -> str:
    schema_lines = "\n".join(json.dumps(t) for t in tools)
    body = (
        "You are a senior engineering assistant. You have access to the "
        "following tools, described within <tools></tools> XML tags:\n"
        f"<tools>\n{schema_lines}\n</tools>\n"
        "When a tool is needed, reply with a JSON object inside "
        "<tool_call></tool_call> tags, like:\n"
        '<tool_call>\n{"name": "<function-name>", "arguments": <args-json>}\n'
        "</tool_call>\n"
        "Only call a tool when it is actually required and you have all the "
        "information its arguments need. If required information is missing, "
        "ask a clarifying question instead of guessing. If no tool is "
        "relevant, answer directly."
    )
    if extra:
        body += "\n\n" + extra
    return body


def agentic_system_prompt() -> str:
    return tool_system_prompt(
        [RUN_COMMANDS_TOOL],
        extra=(
            "You are operating in a small sandboxed shell. Available "
            f"commands: {COMMAND_DOC}\n"
            "Work step by step: plan briefly, run commands, read their "
            "output, and adapt. Inspect files before modifying them. When "
            "the task is fully complete, reply WITHOUT any tool call, "
            "summarizing what you did."
        ),
    )
