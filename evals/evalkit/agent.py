"""The agentic episode runner: multi-turn plan→act→observe loop against
VirtualShell, graded by environment-state goals.

An episode:
    1. system prompt (Hermes run_commands schema + available commands)
       + the user task
    2. model responds; any <tool_call> run_commands batches are executed in
       the shell, and outputs return as `tool` turns (wrapped the way the
       model saw during training)
    3. loop until the model replies with no tool call (done), or the turn
       budget runs out
    4. grade: every goal predicate is checked against the final shell state

Malformed tool calls are answered with an error tool-response instead of
aborting — recovering from its own formatting slip is part of what we're
measuring — but the episode still fails if goals aren't met.

This module is deliberately the shape of an RL environment (reset ->
episode -> verifiable reward) so it can be lifted into a GRPO/RLVR loop
for v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .hermes import agentic_system_prompt
from .shell import VirtualShell
from .textproc import extract_tool_calls, strip_think

DEFAULT_MAX_TURNS = 8
MAX_TOOL_RESPONSE_CHARS = 3000


@dataclass
class EpisodeResult:
    passed: bool
    details: str
    turns_used: int
    goals_passed: int
    goals_total: int
    transcript: List[Dict[str, str]] = field(default_factory=list)


def check_goal(goal: Dict, shell: VirtualShell) -> str | None:
    """None if satisfied, else a reason string."""
    kind, path = goal["kind"], goal.get("path", "")
    if kind == "file_exists":
        return None if shell.is_file(path) else f"file missing: {path}"
    if kind == "file_absent":
        return None if not shell.is_file(path) else f"file still exists: {path}"
    if kind == "dir_exists":
        return None if shell.is_dir(path) else f"directory missing: {path}"
    if kind == "file_contains":
        content = shell.read(path)
        if content is None:
            return f"file missing: {path}"
        return None if goal["text"] in content else \
            f"{path} does not contain {goal['text']!r}"
    if kind == "file_not_contains":
        content = shell.read(path)
        if content is None:
            return f"file missing: {path}"
        return None if goal["text"] not in content else \
            f"{path} still contains {goal['text']!r}"
    if kind == "file_equals":
        content = shell.read(path)
        if content is None:
            return f"file missing: {path}"
        return None if content.strip() == goal["text"].strip() else \
            f"{path} content differs from expected"
    raise ValueError(f"unknown goal kind: {goal['kind']!r}")


def run_episode(task: Dict, client) -> EpisodeResult:
    shell = VirtualShell(files=task.get("initial_files"),
                         dirs=task.get("initial_dirs"),
                         cwd=task.get("cwd", "/work"))
    max_turns = int(task.get("max_turns", DEFAULT_MAX_TURNS))

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": agentic_system_prompt()},
        {"role": "user", "content": task["prompt"]},
    ]
    turns = 0
    for _ in range(max_turns):
        response = client.chat(messages)
        turns += 1
        messages.append({"role": "assistant", "content": response})

        calls, errors = extract_tool_calls(strip_think(response))
        if not calls and not errors:
            break  # model considers the task done

        for err in errors:
            messages.append({
                "role": "tool",
                "content": f"<tool_response>\nERROR: {err}\n</tool_response>",
            })
        for call in calls:
            if call["name"] != "run_commands":
                output = f"ERROR: unknown tool {call['name']!r}"
            else:
                commands = call["arguments"].get("commands")
                if not isinstance(commands, list) or \
                        not all(isinstance(c, str) for c in commands):
                    output = "ERROR: 'commands' must be an array of strings"
                else:
                    output = shell.run("\n".join(commands))
            output = output[:MAX_TOOL_RESPONSE_CHARS] or "(no output)"
            messages.append({
                "role": "tool",
                "content": f"<tool_response>\n{output}\n</tool_response>",
            })
    else:
        # Loop exhausted without a tool-free final answer.
        pass

    goals = task["goals"]
    reasons = [r for r in (check_goal(g, shell) for g in goals) if r]
    passed = not reasons
    ran_out = turns >= max_turns and not passed
    details = (f"all {len(goals)} goals met in {turns} turn(s)" if passed else
               ("turn budget exhausted; " if ran_out else "") + "; ".join(reasons))
    return EpisodeResult(
        passed=passed,
        details=details,
        turns_used=turns,
        goals_passed=len(goals) - len(reasons),
        goals_total=len(goals),
        transcript=messages,
    )
