"""Graders — one per task type, every one mechanical and deterministic.

Each grader returns a GradeResult(passed, details). `details` always says
*why*, because a failed eval you can't diagnose is noise.

Task types:
    code         run extracted code against the task's assert-based tests
    numeric      final-answer number within tolerance
    mcq          final-answer letter
    tool_call    parsed <tool_call> JSON matches expectations
    instruction  mechanical constraint checks on the visible text
    agentic      handled by agent.py (environment goals), not here
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

from .sandbox import run_python
from .textproc import (
    extract_mcq_answer,
    extract_numeric_answer,
    extract_python_code,
    extract_tool_calls,
)


@dataclass
class GradeResult:
    passed: bool
    details: str


# ----------------------------------------------------------------- code

def grade_code(task: Dict, response: str) -> GradeResult:
    code = extract_python_code(response, task.get("entry_point"))
    entry = task.get("entry_point")
    if entry and not re.search(rf"\b(def|class)\s+{re.escape(entry)}\b", code):
        return GradeResult(False, f"no definition of {entry!r} in extracted code")
    program = f"{code}\n\n# --- tests ---\n{task['test_code']}\nprint('ALL_TESTS_PASSED')\n"
    result = run_python(program, timeout=float(task.get("timeout_s", 10)))
    if result.timed_out:
        return GradeResult(False, "execution timed out")
    if result.ok and "ALL_TESTS_PASSED" in result.stdout:
        return GradeResult(True, "all tests passed")
    return GradeResult(False, f"tests failed ({result.summary()})")


# ----------------------------------------------------------------- numeric

def grade_numeric(task: Dict, response: str) -> GradeResult:
    got = extract_numeric_answer(response)
    if got is None:
        return GradeResult(False, "no numeric answer found in response")
    expected = float(task["answer"])
    abs_tol = float(task.get("abs_tol", 1e-9))
    # Default tolerance is tight (0.1%) so integer answers must be exact;
    # tasks with rounding-sensitive answers set rel_tol explicitly.
    rel_tol = float(task.get("rel_tol", 0.001))
    tol = max(abs_tol, rel_tol * abs(expected))
    if abs(got - expected) <= tol:
        return GradeResult(True, f"answer {got} within tolerance of {expected}")
    return GradeResult(False, f"answer {got}, expected {expected} (±{tol:.6g})")


# ----------------------------------------------------------------- mcq

def grade_mcq(task: Dict, response: str) -> GradeResult:
    got = extract_mcq_answer(response)
    if got is None:
        return GradeResult(False, "no letter answer found in response")
    if got == task["answer"].upper():
        return GradeResult(True, f"answered {got}")
    return GradeResult(False, f"answered {got}, expected {task['answer'].upper()}")


# ----------------------------------------------------------------- tool_call

def _match_call(call: Dict, spec: Dict) -> str | None:
    """None if `call` satisfies `spec`, else a reason string."""
    if call["name"] != spec["name"]:
        return f"name {call['name']!r} != {spec['name']!r}"
    args = call["arguments"]
    for key, want in spec.get("arguments_equal", {}).items():
        if key not in args:
            return f"missing argument {key!r}"
        if args[key] != want:
            return f"argument {key}={args[key]!r}, expected {want!r}"
    for key, needle in spec.get("arguments_contain", {}).items():
        if key not in args:
            return f"missing argument {key!r}"
        if str(needle).lower() not in str(args[key]).lower():
            return f"argument {key}={args[key]!r} does not contain {needle!r}"
    return None


def grade_tool_call(task: Dict, response: str) -> GradeResult:
    calls, errors = extract_tool_calls(response)
    expect = task["expect"]

    if expect.get("no_call"):
        if calls or errors:
            return GradeResult(
                False, f"expected no tool call, got {len(calls)} call(s) "
                       f"{'and format errors' if errors else ''}".strip())
        if expect.get("must_ask_question") and "?" not in response:
            return GradeResult(False, "expected a clarifying question, none asked")
        return GradeResult(True, "correctly answered without calling a tool")

    if errors:
        return GradeResult(False, "malformed tool call: " + "; ".join(errors))

    specs: List[Dict] = expect["calls"]
    if len(calls) != len(specs):
        return GradeResult(
            False, f"expected {len(specs)} call(s), got {len(calls)}: "
                   + ", ".join(c["name"] for c in calls))

    # Greedy bipartite match (order-independent unless order_matters).
    if expect.get("order_matters"):
        for i, (call, spec) in enumerate(zip(calls, specs)):
            reason = _match_call(call, spec)
            if reason:
                return GradeResult(False, f"call {i}: {reason}")
        return GradeResult(True, f"{len(calls)} call(s) matched in order")

    unmatched = list(range(len(calls)))
    for si, spec in enumerate(specs):
        hit = None
        reasons = []
        for ci in unmatched:
            reason = _match_call(calls[ci], spec)
            if reason is None:
                hit = ci
                break
            reasons.append(reason)
        if hit is None:
            return GradeResult(
                False, f"no call matched spec {si} ({spec['name']}): "
                       + "; ".join(reasons[:3]))
        unmatched.remove(hit)
    return GradeResult(True, f"{len(calls)} call(s) matched")


# ----------------------------------------------------------------- instruction

def _words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _check(check: Dict, text: str) -> str | None:
    kind = check["kind"]
    if kind == "max_words":
        n = len(_words(text))
        return None if n <= check["value"] else f"{n} words > max {check['value']}"
    if kind == "min_words":
        n = len(_words(text))
        return None if n >= check["value"] else f"{n} words < min {check['value']}"
    if kind == "must_include":
        missing = [s for s in check["values"] if s.lower() not in text.lower()]
        return None if not missing else f"missing required text: {missing}"
    if kind == "must_include_exact":
        missing = [s for s in check["values"] if s not in text]
        return None if not missing else f"missing exact text: {missing}"
    if kind == "must_not_include":
        present = [s for s in check["values"] if s.lower() in text.lower()]
        return None if not present else f"forbidden text present: {present}"
    if kind == "starts_with":
        return (None if text.lstrip().startswith(check["value"])
                else f"does not start with {check['value']!r}")
    if kind == "ends_with":
        return (None if text.rstrip().endswith(check["value"])
                else f"does not end with {check['value']!r}")
    if kind == "json_only":
        try:
            obj = json.loads(text.strip())
        except json.JSONDecodeError as err:
            return f"not valid JSON: {err}"
        keys = check.get("keys")
        if keys is not None:
            if not isinstance(obj, dict):
                return "JSON is not an object"
            if set(obj.keys()) != set(keys):
                return f"JSON keys {sorted(obj)} != required {sorted(keys)}"
        return None
    if kind == "bullet_count":
        n = len([ln for ln in text.splitlines() if ln.lstrip().startswith("- ")])
        return None if n == check["value"] else f"{n} bullets, expected {check['value']}"
    if kind == "line_count":
        n = len([ln for ln in text.strip().splitlines() if ln.strip()])
        return None if n == check["value"] else f"{n} lines, expected {check['value']}"
    if kind == "paragraph_count":
        n = len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])
        return None if n == check["value"] else f"{n} paragraphs, expected {check['value']}"
    if kind == "lowercase_only":
        uppers = sorted({c for c in text if c.isupper()})
        return None if not uppers else f"uppercase characters present: {uppers[:8]}"
    if kind == "regex_count":
        n = len(re.findall(check["pattern"], text, flags=re.MULTILINE))
        return (None if n == check["value"]
                else f"pattern {check['pattern']!r} matched {n}x, expected {check['value']}")
    if kind == "regex_must_match":
        return (None if re.search(check["pattern"], text, flags=re.MULTILINE)
                else f"pattern {check['pattern']!r} did not match")
    raise ValueError(f"unknown instruction check kind: {kind!r}")


def grade_instruction(task: Dict, response: str) -> GradeResult:
    failures = []
    for check in task["checks"]:
        reason = _check(check, response)
        if reason:
            failures.append(f"[{check['kind']}] {reason}")
    if failures:
        return GradeResult(False, "; ".join(failures))
    return GradeResult(True, f"all {len(task['checks'])} constraints satisfied")


GRADERS: Dict[str, Callable[[Dict, str], GradeResult]] = {
    "code": grade_code,
    "numeric": grade_numeric,
    "mcq": grade_mcq,
    "tool_call": grade_tool_call,
    "instruction": grade_instruction,
}
