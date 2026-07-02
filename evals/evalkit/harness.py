"""The harness: task files -> prompts -> model -> graders -> results.jsonl.

Fault tolerance: every result is appended to the output file the moment it
finishes, and a rerun with the same output file skips already-completed
tasks — a crashed or interrupted eval run resumes for free (same philosophy
as the training resume).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .agent import run_episode
from .graders import GRADERS
from .hermes import tool_system_prompt
from .textproc import strip_think

SUITES = (
    "coding",
    "math",
    "stem_engineering",
    "reasoning",
    "tool_use",
    "instruction_following",
    "planning_agentic",
)

NUMERIC_SUFFIX = (
    "\n\nSolve this step by step. End your response with a line of the form:\n"
    "FINAL ANSWER: <number>\n"
    "with a single number in the requested units (no words after it)."
)
MCQ_SUFFIX = (
    "\n\nThink it through, then end your response with a line of the form:\n"
    "FINAL ANSWER: <letter>"
)
CODE_SUFFIX = (
    "\n\nProvide a complete, self-contained Python implementation in a single "
    "```python code block. No usage examples are required."
)


def load_tasks(tasks_dir: Path, suites: List[str]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for suite in suites:
        path = tasks_dir / f"{suite}.jsonl"
        if not path.exists():
            raise SystemExit(f"no task file for suite {suite!r}: {path}")
        tasks = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as err:
                    raise SystemExit(f"{path}:{lineno}: bad JSON: {err}")
                task["suite"] = suite
                tasks.append(task)
        out[suite] = tasks
    return out


def build_messages(task: Dict) -> List[Dict[str, str]]:
    ttype = task["type"]
    if ttype == "numeric":
        return [{"role": "user", "content": task["prompt"] + NUMERIC_SUFFIX}]
    if ttype == "mcq":
        options = "\n".join(f"{letter}) {text}" for letter, text
                            in zip("ABCD", task["options"]))
        return [{"role": "user",
                 "content": f"{task['prompt']}\n\n{options}{MCQ_SUFFIX}"}]
    if ttype == "code":
        return [{"role": "user", "content": task["prompt"] + CODE_SUFFIX}]
    if ttype == "tool_call":
        return [
            {"role": "system", "content": tool_system_prompt(task["tools"])},
            {"role": "user", "content": task["prompt"]},
        ]
    if ttype == "instruction":
        return [{"role": "user", "content": task["prompt"]}]
    raise ValueError(f"no message builder for task type {ttype!r}")


def run_task(task: Dict, client) -> Dict:
    start = time.monotonic()
    row: Dict = {
        "suite": task["suite"],
        "id": task["id"],
        "type": task["type"],
        "model": getattr(client, "model", "unknown"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        if task["type"] == "agentic":
            episode = run_episode(task, client)
            row.update(
                passed=episode.passed,
                details=episode.details,
                turns_used=episode.turns_used,
                goals_passed=episode.goals_passed,
                goals_total=episode.goals_total,
                transcript=episode.transcript,
            )
        else:
            messages = build_messages(task)
            raw = client.chat(messages)
            visible = strip_think(raw)
            grade = GRADERS[task["type"]](task, visible)
            row.update(passed=grade.passed, details=grade.details, response=raw)
    except Exception as err:  # a single bad task/endpoint hiccup must not kill the run
        row.update(passed=False, details=f"harness error: {type(err).__name__}: {err}")
    row["duration_s"] = round(time.monotonic() - start, 2)
    return row


def completed_keys(out_path: Path) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add((row["suite"], row["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue  # torn write from a crash; the task will rerun
    return done


def run_suites(
    tasks_by_suite: Dict[str, List[Dict]],
    client,
    out_path: Path,
    concurrency: int = 1,
    limit: Optional[int] = None,
    fresh: bool = False,
) -> List[Dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and out_path.exists():
        out_path.unlink()
    done = completed_keys(out_path)

    todo: List[Dict] = []
    for suite, tasks in tasks_by_suite.items():
        selected = tasks[:limit] if limit else tasks
        for task in selected:
            if (suite, task["id"]) in done:
                continue
            todo.append(task)
    if done:
        print(f"[harness] resuming: {len(done)} task(s) already complete, "
              f"{len(todo)} to run")

    results: List[Dict] = []
    lock = threading.Lock()

    def finish(row: Dict) -> None:
        with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            results.append(row)
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"[{mark}] {row['suite']}/{row['id']}: {row['details'][:100]}")

    if concurrency <= 1:
        for task in todo:
            finish(run_task(task, client))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_task, t, client) for t in todo]
            for fut in as_completed(futures):
                finish(fut.result())
    return results
