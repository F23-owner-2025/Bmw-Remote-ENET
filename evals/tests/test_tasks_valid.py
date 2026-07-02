"""Self-validation of the authored task suites.

The lesson from Phase 3 applies to eval data too: a mis-specified task
silently poisons results. So this module proves, mechanically, that every
task is well-formed AND solvable:

  * every coding task's reference_solution passes its own test_code in the
    real sandbox;
  * every agentic task's reference_commands, executed in the VirtualShell,
    satisfy every goal;
  * every MCQ has exactly 4 options and a valid answer letter;
  * every numeric task has a parseable expected answer;
  * every tool_use expectation references tools that exist in the task's
    own schema list;
  * the committed JSONL is in sync with generate_tasks.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.agent import check_goal
from evalkit.harness import SUITES, load_tasks
from evalkit.sandbox import run_python
from evalkit.shell import VirtualShell

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
ALL_TASKS = load_tasks(TASKS_DIR, list(SUITES))


def tasks_of(suite):
    return ALL_TASKS[suite]


def test_all_suites_present_and_nonempty():
    for suite in SUITES:
        assert len(tasks_of(suite)) >= 8, f"{suite} has too few tasks"


def test_ids_unique_within_suite():
    for suite, tasks in ALL_TASKS.items():
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids)), f"duplicate ids in {suite}"


def test_generated_files_in_sync():
    """Committed JSONL must match what generate_tasks.py produces."""
    result = subprocess.run(
        [sys.executable, str(TASKS_DIR / "generate_tasks.py")],
        capture_output=True, text=True, cwd=TASKS_DIR.parent,
    )
    assert result.returncode == 0, result.stderr
    regenerated = load_tasks(TASKS_DIR, list(SUITES))
    assert regenerated == ALL_TASKS


@pytest.mark.parametrize("task", tasks_of("coding"), ids=lambda t: t["id"])
def test_coding_reference_solutions_pass(task):
    program = (task["reference_solution"] + "\n\n" + task["test_code"]
               + "\nprint('OK')\n")
    result = run_python(program, timeout=10)
    assert result.ok and "OK" in result.stdout, (
        f"reference solution for {task['id']} failed its own tests: "
        f"{result.summary()}")


@pytest.mark.parametrize("task", tasks_of("coding"), ids=lambda t: t["id"])
def test_coding_tasks_have_entry_point_in_reference(task):
    assert task["entry_point"] in task["reference_solution"]


@pytest.mark.parametrize("task", tasks_of("planning_agentic"),
                         ids=lambda t: t["id"])
def test_agentic_reference_commands_satisfy_goals(task):
    shell = VirtualShell(files=task.get("initial_files"),
                         dirs=task.get("initial_dirs"))
    for batch in task["reference_commands"]:
        shell.run("\n".join(batch))
    for goal in task["goals"]:
        reason = check_goal(goal, shell)
        assert reason is None, (
            f"{task['id']}: goal unmet after reference commands: {reason}")


@pytest.mark.parametrize("task", tasks_of("planning_agentic"),
                         ids=lambda t: t["id"])
def test_agentic_goals_not_pretrivially_satisfied(task):
    """The initial state must NOT already satisfy all goals — otherwise the
    task grades a no-op as success."""
    shell = VirtualShell(files=task.get("initial_files"),
                         dirs=task.get("initial_dirs"))
    reasons = [check_goal(g, shell) for g in task["goals"]]
    assert any(r is not None for r in reasons), (
        f"{task['id']}: goals already satisfied before any command")


@pytest.mark.parametrize("task", tasks_of("reasoning"), ids=lambda t: t["id"])
def test_mcq_shape(task):
    assert task["type"] == "mcq"
    assert len(task["options"]) == 4
    assert task["answer"] in ("A", "B", "C", "D")


@pytest.mark.parametrize(
    "task", tasks_of("math") + tasks_of("stem_engineering"),
    ids=lambda t: t["id"])
def test_numeric_shape(task):
    assert task["type"] == "numeric"
    float(task["answer"])  # must be a number
    assert 0 < float(task.get("rel_tol", 0.001)) < 0.1


@pytest.mark.parametrize("task", tasks_of("tool_use"), ids=lambda t: t["id"])
def test_tool_use_expectations_reference_declared_tools(task):
    declared = {t["function"]["name"] for t in task["tools"]}
    expect = task["expect"]
    if expect.get("no_call"):
        return
    for spec in expect["calls"]:
        assert spec["name"] in declared, (
            f"{task['id']}: expected call to undeclared tool {spec['name']!r}")


@pytest.mark.parametrize("task", tasks_of("instruction_following"),
                         ids=lambda t: t["id"])
def test_instruction_checks_are_known_kinds(task):
    from evalkit.graders import _check
    # calling with obviously-satisfiable text must not raise "unknown kind"
    for check in task["checks"]:
        try:
            _check(check, "placeholder text { } 1. - ")
        except ValueError as err:
            raise AssertionError(f"{task['id']}: {err}")


def test_every_task_has_required_common_fields():
    for suite, tasks in ALL_TASKS.items():
        for task in tasks:
            assert task.get("id"), f"missing id in {suite}"
            assert task.get("type"), f"{suite}/{task.get('id')}: missing type"
            assert task.get("prompt"), f"{suite}/{task['id']}: missing prompt"
