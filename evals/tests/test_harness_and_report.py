import json
from pathlib import Path

from evalkit.client import ScriptedClient
from evalkit.harness import build_messages, load_tasks, run_suites
from evalkit.report import format_comparison, format_report, load_results, suite_stats

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def mini_suite():
    return {
        "math": [
            {"id": "t1", "suite": "math", "type": "numeric", "answer": 4,
             "prompt": "What is 2+2?"},
            {"id": "t2", "suite": "math", "type": "numeric", "answer": 9,
             "prompt": "What is 3*3?"},
        ]
    }


def test_run_and_resume(tmp_path):
    out = tmp_path / "results.jsonl"
    client = ScriptedClient(responses=["FINAL ANSWER: 4", "FINAL ANSWER: 8"])
    results = run_suites(mini_suite(), client, out)
    assert len(results) == 2
    rows = load_results(out)
    assert rows[("math", "t1")]["passed"] is True
    assert rows[("math", "t2")]["passed"] is False

    # Rerun with the same out file: everything already complete, no requests.
    client2 = ScriptedClient(responses=[])
    results2 = run_suites(mini_suite(), client2, out)
    assert results2 == [] and client2.requests == []

    # --fresh discards and reruns
    client3 = ScriptedClient(responses=["FINAL ANSWER: 4", "FINAL ANSWER: 9"])
    results3 = run_suites(mini_suite(), client3, out, fresh=True)
    assert len(results3) == 2
    assert load_results(out)[("math", "t2")]["passed"] is True


def test_client_exception_recorded_not_fatal(tmp_path):
    def boom(messages):
        raise RuntimeError("endpoint down")

    out = tmp_path / "results.jsonl"
    results = run_suites(mini_suite(), ScriptedClient(fn=boom), out)
    assert len(results) == 2
    assert all(not r["passed"] for r in results)
    assert all("harness error" in r["details"] for r in results)


def test_concurrency_produces_all_rows(tmp_path):
    out = tmp_path / "results.jsonl"
    client = ScriptedClient(fn=lambda m: "FINAL ANSWER: 4")
    results = run_suites(mini_suite(), client, out, concurrency=4)
    assert len(results) == 2
    assert len(load_results(out)) == 2


def test_build_messages_shapes():
    numeric = build_messages({"type": "numeric", "prompt": "2+2?"})
    assert "FINAL ANSWER" in numeric[0]["content"]

    mcq = build_messages({"type": "mcq", "prompt": "Pick.",
                          "options": ["w", "x", "y", "z"]})
    assert "A) w" in mcq[0]["content"] and "D) z" in mcq[0]["content"]

    tool = build_messages({"type": "tool_call", "prompt": "Do it.",
                           "tools": [{"type": "function",
                                      "function": {"name": "f", "parameters": {}}}]})
    assert tool[0]["role"] == "system" and "<tools>" in tool[0]["content"]

    code = build_messages({"type": "code", "prompt": "Write f."})
    assert "```python" in code[0]["content"]


def test_load_tasks_real_files():
    tasks = load_tasks(TASKS_DIR, ["math", "planning_agentic"])
    assert len(tasks["math"]) == 10
    assert all(t["suite"] == "math" for t in tasks["math"])


def test_report_and_compare(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    rows_a = [
        {"suite": "math", "id": "t1", "passed": True, "details": "ok"},
        {"suite": "math", "id": "t2", "passed": False, "details": "wrong"},
    ]
    rows_b = [
        {"suite": "math", "id": "t1", "passed": True, "details": "ok"},
        {"suite": "math", "id": "t2", "passed": True, "details": "ok"},
    ]
    a.write_text("\n".join(json.dumps(r) for r in rows_a))
    b.write_text("\n".join(json.dumps(r) for r in rows_b))

    ra, rb = load_results(a), load_results(b)
    assert suite_stats(ra)["math"]["rate"] == 0.5
    report = format_report(ra)
    assert "50%" in report and "t2" in report

    cmp_text = format_comparison(ra, rb, "base", "tuned")
    assert "+50%" in cmp_text and "Fixed in tuned" in cmp_text


def test_latest_result_wins(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(
        json.dumps({"suite": "s", "id": "x", "passed": False, "details": "old"})
        + "\n"
        + json.dumps({"suite": "s", "id": "x", "passed": True, "details": "new"})
        + "\n")
    rows = load_results(p)
    assert rows[("s", "x")]["passed"] is True
