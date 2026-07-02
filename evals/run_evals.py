#!/usr/bin/env python3
"""Phase 6 evaluation CLI.

    # Run everything against a local OpenAI-compatible endpoint
    python run_evals.py run --base-url http://localhost:8000/v1 \\
        --model qwen3.6-27b-assistant --out results/finetuned.jsonl

    # Only some suites, 4 requests in flight
    python run_evals.py run --base-url http://localhost:11434/v1 \\
        --model qwen3.6 --suites coding tool_use --concurrency 4 \\
        --out results/base.jsonl

    # Reports
    python run_evals.py report results/finetuned.jsonl
    python run_evals.py compare results/base.jsonl results/finetuned.jsonl

    # Inventory
    python run_evals.py list

Runs are resumable: rerunning with the same --out skips completed tasks
(use --fresh to discard and redo). Endpoint notes: vLLM serves /v1 natively;
llama.cpp server with --api; Ollama at http://localhost:11434/v1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalkit.client import ChatClient
from evalkit.harness import SUITES, load_tasks, run_suites
from evalkit.report import format_comparison, format_report, load_results

TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run suites against a model endpoint")
    run.add_argument("--base-url", required=True,
                     help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    run.add_argument("--model", required=True, help="model name at the endpoint")
    run.add_argument("--api-key", default=None)
    run.add_argument("--suites", nargs="+", default=["all"],
                     help=f"suites to run (default all): {', '.join(SUITES)}")
    run.add_argument("--out", required=True, help="results JSONL (appended; resumable)")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--limit", type=int, default=None,
                     help="cap tasks per suite (quick sanity runs)")
    run.add_argument("--temperature", type=float, default=0.0,
                     help="0.0 for reproducible grading (default)")
    run.add_argument("--max-tokens", type=int, default=3072)
    run.add_argument("--fresh", action="store_true",
                     help="discard existing results in --out instead of resuming")

    rep = sub.add_parser("report", help="summarize one results file")
    rep.add_argument("results")

    cmp_ = sub.add_parser("compare", help="compare two results files (A -> B)")
    cmp_.add_argument("results_a")
    cmp_.add_argument("results_b")

    sub.add_parser("list", help="list suites and task counts")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "list":
        tasks = load_tasks(TASKS_DIR, list(SUITES))
        total = 0
        for suite, items in tasks.items():
            print(f"{suite:<24} {len(items)} tasks")
            total += len(items)
        print(f"{'total':<24} {total} tasks")
        return

    if args.cmd == "report":
        print(format_report(load_results(args.results),
                            title=f"Evaluation report — {args.results}"))
        return

    if args.cmd == "compare":
        print(format_comparison(
            load_results(args.results_a), load_results(args.results_b),
            name_a=Path(args.results_a).stem, name_b=Path(args.results_b).stem))
        return

    # run
    suites = list(SUITES) if args.suites == ["all"] else args.suites
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; valid: {', '.join(SUITES)}")

    tasks = load_tasks(TASKS_DIR, suites)
    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    results = run_suites(
        tasks, client, Path(args.out),
        concurrency=args.concurrency, limit=args.limit, fresh=args.fresh,
    )
    print()
    print(format_report(load_results(args.out),
                        title=f"Evaluation report — {args.model}"))
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
