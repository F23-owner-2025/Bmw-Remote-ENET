# Phase 6 — Evaluation Suite

Verifiable evaluation of the fine-tuned Qwen3.6-27B assistant across the
seven capability areas from the project plan: **coding, engineering,
mathematics, reasoning, tool use, planning, instruction following** — 68
tasks, every one graded mechanically (executed code, checked environment
state, parsed tool-call JSON, numeric tolerance, constraint checks). No LLM
judges anywhere: grades are reproducible, and the agentic episodes are
exactly the shape of an RLVR reward environment, so this suite becomes the
RL environment if the project moves to a v2 GRPO stage (the Phase 1 plan).

## The suites

| Suite | Tasks | Grading |
|---|---|---|
| `coding` | 10 | generated Python runs against assert tests in a subprocess sandbox |
| `math` | 10 | final numeric answer within per-task tolerance |
| `stem_engineering` | 10 | ME/EE/physics problems, numeric with tolerance |
| `reasoning` | 10 | multiple choice, letter extraction |
| `tool_use` | 10 | parsed `<tool_call>` JSON vs expected calls — incl. parallel calls, "no tool needed", and "ask a clarifying question instead of guessing" |
| `instruction_following` | 10 | mechanical constraints: word caps, JSON-only, exact markers, casing, counts |
| `planning_agentic` | 8 | multi-turn episodes in a deterministic virtual shell, graded by final environment state |

The tool-calling format is the **Hermes convention the model was trained
on** (Phases 3–4): schemas in `<tools>` tags in the system prompt, calls as
`<tool_call>` JSON in content. `<think>` blocks are stripped before grading.

## Running

Works against any OpenAI-compatible endpoint — vLLM, llama.cpp server
(`--api`), or Ollama (`http://localhost:11434/v1`):

```bash
pip install -r requirements.txt

# sanity: task inventory and the harness's own tests
python run_evals.py list
python -m pytest

# evaluate the fine-tune
python run_evals.py run \
    --base-url http://localhost:8000/v1 \
    --model qwen3.6-27b-assistant \
    --out results/finetuned.jsonl

# baseline the stock model the same way, then compare
python run_evals.py run --base-url http://localhost:8000/v1 \
    --model qwen3.6-27b --out results/base.jsonl
python run_evals.py compare results/base.jsonl results/finetuned.jsonl
```

Useful flags: `--suites coding tool_use` (subset), `--limit 3` (quick pass),
`--concurrency 4` (parallel requests), `--fresh` (discard prior results).
Runs are **resumable**: results append to `--out` as they finish, and a
rerun skips completed tasks — an interrupted run costs nothing.

Temperature defaults to 0.0 for reproducible grading.

## Layout

```
evals/
├── run_evals.py            # CLI: run / report / compare / list
├── evalkit/
│   ├── client.py           # OpenAI-compatible client + scripted test client
│   ├── harness.py          # dispatch, incremental results, resume
│   ├── graders.py          # code-exec, numeric, MCQ, tool-call, instruction
│   ├── agent.py            # multi-turn agentic episodes + goal predicates
│   ├── shell.py            # deterministic in-memory VirtualShell
│   ├── sandbox.py          # subprocess sandbox for generated Python
│   ├── hermes.py           # training-matched tool-calling system prompts
│   ├── textproc.py         # think-stripping, answer/code/tool-call extraction
│   └── report.py           # per-suite stats, markdown report, A/B comparison
├── tasks/
│   ├── generate_tasks.py   # source of record for all tasks
│   └── *.jsonl             # generated task files (committed)
└── tests/                  # 154 tests, no GPU or network needed
```

## Why the agentic environment is simulated

`VirtualShell` implements a small, documented command set (`ls`, `cat`,
`echo` with redirects, `mkdir -p`, `rm`, `mv`, `cp`, `grep` with real
exit-code semantics, `&&`/`;` chaining) in memory. Simulated beats real
bash here for three reasons: grading is exactly reproducible, nothing the
model emits can damage the eval box, and episodes are cheap enough to run
thousands of times — which is precisely what an RL loop needs later. The
model is told the available commands in its system prompt.

## Task quality is enforced, not assumed

`tests/test_tasks_valid.py` proves every task is solvable and correctly
specified: each coding task's committed reference solution must pass its
own tests in the real sandbox; each agentic task's reference command
sequence must satisfy every goal in the VirtualShell — and the initial
state must *not* already satisfy the goals (no grading a no-op as
success); MCQ/numeric/tool-use schemas are shape-checked; and the
committed JSONL must match what `generate_tasks.py` regenerates.

## Interpreting results

- Run the **base model first**. The fine-tune's value shows up in the
  comparison, especially `tool_use` (schema exactness, clarify-vs-guess)
  and `planning_agentic` (inspect-before-modify, goal completion) — the
  behaviors the Phase 2/3 data targeted.
- `report` lists every failure with the grader's reason; agentic rows
  include the full transcript in the results JSONL for replay.
- A regression in `coding`/`math`/`reasoning` relative to base is the
  canonical sign of catastrophic forgetting — revisit the Phase 2 mixture
  weights (the general/STEM slices exist to prevent exactly this).

## Extending

Add tasks in `tasks/generate_tasks.py` (with a reference solution or
reference commands — the test suite will hold you to it), rerun it, and
commit both. New instruction-check kinds go in `graders._check`; new goal
predicates in `agent.check_goal`.
