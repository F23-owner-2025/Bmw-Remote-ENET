"""Integration test of train_sft.py's pre-GPU pipeline (the --dry-run path):
mask gate -> tokenize -> split -> pack -> HF columns. Uses the fake
tokenizer, so it runs anywhere; the real dry run against the actual Qwen3.6
tokenizer is step 1 of the runbook on the training box.
"""

import json
import textwrap

import pytest

import train_sft
from tests.conftest import (
    FakeTokenizer,
    make_agentic_conversation,
    make_chat_conversation,
)
from trainkit.config import load_config


@pytest.fixture
def workdir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with open(data / "train.jsonl", "w") as fh:
        for i in range(40):
            fh.write(json.dumps({"messages": make_agentic_conversation(f"task {i}")}) + "\n")
            fh.write(json.dumps({"messages": make_chat_conversation(f"Question {i}?")}) + "\n")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        run_name: dry_run_test
        stage: sft
        model: {{max_seq_len: 2048}}
        data: {{train_path: "{data / 'train.jsonl'}", eval_fraction: 0.05}}
        mask_check: {{sample_size: 50}}
        checkpointing: {{output_dir: "{tmp_path / 'runs'}"}}
    """))
    return cfg_file


def test_dry_run_pipeline(workdir, capsys):
    cfg = load_config(workdir)
    tok = FakeTokenizer()

    train_sft.run_mask_gate(cfg, tok)  # must not SystemExit
    out = capsys.readouterr().out
    assert "PASS" in out

    train_ds, eval_ds, stats = train_sft.build_datasets(cfg, tok)
    assert stats.examples_kept == 80
    assert stats.rows_malformed == 0
    assert len(train_ds) >= 1
    assert eval_ds is not None and len(eval_ds) >= 1

    row = train_ds[0]
    assert len(row["input_ids"]) == 2048
    assert set(row.keys()) == {"input_ids", "labels", "position_ids", "attention_mask"}
    # packed rows contain real trainable content
    assert any(l != -100 for l in row["labels"])


def test_mask_gate_blocks_poisoned_data(tmp_path, capsys):
    """A corpus whose 'assistant' turns are actually empty (mask eats all
    loss) must be refused before training."""
    data = tmp_path / "train.jsonl"
    with open(data, "w") as fh:
        for i in range(10):
            long_question = " ".join(f"detail{j}" for j in range(120))
            msgs = make_chat_conversation(f"Q{i}: {long_question}")
            msgs[-1]["content"] = ""  # degenerate assistant turns
            fh.write(json.dumps({"messages": msgs}) + "\n")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(textwrap.dedent(f"""
        stage: sft
        data: {{train_path: "{data}"}}
        mask_check: {{min_trainable_fraction: 0.05}}
    """))
    cfg = load_config(cfg_file)
    with pytest.raises(SystemExit, match="mask verification FAILED"):
        train_sft.run_mask_gate(cfg, FakeTokenizer())
