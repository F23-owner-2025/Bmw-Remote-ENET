import json

import pytest

from tests.conftest import make_agentic_conversation, make_chat_conversation
from trainkit.data import (
    build_packed_dataset,
    load_and_tokenize,
    rows_to_hf_columns,
    split_train_eval,
)
from trainkit.masking import IGNORE_INDEX, TokenizedExample, tokenize_conversation
from trainkit.packing import pack_ffd, packing_efficiency


def ex_of_len(n: int) -> TokenizedExample:
    return TokenizedExample(input_ids=list(range(n)), labels=list(range(n)))


# --------------------------- packing ---------------------------------------

def test_ffd_respects_capacity():
    examples = [ex_of_len(n) for n in (60, 50, 40, 30, 20, 10)]
    rows = pack_ffd(examples, max_seq_len=100)
    assert all(len(r) <= 100 for r in rows)
    assert sum(r.num_examples for r in rows) == 6


def test_ffd_finds_the_optimum_here():
    # 210 total tokens in bins of 100 -> optimum is 3 bins. FFD packs
    # 60+40, 50+30+20, 10 — exactly 3, with two bins completely full.
    examples = [ex_of_len(n) for n in (60, 50, 40, 30, 20, 10)]
    rows = pack_ffd(examples, max_seq_len=100)
    assert len(rows) == 3
    assert sorted(len(r) for r in rows) == [10, 100, 100]
    assert packing_efficiency(rows, 100) == pytest.approx(210 / 300)


def test_position_ids_reset_per_example():
    rows = pack_ffd([ex_of_len(3), ex_of_len(2)], max_seq_len=10)
    row = rows[0]
    assert row.num_examples == 2
    assert row.position_ids == [0, 1, 2, 0, 1]


def test_no_reset_gives_monotone_positions():
    rows = pack_ffd([ex_of_len(3), ex_of_len(2)], max_seq_len=10,
                    reset_position_ids=False)
    assert rows[0].position_ids == [0, 1, 2, 3, 4]


def test_overlong_example_raises():
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        pack_ffd([ex_of_len(11)], max_seq_len=10)


def test_labels_preserved_through_packing(tokenizer):
    exs = [tokenize_conversation(make_agentic_conversation(f"task {i}"), tokenizer)
           for i in range(5)]
    rows = pack_ffd(exs, max_seq_len=4096)
    packed_trainable = sum(
        1 for r in rows for l in r.labels if l != IGNORE_INDEX)
    orig_trainable = sum(e.num_trainable for e in exs)
    assert packed_trainable == orig_trainable


def test_hf_columns_padding(tokenizer):
    exs = [tokenize_conversation(make_chat_conversation(), tokenizer)]
    rows, _ = build_packed_dataset(exs, max_seq_len=256)
    cols = rows_to_hf_columns(rows, tokenizer.pad_token_id, 256)
    ids, labels, pos, mask = (cols["input_ids"][0], cols["labels"][0],
                              cols["position_ids"][0], cols["attention_mask"][0])
    assert len(ids) == len(labels) == len(pos) == len(mask) == 256
    n_real = sum(mask)
    assert all(t == tokenizer.pad_token_id for t in ids[n_real:])
    assert all(l == IGNORE_INDEX for l in labels[n_real:])
    # padding positions continue monotonically past the last real position
    assert pos[n_real:] == list(range(pos[n_real - 1] + 1,
                                      pos[n_real - 1] + 1 + 256 - n_real))


# --------------------------- data loading ----------------------------------

def write_jsonl(path, objs):
    with open(path, "w", encoding="utf-8") as fh:
        for o in objs:
            fh.write(json.dumps(o) + "\n")


def test_load_and_tokenize_happy_path(tmp_path, tokenizer):
    p = tmp_path / "train.jsonl"
    write_jsonl(p, [{"messages": make_agentic_conversation(f"t{i}")} for i in range(10)])
    exs, stats = load_and_tokenize(p, tokenizer, max_seq_len=4096)
    assert stats.examples_kept == 10 and len(exs) == 10
    assert stats.rows_malformed == 0
    assert stats.trainable_tokens > 0


def test_malformed_rows_skipped_not_fatal(tmp_path, tokenizer):
    p = tmp_path / "train.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"messages": make_chat_conversation()}) + "\n")
        fh.write("{not json}\n")
        fh.write(json.dumps({"no_messages": True}) + "\n")
        # ends on tool turn -> render failure
        fh.write(json.dumps({"messages": make_agentic_conversation()[:-1]}) + "\n")
    exs, stats = load_and_tokenize(p, tokenizer, max_seq_len=4096)
    assert len(exs) == 1
    assert stats.rows_malformed == 2
    assert stats.rows_render_failed == 1


def test_overlong_dropped_and_noted(tmp_path, tokenizer):
    p = tmp_path / "train.jsonl"
    long_msgs = make_chat_conversation()
    long_msgs[-1]["content"] = " ".join(f"w{i}" for i in range(500))
    write_jsonl(p, [{"messages": long_msgs}] * 3)
    exs, stats = load_and_tokenize(p, tokenizer, max_seq_len=64)
    assert len(exs) == 0
    assert stats.rows_overlong_dropped == 3
    assert any("overlong" in n for n in stats.notes)


def test_split_train_eval_disjoint_and_sized():
    exs = [ex_of_len(5) for _ in range(100)]
    train, evals = split_train_eval(exs, eval_fraction=0.1, seed=1)
    assert len(evals) == 10 and len(train) == 90


def test_split_zero_fraction_keeps_all():
    exs = [ex_of_len(5) for _ in range(10)]
    train, evals = split_train_eval(exs, eval_fraction=0.0, seed=1)
    assert len(train) == 10 and evals == []
