import json

from tests.conftest import TOOL_SYSTEM
from train_orpo import render_pair
from trainkit.data import iter_jsonl


def base_pair():
    return {
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content": "Delete all files in /tmp/build."},
        ],
        "chosen": (
            "Before running a destructive command I will confirm the target.\n"
            "<tool_call>\n"
            '{"name": "run_commands", "arguments": {"commands": ["ls /tmp/build"]}}\n'
            "</tool_call>"
        ),
        "rejected": (
            "<tool_call>\n"
            '{"name": "run_commands", "arguments": {"commands": ["rm -rf /tmp/build/*"]}}\n'
            "</tool_call>"
        ),
    }


def test_render_pair_shape():
    out = render_pair(base_pair())
    assert out is not None
    assert out["prompt"].endswith("<|im_start|>assistant\n")
    assert out["prompt"].startswith("<|im_start|>system\n")
    assert out["chosen"].rstrip().endswith("<|im_end|>")
    assert out["rejected"].rstrip().endswith("<|im_end|>")
    assert "Delete all files" in out["prompt"]


def test_completions_not_double_terminated():
    p = base_pair()
    p["chosen"] = p["chosen"] + "<|im_end|>"
    out = render_pair(p)
    assert out["chosen"].count("<|im_end|>") == 1


def test_identical_pair_rejected():
    p = base_pair()
    p["rejected"] = p["chosen"]
    assert render_pair(p) is None


def test_context_ending_on_assistant_rejected():
    p = base_pair()
    p["messages"].append({"role": "assistant", "content": "already answered"})
    assert render_pair(p) is None


def test_missing_fields_rejected():
    for key in ("messages", "chosen", "rejected"):
        p = base_pair()
        del p[key]
        assert render_pair(p) is None
    p = base_pair()
    p["rejected"] = "   "
    assert render_pair(p) is None


def test_multi_turn_context_with_tool_turns():
    p = base_pair()
    p["messages"] += [
        {"role": "assistant", "content": "Checking.\n<tool_call>\n"
         '{"name": "run_commands", "arguments": {"commands": ["ls /tmp/build"]}}\n'
         "</tool_call>"},
        {"role": "tool", "content": "artifact.bin cache/"},
        {"role": "user", "content": "Yes, go ahead."},
    ]
    out = render_pair(p)
    assert out is not None
    assert "<tool_response>" in out["prompt"]
    assert out["prompt"].endswith("<|im_start|>assistant\n")


def test_iter_jsonl_roundtrip(tmp_path):
    p = tmp_path / "pairs.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps(base_pair()) + "\n\n")  # blank line tolerated
        fh.write("garbage\n")
    rows = list(iter_jsonl(p))
    assert len(rows) == 2
    assert rows[0][1] is not None and rows[1][1] is None
