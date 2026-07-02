import pytest

from tests.conftest import make_agentic_conversation, make_chat_conversation
from trainkit.chat_format import (
    ChatFormatError,
    render_conversation,
    render_prompt_and_completion,
    render_text,
)
from trainkit.masking import IGNORE_INDEX, tokenize_conversation


# --------------------------- rendering ------------------------------------

def test_segments_alternate_train_flags():
    segs = render_conversation(make_agentic_conversation())
    # system, user, [asst header, asst body], tool, [asst header, asst body]
    flags = [s.train for s in segs]
    assert flags == [False, False, False, True, False, False, True]


def test_assistant_body_trained_header_not():
    segs = render_conversation(make_chat_conversation())
    header, body = segs[-2], segs[-1]
    assert header.text == "<|im_start|>assistant\n" and not header.train
    assert body.train and body.text.endswith("<|im_end|>\n")
    assert "Torque" in body.text


def test_tool_turn_rendered_as_user_with_tool_response():
    segs = render_conversation(make_agentic_conversation())
    tool_seg = segs[4]
    assert not tool_seg.train
    assert tool_seg.text.startswith("<|im_start|>user\n<tool_response>")
    assert "</tool_response>" in tool_seg.text


def test_tool_content_not_double_wrapped():
    msgs = make_agentic_conversation()
    msgs[3]["content"] = "<tool_response>\nalready wrapped\n</tool_response>"
    segs = render_conversation(msgs)
    assert segs[4].text.count("<tool_response>") == 1


def test_rejects_trailing_non_assistant():
    msgs = make_agentic_conversation()[:-1]  # now ends on tool turn
    with pytest.raises(ChatFormatError, match="assistant turn"):
        render_conversation(msgs)


def test_rejects_bad_role_and_empty():
    with pytest.raises(ChatFormatError):
        render_conversation([])
    with pytest.raises(ChatFormatError, match="invalid role"):
        render_conversation([{"role": "robot", "content": "x"},
                             {"role": "assistant", "content": "y"}])
    with pytest.raises(ChatFormatError, match="string"):
        render_conversation([{"role": "user", "content": 42},
                             {"role": "assistant", "content": "y"}])


def test_prompt_completion_split():
    pc = render_prompt_and_completion(make_chat_conversation())
    assert pc["prompt"].endswith("<|im_start|>assistant\n")
    assert pc["completion"].startswith("Torque")
    assert pc["prompt"] + pc["completion"] == render_text(make_chat_conversation())


# --------------------------- masking --------------------------------------

def test_labels_match_assistant_tokens_exactly(tokenizer):
    msgs = make_agentic_conversation()
    ex = tokenize_conversation(msgs, tokenizer)
    assert len(ex.input_ids) == len(ex.labels)

    trained = tokenizer.decode([t for t, l in zip(ex.input_ids, ex.labels)
                                if l != IGNORE_INDEX])
    # Every assistant content string appears in the trained text...
    assert "I will inspect the directory first." in trained
    assert "Task complete." in trained
    assert "<tool_call>" in trained
    # ...and nothing from the environment or user does.
    assert "drwxr-xr-x" not in trained
    assert "Please list files" not in trained
    assert "<tool_response>" not in trained


def test_trained_labels_equal_input_ids_where_set(tokenizer):
    ex = tokenize_conversation(make_chat_conversation(), tokenizer)
    for tok, lab in zip(ex.input_ids, ex.labels):
        assert lab == IGNORE_INDEX or lab == tok


def test_stop_token_is_trained(tokenizer):
    ex = tokenize_conversation(make_chat_conversation(), tokenizer)
    trained = tokenizer.decode([t for t, l in zip(ex.input_ids, ex.labels)
                                if l != IGNORE_INDEX])
    assert "<|im_end|>" in trained


def test_trainable_fraction_sane(tokenizer):
    ex = tokenize_conversation(make_agentic_conversation(), tokenizer)
    assert 0.0 < ex.trainable_fraction < 1.0
