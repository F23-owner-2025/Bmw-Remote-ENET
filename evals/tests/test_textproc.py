from evalkit.textproc import (
    extract_final_answer_text,
    extract_mcq_answer,
    extract_numeric_answer,
    extract_python_code,
    extract_tool_calls,
    parse_number,
    strip_think,
)


def test_strip_think_closed_and_unclosed():
    assert strip_think("<think>secret</think>visible") == "visible"
    assert strip_think("before<think>never closed...") == "before"
    assert strip_think("plain") == "plain"


def test_extract_tool_calls_good():
    text = ('Let me check.\n<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}\n</tool_call>')
    calls, errors = extract_tool_calls(text)
    assert errors == []
    assert calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def test_extract_tool_calls_multiple_and_errors():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>not json</tool_call>'
            '<tool_call>{"arguments": {}}</tool_call>')
    calls, errors = extract_tool_calls(text)
    assert len(calls) == 1 and len(errors) == 2


def test_extract_tool_calls_unbalanced():
    _, errors = extract_tool_calls('<tool_call>{"name": "x", "arguments": {}}')
    assert any("unbalanced" in e for e in errors)


def test_extract_python_code_prefers_entry_point():
    text = ("Approach:\n```python\nhelper = 1\n```\nSolution:\n"
            "```python\ndef target():\n    return 2\n```\n")
    assert "def target" in extract_python_code(text, "target")
    # without entry point: last block
    assert "def target" in extract_python_code(text)


def test_extract_python_code_no_fence_returns_raw():
    assert extract_python_code("def f():\n    pass") == "def f():\n    pass"


def test_parse_number_forms():
    assert parse_number("5,624.32 dollars") == 5624.32
    assert parse_number("1/9") == 1 / 9
    assert parse_number("-5") == -5
    assert parse_number("9.81e4 Pa") == 98100.0
    assert parse_number("about 3.183 Hz") == 3.183
    assert parse_number("no numbers") is None
    # "1.5 m/s" must parse as 1.5, not as a fraction
    assert parse_number("1.5 m/s") == 1.5


def test_final_answer_extraction():
    assert extract_final_answer_text("blah\nFINAL ANSWER: 42 meters") == "42 meters"
    assert extract_final_answer_text("x\nFINAL ANSWER: 1\nFINAL ANSWER: 2") == "2"
    assert extract_final_answer_text("the answer is \\boxed{16}") == "16"
    assert extract_final_answer_text("nothing here") is None


def test_numeric_answer_with_fallback():
    assert extract_numeric_answer("steps...\nFINAL ANSWER: 352") == 352
    assert extract_numeric_answer("computing 2+2 gives 4") == 4  # last number
    assert extract_numeric_answer("no digits at all") is None


def test_mcq_answer():
    assert extract_mcq_answer("reasoning...\nFINAL ANSWER: B") == "B"
    assert extract_mcq_answer("FINAL ANSWER: (c)") == "C"
    assert extract_mcq_answer("I pick D because...") == "D"
    assert extract_mcq_answer("no letters") is None
