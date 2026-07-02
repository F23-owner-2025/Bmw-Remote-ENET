from evalkit.graders import (
    grade_code,
    grade_instruction,
    grade_mcq,
    grade_numeric,
    grade_tool_call,
)

# ------------------------------------------------------------------- code

CODE_TASK = {
    "id": "sq", "type": "code", "entry_point": "square",
    "test_code": "assert square(3) == 9\nassert square(-2) == 4\n",
}


def test_grade_code_pass():
    resp = "Here you go:\n```python\ndef square(x):\n    return x * x\n```"
    r = grade_code(CODE_TASK, resp)
    assert r.passed, r.details


def test_grade_code_wrong_logic():
    resp = "```python\ndef square(x):\n    return x + x\n```"
    r = grade_code(CODE_TASK, resp)
    assert not r.passed and "tests failed" in r.details


def test_grade_code_missing_entry_point():
    r = grade_code(CODE_TASK, "```python\ndef cube(x):\n    return x**3\n```")
    assert not r.passed and "square" in r.details


def test_grade_code_timeout():
    task = dict(CODE_TASK, timeout_s=2)
    r = grade_code(task, "```python\ndef square(x):\n    while True: pass\n```")
    assert not r.passed and "timed out" in r.details


# ------------------------------------------------------------------- numeric

def test_grade_numeric_within_tolerance():
    task = {"answer": 3.183, "rel_tol": 0.02}
    assert grade_numeric(task, "FINAL ANSWER: 3.18").passed
    assert not grade_numeric(task, "FINAL ANSWER: 3.5").passed
    assert not grade_numeric(task, "I cannot say").passed


def test_grade_numeric_exact_integers():
    task = {"answer": 352}
    assert grade_numeric(task, "so the total is\nFINAL ANSWER: 352").passed
    assert not grade_numeric(task, "FINAL ANSWER: 351").passed


# ------------------------------------------------------------------- mcq

def test_grade_mcq():
    task = {"answer": "B"}
    assert grade_mcq(task, "FINAL ANSWER: B").passed
    assert not grade_mcq(task, "FINAL ANSWER: A").passed
    assert not grade_mcq(task, "no answer").passed


# ------------------------------------------------------------------- tool_call

WEATHER_EXPECT = {"calls": [{"name": "get_weather",
                             "arguments_contain": {"city": "tokyo"},
                             "arguments_equal": {"units": "celsius"}}]}


def call_text(name, args_json):
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args_json}}}\n</tool_call>'


def test_tool_call_match():
    resp = "Checking.\n" + call_text("get_weather",
                                     '{"city": "Tokyo", "units": "celsius"}')
    r = grade_tool_call({"expect": WEATHER_EXPECT}, resp)
    assert r.passed, r.details


def test_tool_call_wrong_enum():
    resp = call_text("get_weather", '{"city": "Tokyo", "units": "fahrenheit"}')
    r = grade_tool_call({"expect": WEATHER_EXPECT}, resp)
    assert not r.passed and "units" in r.details


def test_tool_call_wrong_count():
    resp = (call_text("get_weather", '{"city": "Tokyo", "units": "celsius"}')
            + call_text("get_weather", '{"city": "Kyoto", "units": "celsius"}'))
    r = grade_tool_call({"expect": WEATHER_EXPECT}, resp)
    assert not r.passed and "expected 1" in r.details


def test_tool_call_parallel_order_independent():
    expect = {"calls": [
        {"name": "get_weather", "arguments_contain": {"city": "berlin"}},
        {"name": "get_weather", "arguments_contain": {"city": "madrid"}}]}
    resp = (call_text("get_weather", '{"city": "Madrid", "units": "celsius"}')
            + call_text("get_weather", '{"city": "Berlin", "units": "celsius"}'))
    assert grade_tool_call({"expect": expect}, resp).passed


def test_tool_call_malformed_json_fails():
    r = grade_tool_call({"expect": WEATHER_EXPECT},
                        "<tool_call>{bad json}</tool_call>")
    assert not r.passed and "malformed" in r.details


def test_tool_call_no_call_expected():
    task = {"expect": {"no_call": True}}
    assert grade_tool_call(task, "HTTP 404 means Not Found.").passed
    resp = call_text("get_weather", '{"city": "x", "units": "celsius"}')
    assert not grade_tool_call(task, resp).passed


def test_tool_call_clarifying_question():
    task = {"expect": {"no_call": True, "must_ask_question": True}}
    assert grade_tool_call(task, "What time should the meeting start?").passed
    assert not grade_tool_call(task, "I scheduled it for tomorrow noon.").passed


def test_tool_call_arguments_contain_on_list():
    task = {"expect": {"calls": [{"name": "run_commands",
                                  "arguments_contain": {"commands": "df"}}]}}
    resp = call_text("run_commands", '{"commands": ["df -h"]}')
    assert grade_tool_call(task, resp).passed


# ------------------------------------------------------------------- instruction

def test_instruction_bullets_and_words():
    task = {"checks": [{"kind": "bullet_count", "value": 3},
                       {"kind": "max_words", "value": 30}]}
    good = "- one\n- two\n- three"
    assert grade_instruction(task, good).passed
    assert not grade_instruction(task, "- one\n- two").passed


def test_instruction_json_only():
    task = {"checks": [{"kind": "json_only", "keys": ["name", "port", "debug"]}]}
    assert grade_instruction(
        task, '{"name": "dev", "port": 8080, "debug": true}').passed
    assert not grade_instruction(
        task, 'Sure! {"name": "dev", "port": 8080, "debug": true}').passed
    assert not grade_instruction(task, '{"name": "dev", "port": 8080}').passed


def test_instruction_case_and_markers():
    task = {"checks": [{"kind": "lowercase_only"}]}
    assert grade_instruction(task, "the kernel manages hardware.").passed
    assert not grade_instruction(task, "The kernel manages hardware.").passed

    task2 = {"checks": [{"kind": "must_include_exact", "values": ["ERROR_CODE_42"]},
                        {"kind": "ends_with", "value": "Done."}]}
    assert grade_instruction(task2, "It is ERROR_CODE_42 related. Done.").passed
    assert not grade_instruction(task2, "error_code_42 related. Done.").passed


def test_instruction_regex_count_and_paragraphs():
    task = {"checks": [{"kind": "regex_count", "pattern": "^\\d+\\.", "value": 3}]}
    assert grade_instruction(task, "1. a\n2. b\n3. c").passed
    assert not grade_instruction(task, "1. a\n2. b").passed

    task2 = {"checks": [{"kind": "paragraph_count", "value": 2}]}
    assert grade_instruction(task2, "First para.\n\nSecond para.").passed
    assert not grade_instruction(task2, "Only one paragraph.").passed


def test_instruction_forbidden_word_case_insensitive():
    task = {"checks": [{"kind": "must_not_include", "values": ["database"]}]}
    assert not grade_instruction(task, "A Database index is...").passed
    assert grade_instruction(task, "An index on a table speeds lookups.").passed
