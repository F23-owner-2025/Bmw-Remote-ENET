import json

from evalkit.agent import check_goal, run_episode
from evalkit.client import ScriptedClient
from evalkit.shell import VirtualShell


def tool_call(commands):
    payload = json.dumps({"name": "run_commands",
                          "arguments": {"commands": commands}})
    return f"Working on it.\n<tool_call>\n{payload}\n</tool_call>"


TASK = {
    "id": "demo", "type": "agentic", "suite": "planning_agentic", "max_turns": 5,
    "prompt": "Create hello.txt containing 'hi'.",
    "initial_files": {},
    "goals": [{"kind": "file_contains", "path": "hello.txt", "text": "hi"}],
}


def test_successful_episode():
    client = ScriptedClient(responses=[
        tool_call(["echo hi > hello.txt"]),
        "Done — created hello.txt containing 'hi'.",
    ])
    result = run_episode(TASK, client)
    assert result.passed, result.details
    assert result.turns_used == 2
    # tool output went back as a tool turn wrapped in <tool_response>
    tool_turns = [m for m in result.transcript if m["role"] == "tool"]
    assert len(tool_turns) == 1
    assert tool_turns[0]["content"].startswith("<tool_response>")


def test_multi_step_read_then_write():
    task = {
        "id": "ver", "type": "agentic", "max_turns": 5,
        "prompt": "Copy the version from version.txt into out.txt.",
        "initial_files": {"version.txt": "3.1.4\n"},
        "goals": [{"kind": "file_contains", "path": "out.txt", "text": "3.1.4"}],
    }

    def respond(messages):
        last = messages[-1]
        if last["role"] == "user":
            return tool_call(["cat version.txt"])
        if "3.1.4" in last["content"] and "out.txt" not in str(messages):
            pass
        # after seeing the version, write it out; after that, finish
        wrote = any("out.txt" in m["content"] for m in messages
                    if m["role"] == "assistant")
        if not wrote:
            return tool_call(["echo '3.1.4' > out.txt"])
        return "Done."

    result = run_episode(task, ScriptedClient(fn=respond))
    assert result.passed, result.details


def test_goal_failure_reported():
    client = ScriptedClient(responses=[
        tool_call(["echo wrong > other.txt"]),
        "All done!",
    ])
    result = run_episode(TASK, client)
    assert not result.passed
    assert "hello.txt" in result.details
    assert result.goals_passed == 0 and result.goals_total == 1


def test_turn_budget_exhaustion():
    client = ScriptedClient(fn=lambda m: tool_call(["ls"]))  # loops forever
    result = run_episode(TASK, client)
    assert not result.passed
    assert result.turns_used == TASK["max_turns"]
    assert "turn budget exhausted" in result.details


def test_malformed_tool_call_gets_error_response_and_can_recover():
    client = ScriptedClient(responses=[
        "<tool_call>{not valid json}</tool_call>",
        tool_call(["echo hi > hello.txt"]),
        "Done.",
    ])
    result = run_episode(TASK, client)
    assert result.passed, result.details
    error_turns = [m for m in result.transcript
                   if m["role"] == "tool" and "ERROR" in m["content"]]
    assert len(error_turns) == 1


def test_unknown_tool_name_reported_to_model():
    client = ScriptedClient(responses=[
        '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>',
        tool_call(["echo hi > hello.txt"]),
        "Done.",
    ])
    result = run_episode(TASK, client)
    assert result.passed
    assert any("unknown tool" in m["content"] for m in result.transcript
               if m["role"] == "tool")


def test_think_blocks_do_not_hide_tool_calls():
    client = ScriptedClient(responses=[
        "<think>planning...</think>" + tool_call(["echo hi > hello.txt"]),
        "<think>done now</think>Finished.",
    ])
    result = run_episode(TASK, client)
    assert result.passed, result.details


def test_check_goal_kinds():
    s = VirtualShell(files={"a.txt": "alpha beta\n"}, dirs=["d"])
    assert check_goal({"kind": "file_exists", "path": "a.txt"}, s) is None
    assert check_goal({"kind": "file_exists", "path": "b.txt"}, s) is not None
    assert check_goal({"kind": "file_absent", "path": "b.txt"}, s) is None
    assert check_goal({"kind": "dir_exists", "path": "d"}, s) is None
    assert check_goal({"kind": "file_contains", "path": "a.txt",
                       "text": "beta"}, s) is None
    assert check_goal({"kind": "file_contains", "path": "a.txt",
                       "text": "gamma"}, s) is not None
    assert check_goal({"kind": "file_not_contains", "path": "a.txt",
                       "text": "gamma"}, s) is None
    assert check_goal({"kind": "file_equals", "path": "a.txt",
                       "text": "alpha beta"}, s) is None
