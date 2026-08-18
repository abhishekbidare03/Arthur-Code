"""
Failing fast, and failing with something the model can act on.

Observed live, from a single `hey`:

    step 2  ACTION: search_code()   ERROR: missing required argument 'query'
    step 3  ACTION: search_code()   ERROR: missing required argument 'query'
    step 4  ACTION: search_code()   ERROR: missing required argument 'query'
    step 5  FAILED: the model repeated the same reply 3 times

The observation is accurate and useless. It names the missing key, which the
model already knew, and says nothing about the shape of a call that would
supply it -- so the model rewrote its THOUGHT each turn and re-sent the same
empty call underneath. This is the same failure that `edit_file` had before
observations started echoing the file back: for a small model the error
message is not a report, it is the only teaching material available.

Two changes here. The observation now hands back a line to copy, and the stall
detector compares CALLS rather than response text, so rewording the reasoning
no longer buys three more steps.
"""

import pytest

from arthur import agent, config
from arthur.llm_backend import MockBackend


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calculator.py").write_text("def divide(a, b):\n    return a / b\n",
                                            encoding="utf-8")
    return str(tmp_path)


def scripted(monkeypatch, turns):
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(turns))


def collect(gen):
    out = []
    while True:
        try:
            out.append(gen.send(None))
        except StopIteration:
            return out


def observations(stream):
    return [e.text for e in stream if e.kind == "observation"]


# --- the observation has to be actionable ------------------------------------

def test_an_argumentless_call_is_answered_with_a_line_to_copy(repo, monkeypatch):
    """Exactly the turn from the transcript above."""
    scripted(monkeypatch, ["THOUGHT: searching.\nACTION: search_code\nARGS: {}",
                           "FINAL: done."])
    stream = collect(agent.run("find the parser", repo, backend="mock"))

    first = observations(stream)[0]
    assert "search_code needs query" in first
    assert 'ARGS: {"query"' in first          # the copyable line


@pytest.mark.parametrize("action,expected", [
    ("search_code", '"query"'),
    ("read_file", '"path"'),
    ("run_command", '"command"'),
])
def test_every_read_tool_has_a_worked_call(action, expected):
    assert expected in agent.READ_USAGE_HINT[action]
    assert agent.READ_USAGE_HINT[action].startswith(f"ACTION: {action}")


def test_the_hints_are_calls_the_agent_would_actually_accept():
    """A hint that does not parse would teach the model a broken form."""
    for action, hint in agent.READ_USAGE_HINT.items():
        parsed = agent.parse_response(f"THOUGHT: x.\n{hint}")
        assert parsed["action"] == action, hint
        for key in agent.READ_REQUIRED_ARGS.get(action, ()):
            assert parsed["args"].get(key), (action, key)


def test_a_blank_argument_counts_as_missing(repo, monkeypatch):
    """`ARGS: {"query": ""}` is the same dead end as sending none."""
    scripted(monkeypatch, ['THOUGHT: hunting.\nACTION: search_code\n'
                           'ARGS: {"query": "  "}',
                           "FINAL: gave up."])
    stream = collect(agent.run("find it", repo, backend="mock"))
    assert "search_code needs query" in observations(stream)[0]


# --- stalling on a reworded thought ------------------------------------------

def test_repeating_a_call_aborts_even_when_the_wording_changes(repo, monkeypatch):
    """
    The case the text-level detector misses. Three identical calls under three
    different THOUGHTs is a stall, and burning the rest of the cap to confirm
    it wastes a minute of a 4GB card.
    """
    monkeypatch.setattr(config, "MAX_STEPS", 12)
    scripted(monkeypatch, [
        f'THOUGHT: let me try approach {i} instead.\n'
        f'ACTION: read_file\nARGS: {{"path": "calculator.py"}}'
        for i in range(12)
    ])
    stream = collect(agent.run("read it", repo, backend="mock"))

    assert stream[-1].kind == "run_failed"
    assert "same arguments" in stream[-1].error
    steps = [e for e in stream if e.kind == "step_started"]
    assert len(steps) == config.REPEAT_LIMIT


def test_two_identical_calls_are_not_yet_a_stall(repo, monkeypatch):
    """Retrying once is normal. The limit exists to stop a loop, not a retry."""
    read = 'THOUGHT: reading.\nACTION: read_file\nARGS: {"path": "calculator.py"}'
    scripted(monkeypatch, [read, read, "FINAL: read it twice, all fine."])
    stream = collect(agent.run("read it", repo, backend="mock"))
    assert stream[-1].kind == "final"


def test_different_arguments_never_count_as_a_stall(repo, monkeypatch):
    """Reading three files in a row is progress, not a loop."""
    scripted(monkeypatch, [
        f'THOUGHT: checking.\nACTION: read_file\nARGS: {{"path": "f{i}.py"}}'
        for i in range(3)
    ] + ["FINAL: looked at all three."])
    stream = collect(agent.run("survey the repo", repo, backend="mock"))
    assert stream[-1].kind == "final"


# --- the cap -----------------------------------------------------------------

def test_the_step_cap_is_short_enough_to_watch():
    """
    Not a style assertion. At ~11 tok/s a step is ten-odd seconds, and the cap
    is the worst case a user sits through before getting the terminal back.
    """
    assert config.MAX_STEPS <= 8


def test_the_cap_leaves_room_for_a_real_task():
    """read -> edit -> verify -> final, with a failed match somewhere, is 5."""
    assert config.MAX_STEPS >= 6
