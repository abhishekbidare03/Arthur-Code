"""
Adding a new function to a file that already exists.

This is the one common task search/replace handles badly, and it took a live
run to see why. To append with edit_file the model has to FIND the last lines
of the file and repeat them in REPLACE before its new code. Asked to "add a
second function three_sum to twosum.py", phi4-mini instead put the OLD
function in FIND and the NEW one in REPLACE -- which deletes the old one. The
gate blocked it three turns running and the file was never damaged, but the
model never recovered either, because the feedback it was getting was about
REPLACE blocks when the real answer was a different tool.

Appending is not a matching problem, so it should not be expressed as one.
What makes this tool safe is structural rather than checked: the old content
is a strict prefix of the new one, so nothing the model writes in CONTENT can
alter a byte that is already there.
"""

import pytest

from arthur import agent, tools
from arthur.llm_backend import MockBackend


EXISTING = ("def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "        seen[n] = i\n")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "twosum.py").write_text(EXISTING, encoding="utf-8")
    return str(tmp_path)


def scripted(monkeypatch, coder, critic="VERDICT: APPROVE\nREASON: fine."):
    from arthur import llm_backend, safety_gate
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(coder))
    for module in (llm_backend, safety_gate):
        monkeypatch.setattr(module, "get_critic_backend",
                            lambda name=None: MockBackend([critic]))


def collect(gen, answer=None):
    out, reply = [], None
    while True:
        try:
            event = gen.send(reply)
        except StopIteration:
            return out
        reply = None
        out.append(event)
        if event.kind == "approval_needed":
            reply = answer


APPEND_TURN = (
    'THOUGHT: adding a second function.\n'
    'CONFIDENCE: 0.9\n'
    'ACTION: append_file\n'
    'PATH: twosum.py\n'
    'CONTENT:\n```\n'
    'def three_sum(nums, target):\n'
    '    return []\n'
    '```'
)


# --- what it cannot do -------------------------------------------------------

def test_the_old_content_is_always_a_prefix_of_the_new():
    """The safety property, stated directly. Nothing else needs checking."""
    for addition in ("def f():\n    return 1\n", "", "\n\n\nx = 1", "}}}garbage"):
        assert tools.compute_append(EXISTING, addition).startswith(EXISTING)


def test_appending_never_removes_a_symbol():
    from arthur import patcher
    new = tools.compute_append(EXISTING, "def three_sum(nums, target):\n    return []\n")
    shape = patcher.analyze(EXISTING, new, "twosum.py")
    assert shape.removed_symbols == []
    assert shape.added_symbols == ["three_sum"]
    assert not shape.destructive


# --- spacing -----------------------------------------------------------------

def test_two_blank_lines_are_inserted_between_definitions():
    new = tools.compute_append(EXISTING, "def three_sum(nums, target):\n    return []\n")
    assert new == EXISTING + "\n\ndef three_sum(nums, target):\n    return []\n"
    compile(new, "<test>", "exec")


def test_existing_trailing_blank_lines_are_not_doubled():
    new = tools.compute_append(EXISTING + "\n\n", "def f():\n    return 1\n")
    assert "\n\n\n\n" not in new
    compile(new, "<test>", "exec")


def test_leading_blank_lines_in_the_addition_are_absorbed():
    new = tools.compute_append(EXISTING, "\n\n\ndef f():\n    return 1\n")
    assert "\n\n\n\n" not in new
    compile(new, "<test>", "exec")


def test_a_file_without_a_final_newline_is_handled():
    new = tools.compute_append("x = 1", "y = 2")
    assert new == "x = 1\n\n\ny = 2\n"
    compile(new, "<test>", "exec")


def test_appending_to_an_empty_file_adds_no_leading_blanks():
    assert tools.compute_append("", "x = 1") == "x = 1\n"


# --- through the agent -------------------------------------------------------

def test_append_lands_through_the_gate(repo, monkeypatch):
    scripted(monkeypatch, [APPEND_TURN, "FINAL: added three_sum."])
    stream = collect(agent.run("add three_sum to twosum.py", repo,
                               backend="mock", auto_approve=True))

    assert "gate_decision" in [e.kind for e in stream]
    written = open(f"{repo}/twosum.py").read()
    assert "def two_sum" in written          # the original survived
    assert "def three_sum" in written        # and the new one arrived
    compile(written, "<test>", "exec")


def test_appending_to_a_missing_file_is_refused(repo, monkeypatch):
    turn = APPEND_TURN.replace("PATH: twosum.py", "PATH: nope.py")
    scripted(monkeypatch, [turn, "FINAL: gave up."])
    stream = collect(agent.run("add to nope.py", repo, backend="mock",
                               auto_approve=True))

    observation = next(e for e in stream if e.kind == "observation")
    assert "does not exist" in observation.text
    assert "apply_patch" in observation.text


def test_editing_a_missing_file_says_so(repo, monkeypatch):
    """
    Not "the FIND text does not appear in the file", which is true and useless.
    Observed live: the model read that as a matching problem, rewrote the FIND
    block, and got the identical answer five turns running.
    """
    turn = ('ACTION: edit_file\nPATH: nope.py\n'
            'FIND:\n```\nanything\n```\nREPLACE:\n```\nsomething\n```')
    scripted(monkeypatch, [turn, "FINAL: gave up."])
    stream = collect(agent.run("edit nope.py", repo, backend="mock",
                               auto_approve=True))

    observation = next(e for e in stream if e.kind == "observation")
    assert "does not exist" in observation.text
    assert "apply_patch" in observation.text
    assert "FIND text does not appear" not in observation.text


def test_the_deleting_edit_is_told_to_append_instead(repo, monkeypatch):
    """
    The recovery path for the live loop: a FIND that removes a definition while
    REPLACE introduces a new one is an ADD expressed with the wrong tool, and
    saying so is more use than another lecture about REPLACE blocks.
    """
    deleting_edit = (
        'THOUGHT: adding three_sum.\n'
        'CONFIDENCE: 0.9\n'
        'ACTION: edit_file\n'
        'PATH: twosum.py\n'
        'FIND:\n```\n' + EXISTING.rstrip("\n") + '\n```\n'
        'REPLACE:\n```\n'
        'def three_sum(nums, target):\n'
        '    return []\n'
        '```'
    )
    scripted(monkeypatch, [deleting_edit, "FINAL: stopped."])
    stream = collect(agent.run("add three_sum", repo, backend="mock",
                               auto_approve=True), answer=None)

    gate = next(e for e in stream if e.kind == "gate_decision")
    assert gate.blocked_structurally

    observation = next(e for e in stream if e.kind == "observation")
    assert "append_file" in observation.text
    assert "def two_sum" in open(f"{repo}/twosum.py").read()   # never applied


# --- edits that achieve nothing ----------------------------------------------

def test_a_no_op_edit_is_refused_rather_than_reported_as_success(repo, monkeypatch):
    """
    An edit that changes nothing is not a success, and left alone it reads as
    one: the tool reports EDITED, the model announces the task is complete, and
    the file is untouched. Observed live -- told its REPLACE block had an
    indentation problem, the model fixed the indentation and dropped the
    docstring it was supposed to be adding, reproducing the file exactly.
    """
    unchanged = (
        'ACTION: edit_file\nPATH: twosum.py\n'
        'FIND:\n```\n    seen = {}\n```\n'
        'REPLACE:\n```\n    seen = {}\n```'
    )
    scripted(monkeypatch, [unchanged, "FINAL: all done!"])
    stream = collect(agent.run("change something", repo, backend="mock",
                               auto_approve=True))

    kinds = [e.kind for e in stream]
    assert "gate_decision" not in kinds          # never reached the gate
    observation = next(e for e in stream if e.kind == "observation")
    assert "exactly as it is" in observation.text
    assert stream[-1].files_changed == []        # and the run admits it


def test_dropping_the_anchor_line_is_named_precisely(repo, monkeypatch):
    """
    A REPLACE that lost the `def` line breaks the file, so it surfaces as "no
    longer valid Python" -- and answering that with advice about indentation
    sends the model to fix something that was never wrong.
    """
    dropped = (
        'ACTION: edit_file\nPATH: twosum.py\n'
        'FIND:\n```\ndef two_sum(nums, target):\n```\n'
        'REPLACE:\n```\n    """Return the indices."""\n```'
    )
    scripted(monkeypatch, [dropped, "FINAL: stopped."])
    stream = collect(agent.run("add a docstring", repo, backend="mock",
                               auto_approve=True), answer=None)

    observation = next(e for e in stream if e.kind == "observation")
    assert "def two_sum(nums, target):" in observation.text
    assert "would be deleted" in observation.text
    assert "INDENTATION" not in observation.text     # not the wrong diagnosis


# --- parsing -----------------------------------------------------------------

def test_the_append_form_parses():
    parsed = agent.parse_response(APPEND_TURN)
    assert parsed["action"] == "append_file"
    assert parsed["args"]["path"] == "twosum.py"
    assert parsed["args"]["content"] == "def three_sum(nums, target):\n    return []\n"


def test_append_is_listed_in_the_prompt():
    prompt = agent.build_system_prompt("/repo")
    assert "append_file" in prompt
    assert "ADDING a whole new function" in prompt
