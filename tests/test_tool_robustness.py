"""
Tools must never crash the run, whatever the model hands them.

Every argument a tool receives was invented by a language model, so bad paths,
missing keys and wrong types are ordinary inputs rather than exceptional ones.
Each belongs in an observation the model can act on -- never in a traceback
that takes down an interactive session the user has been building context in.

The directory case here is not hypothetical: phi4-mini answered "create
twosum.py" with `path: "."`, and the resulting PermissionError killed a live
session mid-task.
"""

import os

import pytest

from arthur import agent, events as ev, tools


# --- the crash that started this ---------------------------------------------

def test_directory_as_path_is_refused_not_raised(tmp_path):
    """`path: "."` -> PermissionError on Windows, IsADirectoryError on POSIX."""
    out = tools.apply_patch({"path": ".", "new_content": "x = 1\n"}, str(tmp_path))
    assert out.startswith("ERROR:")
    assert "directory" in out


def test_current_content_refuses_a_directory(tmp_path):
    with pytest.raises(tools.BadTarget, match="directory"):
        tools.current_content({"path": "."}, str(tmp_path))


def test_subdirectory_as_path_is_refused(tmp_path):
    (tmp_path / "src").mkdir()
    out = tools.apply_patch({"path": "src", "new_content": "x = 1\n"}, str(tmp_path))
    assert out.startswith("ERROR:")


def test_empty_path_is_refused(tmp_path):
    out = tools.apply_patch({"path": "", "new_content": "x\n"}, str(tmp_path))
    assert out.startswith("ERROR:")
    assert "filename" in out


def test_error_message_tells_the_model_what_to_do(tmp_path):
    """A refusal the model can't act on just becomes a loop."""
    out = tools.apply_patch({"path": ".", "new_content": "x\n"}, str(tmp_path))
    assert ".py" in out          # shows the shape of a valid answer


def test_writing_a_real_file_still_works(tmp_path):
    out = tools.apply_patch({"path": "twosum.py", "new_content": "x = 1\n"}, str(tmp_path))
    assert out.startswith("WROTE")
    assert (tmp_path / "twosum.py").read_text(encoding="utf-8") == "x = 1\n"


def test_creating_in_a_new_subdirectory_still_works(tmp_path):
    out = tools.apply_patch({"path": "pkg/mod.py", "new_content": "x = 1\n"}, str(tmp_path))
    assert out.startswith("WROTE")
    assert (tmp_path / "pkg" / "mod.py").exists()


# --- every tool degrades to an observation -----------------------------------

@pytest.mark.parametrize("name", sorted(tools.DISPATCH))
def test_no_tool_raises_on_empty_args(name, tmp_path):
    """A missing key must read back as an error, not a KeyError."""
    result = tools.DISPATCH[name]({}, str(tmp_path))
    assert isinstance(result, str)


@pytest.mark.parametrize("name", sorted(tools.DISPATCH))
def test_no_tool_raises_on_a_directory_path(name, tmp_path):
    args = {"path": ".", "query": "x", "command": "echo hi",
            "new_content": "x\n", "find": "a", "replace": "b"}
    result = tools.DISPATCH[name](args, str(tmp_path))
    assert isinstance(result, str)


@pytest.mark.parametrize("name", ["read_file", "edit_file", "apply_patch", "list_dir"])
def test_no_tool_raises_on_a_path_outside_the_repo(name, tmp_path):
    args = {"path": "../../../etc/passwd", "new_content": "x\n",
            "find": "a", "replace": "b"}
    result = tools.DISPATCH[name](args, str(tmp_path))
    assert result.startswith("ERROR:")


def test_edit_file_on_a_directory_is_refused(tmp_path):
    out = tools.edit_file({"path": ".", "find": "a", "replace": "b"}, str(tmp_path))
    assert out.startswith("ERROR:")


# --- the loop survives it ----------------------------------------------------

def collect(gen, answer=None):
    out, reply = [], None
    while True:
        try:
            event = gen.send(reply)
        except StopIteration:
            return out
        reply = None
        out.append(event)
        if isinstance(event, ev.ApprovalNeeded):
            reply = answer


def scripted(monkeypatch, script):
    from arthur.llm_backend import MockBackend
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(script))


def test_directory_patch_does_not_kill_the_run(tmp_path, monkeypatch):
    """
    The exact live failure: the model targets '.', and before this fix the
    PermissionError escaped the generator and ended the whole session.
    """
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n",
                                            encoding="utf-8")
    scripted(monkeypatch, [
        "ACTION: apply_patch\nPATH: .\nCONTENT:\n```\nx = 1\n```",
        "FINAL: gave up on that one.",
    ])

    stream = collect(agent.run("create twosum.py", str(tmp_path), backend="mock"))

    assert stream[-1].kind == "final"                      # ran to completion
    observation = next(e for e in stream if e.kind == "observation")
    assert "ERROR" in observation.text
    assert "diff_ready" not in [e.kind for e in stream]    # never reached the gate


def test_unexpected_tool_exception_becomes_an_observation(tmp_path, monkeypatch):
    """Defence in depth: a tool raising something nobody anticipated."""
    def explode(args, root):
        raise TypeError("args were not the shape anyone expected")

    monkeypatch.setitem(tools.DISPATCH, "list_dir", explode)
    scripted(monkeypatch, ['ACTION: list_dir\nARGS: {"path": "."}',
                           "FINAL: moving on."])

    stream = collect(agent.run("look around", str(tmp_path), backend="mock"))

    assert stream[-1].kind == "final"
    observation = next(e for e in stream if e.kind == "observation")
    assert "TypeError" in observation.text
