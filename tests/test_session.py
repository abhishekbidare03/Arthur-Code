"""
Interactive session and transcript tests.

The behaviour worth pinning down is the thing a one-shot script can't do:
conversation state surviving across tasks. Plus the slash commands, which are
handled locally and must never reach the model.
"""

import json
import os

import pytest

from arthur import config, events as ev, transcript
from arthur.session import Session, _handle_command, expand_mentions
from arthur.render import TerminalRenderer


@pytest.fixture
def session(tmp_path):
    return Session(repo=str(tmp_path), backend="mock")


@pytest.fixture
def renderer():
    return TerminalRenderer()


def run_cmd(line, session, renderer):
    return _handle_command(line, session, renderer)


# --- session state -----------------------------------------------------------

def test_history_is_the_thing_that_persists(session):
    """A one-shot script cannot do this; it's the whole point of the session."""
    session.messages.extend([{"role": "system", "content": "protocol"},
                             {"role": "user", "content": "first task"}])
    assert session.context_tokens() > 0
    assert len(session.messages) == 2


def test_clear_forgets_conversation_but_keeps_the_repo(session, renderer):
    repo_before = session.repo
    session.messages.append({"role": "user", "content": "something"})
    session.tasks_run = 3

    run_cmd("/clear", session, renderer)

    assert session.messages == []
    assert session.tasks_run == 0
    assert session.repo == repo_before


# --- slash commands ----------------------------------------------------------

def test_exit_ends_the_loop(session, renderer):
    assert run_cmd("/exit", session, renderer) is False
    assert run_cmd("/quit", session, renderer) is False


def test_other_commands_continue_the_loop(session, renderer):
    for cmd in ("/help", "/context", "/diff", "/runs", "/model", "/backend"):
        assert run_cmd(cmd, session, renderer) is True


def test_unknown_command_does_not_kill_the_session(session, renderer, capsys):
    assert run_cmd("/nonsense", session, renderer) is True
    assert "unknown command" in capsys.readouterr().out


def test_auto_toggles(session, renderer):
    assert session.auto_approve is False
    run_cmd("/auto", session, renderer)
    assert session.auto_approve is True
    run_cmd("/auto", session, renderer)
    assert session.auto_approve is False


def test_backend_switch_updates_session_and_config(session, renderer, monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "mock", raising=False)
    run_cmd("/backend gemini", session, renderer)
    assert session.backend == "gemini"
    assert config.BACKEND == "gemini"


def test_invalid_backend_is_rejected(session, renderer):
    run_cmd("/backend nonsense", session, renderer)
    assert session.backend == "mock"


def test_model_change_targets_the_active_backend(session, renderer, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_MODEL", "before", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "untouched", raising=False)
    session.backend = "ollama"

    run_cmd("/model qwen2.5-coder:7b", session, renderer)

    assert config.OLLAMA_MODEL == "qwen2.5-coder:7b"
    assert config.GEMINI_MODEL == "untouched"


def test_repo_change_clears_stale_conversation(session, renderer, tmp_path):
    """Old context is about the old code, so keeping it would mislead."""
    other = tmp_path / "other"
    other.mkdir()
    session.messages.append({"role": "user", "content": "about the old repo"})

    run_cmd(f"/repo {other}", session, renderer)

    assert session.repo == str(other)
    assert session.messages == []


def test_repo_change_to_missing_dir_is_refused(session, renderer, tmp_path):
    before = session.repo
    run_cmd(f"/repo {tmp_path / 'nope'}", session, renderer)
    assert session.repo == before


# --- @file mentions ----------------------------------------------------------

def test_mention_inlines_the_file(tmp_path):
    (tmp_path / "utils.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    out = expand_mentions("explain @utils.py please", str(tmp_path))
    assert "def helper():" in out
    assert "explain @utils.py please" in out


def test_mention_of_a_missing_file_is_ignored(tmp_path):
    text = "look at @ghost.py"
    assert expand_mentions(text, str(tmp_path)) == text


def test_text_without_mentions_is_unchanged(tmp_path):
    text = "just a normal request"
    assert expand_mentions(text, str(tmp_path)) == text


def test_bare_at_sign_is_not_a_mention(tmp_path):
    assert expand_mentions("email me @ home", str(tmp_path)) == "email me @ home"


# --- transcripts -------------------------------------------------------------

def sample_events():
    return [
        ev.RunStarted(task="fix a bug", repo_root="/repo", backend="mock", model="scripted"),
        ev.Thought(text="looking at it"),
        ev.GateDecision(self_confidence=0.9, critic_verdict="APPROVE",
                        critic_reason="fine", score=0.95, threshold=0.65),
        ev.Final(summary="done", steps_used=2, files_changed=["a.py"], elapsed=1.5),
    ]


def test_transcript_round_trips(tmp_path):
    path = str(tmp_path / "run.json")
    transcript.save(path, sample_events(), meta={"task": "fix a bug"})

    restored, meta = transcript.load(path)
    assert [e.kind for e in restored] == ["run_started", "thought", "gate_decision", "final"]
    assert restored[-1].files_changed == ["a.py"]
    assert meta["task"] == "fix a bug"


def test_replay_prints_without_calling_a_model(tmp_path, capsys):
    path = str(tmp_path / "run.json")
    transcript.save(path, sample_events(), meta={"task": "fix a bug"})

    rc = transcript.replay(path, TerminalRenderer())
    out = capsys.readouterr().out

    assert rc == 0
    assert "GATE" in out and "DONE" in out
    assert "replaying" in out


def test_replay_reports_failure_as_nonzero(tmp_path):
    path = str(tmp_path / "run.json")
    transcript.save(path, [ev.RunStarted(task="t"), ev.RunFailed(error="boom", step=1)])
    assert transcript.replay(path, TerminalRenderer()) == 1


def test_unknown_event_kinds_are_skipped_not_fatal(tmp_path):
    """A transcript from a newer build should still mostly replay."""
    path = tmp_path / "run.json"
    payload = {
        "arthur_version": "99.0.0",
        "meta": {},
        "events": [ev.Thought(text="kept").to_dict(),
                   {"kind": "from_the_future", "wat": 1}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored, meta = transcript.load(str(path))
    assert len(restored) == 1
    assert meta["_skipped_events"] == 1


def test_non_transcript_json_is_rejected(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"something": "else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an arthur transcript"):
        transcript.load(str(path))


def test_run_filename_is_scannable():
    path = transcript.new_path("Fix the divide by zero bug")
    name = os.path.basename(path)
    assert name.endswith(".json")
    assert "fix-the-divide" in name


def test_listing_an_empty_dir_is_not_an_error(tmp_path):
    assert transcript.list_runs(str(tmp_path / "nothing-here")) == []
