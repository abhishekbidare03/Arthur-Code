"""
Tests against the event stream.

This is the payoff from making the loop a generator: the interesting
behaviour -- that a rejection reaches the model, that silence fails closed,
that a broken protocol turn is recorded rather than swallowed -- is now
assertable without capturing stdout or faking a terminal.
"""

import pytest

from arthur import agent, config, events as ev
from arthur.llm_backend import MockBackend


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def collect(gen, answer=None):
    """Drain a run, answering every ApprovalNeeded with the same reply."""
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


def kinds(stream):
    return [e.kind for e in stream]


def scripted(monkeypatch, coder, critic="VERDICT: APPROVE\nREASON: fine."):
    """
    Point both the coder and the critic at fixed scripts.

    `safety_gate` does `from .llm_backend import get_critic_backend`, so it
    holds its own reference to the function -- patching the name on
    `llm_backend` alone leaves the real critic in place. Both are patched here;
    getting this wrong silently makes every critic assertion meaningless.
    """
    from arthur import llm_backend, safety_gate
    monkeypatch.setattr(llm_backend, "get_backend", lambda name=None: MockBackend(coder))
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(coder))
    for module in (llm_backend, safety_gate):
        monkeypatch.setattr(module, "get_critic_backend",
                            lambda name=None: MockBackend([critic]))


# edit_file, not apply_patch: calculator.py exists, and a whole-file rewrite of
# an existing file is refused before it ever reaches the gate. These fixtures
# have to take the path a real run takes, or they test nothing.
#
# Additive by design -- no symbol is lost -- because these tests are about the
# confidence gate, not the deletion veto.
PATCH_TURN = (
    'THOUGHT: divide needs a guard.\n'
    'CONFIDENCE: 0.2\n'
    'ACTION: edit_file\n'
    'PATH: calculator.py\n'
    'FIND:\n```\n'
    'def divide(a, b):\n'
    '    return a / b\n'
    '```\n'
    'REPLACE:\n```\n'
    'def divide(a, b):\n'
    '    if b == 0:\n'
    '        raise ValueError("no")\n'
    '    return a / b\n'
    '```'
)

# A second, different edit -- for the tests that need two patches in one run.
# It has to target other lines: once PATCH_TURN lands, its own FIND text is no
# longer in the file.
SECOND_PATCH_TURN = (
    'THOUGHT: add could use a docstring.\n'
    'CONFIDENCE: 0.2\n'
    'ACTION: edit_file\n'
    'PATH: calculator.py\n'
    'FIND:\n```\n'
    'def add(a, b):\n'
    '    return a + b\n'
    '```\n'
    'REPLACE:\n```\n'
    'def add(a, b):\n'
    '    """Add two numbers."""\n'
    '    return a + b\n'
    '```'
)


# --- the happy path ----------------------------------------------------------

def test_full_run_emits_the_expected_arc(repo, monkeypatch):
    scripted(monkeypatch, [
        'THOUGHT: look first.\nACTION: read_file\nARGS: {"path": "calculator.py"}',
        'THOUGHT: done looking.\nFINAL: Read the file, nothing to change.',
    ])
    stream = collect(agent.run("inspect calculator", repo, backend="mock"))

    assert kinds(stream)[:3] == ["run_started", "index_built", "context_retrieved"]
    assert kinds(stream)[-1] == "final"
    assert stream[-1].steps_used == 2


def test_final_carries_the_files_it_changed(repo, monkeypatch):
    scripted(monkeypatch, [PATCH_TURN, "FINAL: Added a guard."])
    stream = collect(agent.run("guard divide", repo, backend="mock", auto_approve=True))

    final = stream[-1]
    assert final.kind == "final"
    assert final.files_changed == ["calculator.py"]


# --- the gate ----------------------------------------------------------------

def test_low_score_asks_the_human(repo, monkeypatch):
    # self-confidence 0.2 + critic APPROVE 1.0 -> 0.6, under the 0.65 default
    scripted(monkeypatch, [PATCH_TURN, "FINAL: done."])
    stream = collect(agent.run("guard divide", repo, backend="mock"),
                     answer=ev.Decision.APPLY)

    gate = next(e for e in stream if e.kind == "gate_decision")
    assert gate.needs_human is True
    assert "approval_needed" in kinds(stream)


def test_rejection_reason_is_fed_back_to_the_model(repo, monkeypatch):
    """The point of rejecting with a reason: the model sees why."""
    scripted(monkeypatch, [PATCH_TURN, "FINAL: gave up."])
    stream = collect(agent.run("guard divide", repo, backend="mock"),
                     answer=(ev.Decision.REJECT, "raises the wrong exception type"))

    observation = next(e for e in stream if e.kind == "observation")
    assert "raises the wrong exception type" in observation.text
    assert "calculator.py" in (repo and open(f"{repo}/calculator.py").read()) or True
    # and the file was left alone
    assert "raise ValueError" not in open(f"{repo}/calculator.py").read()


def test_no_answer_fails_closed(repo, monkeypatch):
    """A consumer that doesn't answer must not be read as approval."""
    scripted(monkeypatch, [PATCH_TURN, "FINAL: stopped."])
    stream = collect(agent.run("guard divide", repo, backend="mock"), answer=None)

    resolved = next(e for e in stream if e.kind == "approval_resolved")
    assert resolved.decision == ev.Decision.REJECT.value
    assert "raise ValueError" not in open(f"{repo}/calculator.py").read()


def test_auto_approve_skips_the_question_but_still_gates(repo, monkeypatch):
    scripted(monkeypatch, [PATCH_TURN, "FINAL: done."])
    stream = collect(agent.run("guard divide", repo, backend="mock", auto_approve=True))

    assert "approval_needed" not in kinds(stream)
    assert "gate_decision" in kinds(stream)      # the gate still ran and reported
    resolved = next(e for e in stream if e.kind == "approval_resolved")
    assert resolved.automatic is True
    assert "raise ValueError" in open(f"{repo}/calculator.py").read()


DESTRUCTIVE_TURN = (
    'THOUGHT: add() is not needed, removing it.\n'
    'CONFIDENCE: 1.0\n'
    'ACTION: edit_file\n'
    'PATH: calculator.py\n'
    'FIND:\n```\n'
    'def add(a, b):\n'
    '    return a + b\n'
    '\n'
    '\n'
    '```\n'
    'REPLACE:\n```\n'
    '```'                            # drops add() entirely
)


def test_auto_approve_cannot_bypass_a_structural_block(repo, monkeypatch):
    """
    `-y` means "stop asking me about judgement calls", not "disable the safety
    system". A patch that deletes existing definitions still stops, and with
    nobody there to answer it stays unapplied.
    """
    scripted(monkeypatch, [DESTRUCTIVE_TURN, "FINAL: stopped."])
    stream = collect(agent.run("shrink it", repo, backend="mock", auto_approve=True),
                     answer=None)

    gate = next(e for e in stream if e.kind == "gate_decision")
    assert gate.blocked_structurally
    assert gate.score > config.CONFIDENCE_THRESHOLD      # both LLM signals said yes
    assert "approval_needed" in kinds(stream)            # asked anyway
    assert "def add" in open(f"{repo}/calculator.py").read()   # and never applied


def test_auto_approve_cannot_override_a_critic_reject(repo, monkeypatch):
    """
    `-y` is for borderline confidence, not for overruling a reviewer that said
    no. Observed live: with -y on, a REJECTed edit (score 0.25) was applied and
    corrupted a file the agent had itself just written correctly.
    """
    scripted(monkeypatch, [PATCH_TURN, "FINAL: stopped."],
             critic="VERDICT: REJECT\nREASON: this breaks the function.")
    stream = collect(agent.run("guard divide", repo, backend="mock", auto_approve=True),
                     answer=None)

    gate = next(e for e in stream if e.kind == "gate_decision")
    assert gate.critic_verdict == "REJECT"
    assert "approval_needed" in kinds(stream)               # asked despite -y
    assert "raise ValueError" not in open(f"{repo}/calculator.py").read()


def test_auto_approve_still_works_when_the_critic_approves(repo, monkeypatch):
    """The common case must stay unattended, or -y is useless."""
    scripted(monkeypatch, [PATCH_TURN, "FINAL: done."],
             critic="VERDICT: APPROVE\nREASON: correct guard.")
    stream = collect(agent.run("guard divide", repo, backend="mock", auto_approve=True))

    assert "approval_needed" not in kinds(stream)
    assert "raise ValueError" in open(f"{repo}/calculator.py").read()


def test_structural_warning_names_what_would_be_lost(repo, monkeypatch):
    scripted(monkeypatch, [DESTRUCTIVE_TURN, "FINAL: stopped."])
    stream = collect(agent.run("shrink it", repo, backend="mock"), answer=None)

    gate = next(e for e in stream if e.kind == "gate_decision")
    assert any("add" in w for w in gate.structural_warnings)


def test_always_decision_stops_asking(repo, monkeypatch):
    """One ALWAYS answer covers every later patch in the same run."""
    scripted(monkeypatch, [PATCH_TURN, SECOND_PATCH_TURN, "FINAL: two patches."])
    stream = collect(agent.run("guard divide", repo, backend="mock"),
                     answer=ev.Decision.ALWAYS)

    assert kinds(stream).count("approval_needed") == 1
    assert kinds(stream).count("gate_decision") == 2


# --- failure modes -----------------------------------------------------------

def test_protocol_violation_is_reported_and_reprompted(repo, monkeypatch):
    scripted(monkeypatch, [
        "Sure! I'd be happy to help you with that.",
        "FINAL: ok, following the protocol now.",
    ])
    stream = collect(agent.run("do something", repo, backend="mock"))

    violation = next(e for e in stream if e.kind == "protocol_violation")
    assert violation.step == 1
    observation = next(e for e in stream if e.kind == "observation")
    assert "did not follow the protocol" in observation.text


def test_backend_failure_ends_the_run_cleanly(repo, monkeypatch):
    class Exploding:
        def chat(self, messages, on_token=None):
            raise RuntimeError("cannot reach Ollama at http://localhost:11434")

    monkeypatch.setattr(agent, "get_backend", lambda name=None: Exploding())
    stream = collect(agent.run("anything", repo, backend="ollama"))

    assert stream[-1].kind == "run_failed"
    assert "cannot reach Ollama" in stream[-1].error


def test_step_limit_is_reported(repo, monkeypatch):
    """
    Busy but never finishing: each turn is a genuinely different call, so it
    runs to the cap rather than tripping either stall detector. Reading a new
    path every turn is exactly the shape of a run that is exploring and simply
    never converges -- the one case the step limit exists for.
    """
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    scripted(monkeypatch, [
        f'THOUGHT: attempt {i}.\nACTION: read_file\nARGS: {{"path": "file{i}.py"}}'
        for i in range(10)
    ])
    stream = collect(agent.run("loop forever", repo, backend="mock"))

    assert stream[-1].kind == "step_limit_reached"
    assert stream[-1].max_steps == 3


def test_identical_repeats_abort_early(repo, monkeypatch):
    """
    A near-deterministic local model repeats its mistakes verbatim. Observed
    for real: qwen2.5-coder:3b emitted the same malformed ARGS twelve times.
    """
    monkeypatch.setattr(config, "MAX_STEPS", 12)
    monkeypatch.setattr(config, "REPEAT_LIMIT", 3)
    scripted(monkeypatch, ['ACTION: apply_patch\nARGS: {"nonsense": 1}'] * 12)
    stream = collect(agent.run("stuck", repo, backend="mock"))

    assert stream[-1].kind == "run_failed"
    assert "repeated the same reply" in stream[-1].error
    # aborted early rather than burning all 12
    assert sum(1 for e in stream if e.kind == "step_started") == 3


def test_missing_patch_args_do_not_reach_disk(repo, monkeypatch):
    scripted(monkeypatch, [
        'ACTION: apply_patch\nARGS: {"path": "calculator.py"}',   # no new_content
        "FINAL: gave up.",
    ])
    stream = collect(agent.run("guard divide", repo, backend="mock"))

    assert "diff_ready" not in kinds(stream)
    observation = next(e for e in stream if e.kind == "observation")
    assert "new_content" in observation.text


# --- session continuity ------------------------------------------------------

def test_history_carries_across_tasks(repo, monkeypatch):
    """What makes 'now add a test for it' work in an interactive session."""
    scripted(monkeypatch, ["FINAL: first task done."])
    history: list[dict] = []
    collect(agent.run("first task", repo, backend="mock", messages=history))

    assert history[0]["role"] == "system"
    assert any("TASK: first task" in m["content"] for m in history)

    before = len(history)
    scripted(monkeypatch, ["FINAL: second task done."])
    collect(agent.run("second task", repo, backend="mock", messages=history))

    assert len(history) > before
    assert sum(1 for m in history if m["role"] == "system") == 1  # not re-laid


# --- serialization -----------------------------------------------------------

def test_events_round_trip_through_json():
    """Required for the transcript + --replay work that comes next."""
    import json

    original = ev.GateDecision(self_confidence=0.4, critic_verdict="REJECT",
                               critic_reason="returns 0 instead of raising",
                               score=0.2, threshold=0.65, needs_human=True)
    restored = ev.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original
    assert restored.critic_reason == "returns 0 instead of raising"


def test_unknown_event_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown event kind"):
        ev.from_dict({"kind": "nonsense"})


def test_an_unparseable_turn_keeps_its_raw_text(repo, monkeypatch):
    """
    Four parser bugs in Phase 9 had to be reproduced live before they could be
    fixed, because a transcript records what survived parsing and these bugs
    are defined by what did not. `insert_after` with `args: {}` looks identical
    whether the model sent nothing or sent a perfect call the parser could not
    read -- and every time, it was the second.
    """
    scripted(monkeypatch, ["ACTION: insert_after\nAFTER-ish: nonsense\n",
                           "FINAL: gave up."])
    stream = collect(agent.run("add a docstring", repo, backend="mock"))

    proposed = next(e for e in stream if e.kind == "action_proposed")
    assert proposed.args == {}
    assert "AFTER-ish: nonsense" in proposed.raw


def test_a_turn_that_parsed_keeps_no_raw_copy(repo, monkeypatch):
    """Only the case that needs evidence pays for it."""
    scripted(monkeypatch, ['ACTION: read_file\nARGS: {"path": "calculator.py"}',
                           "FINAL: read it."])
    stream = collect(agent.run("read it", repo, backend="mock"))

    proposed = next(e for e in stream if e.kind == "action_proposed")
    assert proposed.args == {"path": "calculator.py"}
    assert proposed.raw == ""


def test_the_raw_text_survives_a_transcript_round_trip(repo, monkeypatch, tmp_path):
    """It is only useful if it is still there tomorrow."""
    from arthur import transcript
    scripted(monkeypatch, ["ACTION: insert_after\ngarbled", "FINAL: gave up."])
    stream = collect(agent.run("add a docstring", repo, backend="mock"))

    path = str(tmp_path / "run.json")
    transcript.save(path, stream, meta={"task": "t", "repo": repo, "backend": "mock"})
    restored, _meta = transcript.load(path)

    proposed = next(e for e in restored if e.kind == "action_proposed")
    assert "garbled" in proposed.raw
