"""
Answering a question instead of finding something to rewrite.

Live, on the real repository, `what does the retriever score files on?`:

    step 3  ACTION: edit_file(path='arthur/retriever.py')
            ERROR: edit_file is missing: find
    step 4  THOUGHT: The user has not provided the specific lines to find and
            replace. I need to prompt the user to supply the FIND content.
    step 5  ACTION: edit_file(path='arthur/retriever.py')

Read the thought: the model has worked out there is nothing to replace, and
keeps calling the replacing tool anyway. It is not confused about the question,
it is confused about what it is allowed to do -- the briefing's closing
paragraph is entirely about which file to CHANGE, and the protocol's only
non-editing exit is FINAL, twenty lines below eight editing examples.

So answer mode is two things, and the second is the one that works: a briefing
that asks for an answer, and a refusal that makes editing impossible. Telling a
3B model not to edit gets it to agree and then edit anyway.
"""

import pytest

from arthur import agent
from arthur.llm_backend import MockBackend


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "retriever.py").write_text(
        "PATH_WEIGHT = 8\n\n\ndef retrieve(task, index):\n"
        "    return sorted(index.files)\n", encoding="utf-8")
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


EDIT_TURN = ('THOUGHT: I will document it.\nCONFIDENCE: 0.9\n'
             'ACTION: edit_file\nPATH: retriever.py\n'
             'FIND:\n```\nPATH_WEIGHT = 8\n```\n'
             'REPLACE:\n```\nPATH_WEIGHT = 9\n```')


# --- the wall ----------------------------------------------------------------

@pytest.mark.parametrize("tool", agent.WRITE_TOOLS)
def test_every_writing_tool_is_refused(tool, repo, monkeypatch):
    turn = EDIT_TURN.replace("ACTION: edit_file", f"ACTION: {tool}")
    scripted(monkeypatch, [turn, "FINAL: it scores on path and content."])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", auto_approve=True, mode="answer"))

    observation = next(e for e in stream if e.kind == "observation")
    assert "switched off" in observation.text
    assert "question" in observation.text


def test_the_file_on_disk_is_untouched(repo, monkeypatch):
    """The property that makes this worth having. Not advice -- a guarantee."""
    before = open(f"{repo}/retriever.py").read()
    scripted(monkeypatch, [EDIT_TURN, "FINAL: answered."])
    collect(agent.run("what does it score on?", repo, backend="mock",
                      auto_approve=True, mode="answer"))
    assert open(f"{repo}/retriever.py").read() == before


def test_auto_approve_does_not_open_the_door(repo, monkeypatch):
    """
    -y turns off the human check, not the mode. A question asked in an
    auto-approving session must still not write.
    """
    before = open(f"{repo}/retriever.py").read()
    scripted(monkeypatch, [EDIT_TURN, EDIT_TURN, "FINAL: answered."])
    stream = collect(agent.run("why is it slow?", repo, backend="mock",
                               auto_approve=True, mode="answer"))
    assert open(f"{repo}/retriever.py").read() == before
    assert not any(e.kind == "gate_decision" for e in stream)


def test_the_refusal_names_the_way_out(repo, monkeypatch):
    """
    A refusal that only says no is the useless-observation failure again. This
    one hands back the exact two lines that end the run correctly.
    """
    scripted(monkeypatch, [EDIT_TURN, "FINAL: done."])
    stream = collect(agent.run("what does it score on?", repo, backend="mock",
                               mode="answer"))
    text = next(e for e in stream if e.kind == "observation").text
    assert "FINAL:" in text
    assert "read_file" in text


# --- reading still works -----------------------------------------------------

def test_reading_tools_are_untouched(repo, monkeypatch):
    scripted(monkeypatch, [
        'THOUGHT: let me look.\nACTION: read_file\nARGS: {"path": "retriever.py"}',
        "FINAL: it scores on path, symbols and content.",
    ])
    stream = collect(agent.run("what does it score on?", repo, backend="mock",
                               mode="answer"))

    observation = next(e for e in stream if e.kind == "observation")
    assert "PATH_WEIGHT = 8" in observation.text
    assert stream[-1].kind == "final"


def test_the_answer_comes_back_as_the_final(repo, monkeypatch):
    scripted(monkeypatch, ["THOUGHT: I can see it.\n"
                           "FINAL: path matches score 8, content 1."])
    stream = collect(agent.run("what does it score on?", repo, backend="mock",
                               mode="answer"))
    assert stream[-1].summary == "path matches score 8, content 1."
    assert stream[-1].files_changed == []


# --- prose is an answer ------------------------------------------------------

PROSE = ("The retriever scores files on two mechanisms. A file the task names "
         "by name is pinned to the front; everything else is ranked by "
         "weighted keyword overlap across path, symbols and contents.")


def test_a_turn_with_no_action_is_taken_as_the_answer(repo, monkeypatch):
    """
    Observed live. The model answered the question correctly and in full, in
    plain prose, and was told it had broken protocol -- because there was no
    `FINAL:` in front of it. Re-prompted, it replied "No action taken". A
    correct answer was discarded and replaced with a worse one over a missing
    five-character prefix.
    """
    scripted(monkeypatch, [PROSE, "FINAL: unused."])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))

    assert stream[-1].kind == "final"
    assert "weighted keyword overlap" in stream[-1].summary
    assert not any(e.kind == "protocol_violation" for e in stream)


def test_the_run_stops_there_rather_than_asking_again(repo, monkeypatch):
    scripted(monkeypatch, [PROSE] + ["FINAL: should never be reached."] * 5)
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))
    assert len([e for e in stream if e.kind == "step_started"]) == 1


def test_an_answer_labelled_as_a_thought_is_still_the_answer(repo, monkeypatch):
    """
    The parser reads an unstructured turn as one long THOUGHT, so the label is
    all that distinguishes these two cases -- and the label is the part the
    model got wrong. Only the keyword is stripped; every word after it is kept.
    """
    scripted(monkeypatch, [f"THOUGHT: {PROSE}"])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))
    assert stream[-1].summary == PROSE


def test_the_answer_is_not_printed_as_both_a_thought_and_an_answer(repo, monkeypatch):
    scripted(monkeypatch, [f"THOUGHT: let me summarise.\n\n{PROSE}"])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))

    assert not any(e.kind == "thought" for e in stream)
    assert PROSE in stream[-1].summary
    assert stream[-1].summary.count("weighted keyword overlap") == 1


def test_edit_mode_still_treats_prose_as_a_violation(repo, monkeypatch):
    """
    The leniency is safe only because nothing can be written. In edit mode a
    turn with no action has produced no patch, and accepting it as done would
    report success for a run that changed nothing.
    """
    scripted(monkeypatch, [PROSE, "FINAL: gave up."])
    stream = collect(agent.run("fix the scoring", repo, backend="mock"))
    assert any(e.kind == "protocol_violation" for e in stream)


def test_an_empty_turn_is_still_a_violation(repo, monkeypatch):
    """Nothing is not an answer."""
    scripted(monkeypatch, ["   ", "FINAL: recovered."])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))
    assert any(e.kind == "protocol_violation" for e in stream)


def test_an_action_is_still_honoured_in_answer_mode(repo, monkeypatch):
    """Prose is a fallback, not a shortcut past the tools."""
    scripted(monkeypatch, [
        'THOUGHT: reading first.\nACTION: read_file\nARGS: {"path": "retriever.py"}',
        PROSE,
    ])
    stream = collect(agent.run("what does it score on?", repo,
                               backend="mock", mode="answer"))
    assert any(e.kind == "observation" for e in stream)
    assert stream[-1].kind == "final"


# --- the briefing ------------------------------------------------------------

def _briefing(mode):
    from arthur.indexer import build_index
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        open(f"{root}/a.py", "w").write("x = 1\n")
        return agent.build_task_briefing("what does a.py do?",
                                         build_index(root), [], mode)


def test_the_answer_briefing_asks_for_an_answer():
    text = _briefing("answer")
    assert "QUESTION:" in text
    assert "FINAL" in text


def test_the_answer_briefing_drops_the_which_file_to_change_paragraph():
    """
    The specific line that produced the loop above. In edit mode it tells the
    model to work out which file the task is about and change it, which for a
    question is an instruction to go and do damage.
    """
    text = _briefing("answer")
    # edit_file may still be NAMED -- the refusal list has to name it. What
    # must be gone is any instruction to use it.
    assert "Edit it with edit_file" not in text
    assert "Only create a new file" not in text
    assert "change it with edit_file" not in text
    assert "switched off" in text


def test_edit_mode_is_unchanged_by_all_this():
    text = _briefing("edit")
    assert "TASK:" in text
    assert "QUESTION:" not in text


def test_edit_mode_is_the_default():
    """Anything that has not been taught about modes keeps writing."""
    import inspect
    assert inspect.signature(agent.run).parameters["mode"].default == "edit"
    assert inspect.signature(
        agent.build_task_briefing).parameters["mode"].default == "edit"


# --- the routing that picks the mode -----------------------------------------

def test_the_session_routes_a_question_to_answer_mode(tmp_path, monkeypatch):
    from arthur import session

    modes = []

    def fake_run(task, repo_root, mode="edit", **kwargs):
        modes.append((task, mode))
        yield from ()

    monkeypatch.setattr("arthur.agent.run", fake_run)
    typed = iter(["what does the indexer do", "add a docstring to it"])

    def read():
        try:
            return next(typed)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(session, "_make_reader", lambda s: read)
    session.run_session(str(tmp_path), backend="mock", save_transcript=False)

    assert modes == [("what does the indexer do", "answer"),
                     ("add a docstring to it", "edit")]
