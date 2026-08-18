"""
Telling a task from a hello.

The bug that produced this module: typing `hey` at the prompt started a full
run -- 55 files indexed, a briefing built, a 3B model handed `TASK: hey`. It
replied "The task is to add a new function to an existing file", which is not
in the input anywhere; it is the subject of the system prompt's own worked
examples. Given nothing to do and a prompt that assumes there is something, the
model borrowed the nearest plausible job and spent five steps failing at it.

The classifier's one dangerous failure is the opposite direction: calling a
real request "chat" answers work with a canned greeting, and the user has no
way to override it. So the bar for chat is high and the bar for task is zero,
and most of what follows is evidence that the bar is where it should be.
"""

import pytest

from arthur import intent


# --- things that must never be treated as chat -------------------------------

TASKS = [
    # plain requests
    "add a docstring to retrieve",
    "fix the off-by-one in the pagination helper",
    "create a test for the indexer",
    "rename the score field to weight",
    "delete the unused import",
    "write a function that reverses a list",
    # a greeting wrapped around a real request -- the code signal must win
    "hey can you fix the parser",
    "hi, add a docstring to total_count",
    "hello! please read arthur/agent.py",
    "thanks, now add a test for it",
    "ok now fix it",
    "cool, run the tests",
    # terse but real
    "run pytest",
    "read config.py",
    "@arthur/agent.py",
    # follow-ups that lean on session history
    "now do the same for the other one",
    "do that again but for the class below it",
    "do it again for the other class",
]


def test_do_opens_a_command_as_often_as_a_question():
    """
    The one genuinely ambiguous opener in English. Settled by what follows:
    a subject pronoun asks, anything else instructs.
    """
    assert intent.classify("do you ever call run_command") == "question"
    assert intent.classify("do that again") == "task"
    assert intent.classify("do it again") == "task"
    # an explicit question mark overrules the heuristic either way
    assert intent.classify("do that again?") == "question"


@pytest.mark.parametrize("line", TASKS)
def test_real_requests_are_tasks(line):
    assert intent.classify(line) == "task", line


# --- questions: run the agent, but read-only ---------------------------------

QUESTIONS = [
    "what does the retriever score files on?",
    "what does the retriever do",
    "why is the gate rejecting my patch",
    "where is the step limit set",
    "how does trim_history decide what to drop",
    "explain how the safety gate works",
    "which file holds the parser",
    "is there a test for the indexer",
    "does the agent ever call run_command?",
    "tell me what apply_patch refuses",
]


@pytest.mark.parametrize("line", QUESTIONS)
def test_questions_are_recognised(line):
    assert intent.classify(line) == "question", line


@pytest.mark.parametrize("line", [
    # A change request keeps its question mark and stays a change request.
    "can you add a docstring to total_count?",
    "how about we rename score to weight?",
    "could you fix the off-by-one?",
    "would you create a test file for this?",
    "what if we extract that into a helper?",
])
def test_a_polite_request_is_not_a_question(line):
    """
    The dangerous confusion in the other direction. "can you add X?" is shaped
    exactly like a question and wants a patch; a write verb settles it.
    """
    assert intent.classify(line) == "task", line


def test_a_question_still_runs_the_agent():
    """It needs to read files to answer, so it is not answered locally."""
    assert intent.reply("question") == ""


def test_a_long_line_is_always_a_task():
    """
    Past a few words a line is making a request, whatever its vocabulary. This
    is the backstop for every phrasing the word lists do not know about.
    """
    assert intent.classify("hi there my friend how is everything going today") == "task"


def test_the_empty_line_is_not_classified_as_chat():
    """The session skips blank input before asking, but the default still
    has to be the safe one."""
    assert intent.classify("") == "task"
    assert intent.classify("   ") == "task"


# --- things that are chat ----------------------------------------------------

@pytest.mark.parametrize("line,kind", [
    ("hey", "greeting"),
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("Hello!", "greeting"),
    ("yo", "greeting"),
    ("hey there", "greeting"),
    ("hi arthur", "greeting"),
    ("good morning", "greeting"),
    ("how are you", "greeting"),
    ("what's up", "greeting"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("thanks!", "thanks"),
    ("nice", "thanks"),
    ("perfect thanks", "thanks"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("what can you do", "capability"),
    ("what can you do?", "capability"),
    ("who are you", "capability"),
    ("what is this", "capability"),
    ("how do i use this", "capability"),
])
def test_chat_is_recognised(line, kind):
    assert intent.classify(line) == kind, line


def test_case_and_punctuation_do_not_matter():
    for line in ("HEY", "hey!!", "  hey  ", "Hey."):
        assert intent.classify(line) == "greeting", line


# --- the replies -------------------------------------------------------------

def test_every_chat_kind_has_a_reply():
    for kind in ("greeting", "thanks", "farewell", "capability"):
        assert intent.reply(kind).strip(), kind


def test_a_task_has_no_reply():
    """The session uses a non-empty reply as the signal to skip the run."""
    assert intent.reply("task") == ""


def test_the_greeting_reply_teaches_the_thing_that_actually_helps():
    """
    Retrieval pins a file the task names above everything else, so naming one
    is worth more than any phrasing advice. A greeting is the only moment the
    user is guaranteed to be reading.
    """
    text = intent.reply("greeting").lower()
    assert "file" in text
    assert "/help" in text


def test_the_capability_reply_shows_real_examples():
    text = intent.reply("capability")
    assert "arthur >" in text
    # and the examples must be things the agent can actually do
    assert any(v in text for v in ("add", "fix", "create"))


# --- through the session -----------------------------------------------------

def _typing(monkeypatch, session, *lines):
    """A reader that types `lines` and then hits Ctrl+D, as a person would."""
    remaining = iter(lines)

    def read():
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(session, "_make_reader", lambda s: read)


def test_a_greeting_never_reaches_the_agent(tmp_path, monkeypatch, capsys):
    """
    The whole point. If this passes, `hey` cannot index a repo, build a
    briefing, or spend a step of anyone's GPU.
    """
    from arthur import session

    def explode(*a, **k):                     # pragma: no cover - must not run
        raise AssertionError("the agent was started for a greeting")

    monkeypatch.setattr("arthur.agent.run", explode)
    _typing(monkeypatch, session, "hey", "thanks", "what can you do")

    session.run_session(str(tmp_path), backend="mock", save_transcript=False)

    out = capsys.readouterr().out
    assert "Tell me what you'd like changed" in out
    assert "Anytime" in out
    assert "arthur >" in out                  # the capability examples


def test_a_task_still_reaches_the_agent(tmp_path, monkeypatch):
    from arthur import session

    seen = []

    def fake_run(task, repo_root, **kwargs):
        seen.append(task)
        yield from ()

    monkeypatch.setattr("arthur.agent.run", fake_run)
    _typing(monkeypatch, session, "add a docstring to foo")

    session.run_session(str(tmp_path), backend="mock", save_transcript=False)
    assert seen == ["add a docstring to foo"]


def test_one_shot_mode_short_circuits_too(tmp_path, monkeypatch, capsys):
    """`arthur -p "hi"` has the same problem and gets the same guard."""
    from arthur import cli

    monkeypatch.setattr("arthur.render.run_to_terminal",
                        lambda **k: (_ for _ in ()).throw(
                            AssertionError("agent started for a greeting")))

    assert cli._run_once("hi", str(tmp_path), "mock", False, False) == 0
    assert "Tell me what you'd like changed" in capsys.readouterr().out
