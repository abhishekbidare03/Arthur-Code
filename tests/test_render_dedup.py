"""
Not printing the same paragraph twice.

Streaming exists because a 3B on a 4GB card takes ten seconds a turn and a
silent terminal looks hung. It works by echoing tokens until a protocol keyword
shows up, then handing over to the structured renderer -- which is why THOUGHT
is not printed again after being streamed.

An answer in prose has no protocol keyword to stop at, so the whole thing gets
echoed, and then arrives a second time as the Final summary. On a three
paragraph answer that is a screen and a half of duplicate text.
"""

import io

from arthur import events as ev
from arthur.render import TerminalRenderer


ANSWER = ("The retriever pins any file the task names, then ranks the rest by "
          "weighted keyword overlap across path, symbols and contents.")


def streamed(text, summary, chunk=8):
    """Stream `text`, then render a Final carrying `summary`."""
    out = io.StringIO()
    renderer = TerminalRenderer(out=out)
    renderer.render(ev.StepStarted(step=1, max_steps=8))
    for i in range(0, len(text), chunk):
        renderer.on_token(text[i:i + chunk])
    renderer.render(ev.Final(summary=summary, steps_used=1, elapsed=1.0))
    return out.getvalue()


def test_a_streamed_answer_is_not_repeated_by_the_summary():
    assert streamed(ANSWER, ANSWER).count("weighted keyword overlap") == 1


def test_it_is_still_shown_once():
    assert "weighted keyword overlap" in streamed(ANSWER, ANSWER)


def test_whitespace_differences_do_not_defeat_the_check():
    """The buffer is raw chunks; the summary has been stripped."""
    assert streamed(f"\n  {ANSWER}\n\n", ANSWER).count("weighted keyword") == 1


def test_an_ordinary_final_is_still_printed():
    """
    A normal run streams a THOUGHT and ends with a one-line report of what was
    changed. That report is not in the stream and must not be suppressed.
    """
    out = streamed("THOUGHT: I will add the docstring.",
                   "Added a docstring to total_count in inventory.py.")
    assert "Added a docstring to total_count" in out


def test_nothing_streamed_means_nothing_suppressed():
    """The non-streaming path -- mock backend, gemini, --replay."""
    out = io.StringIO()
    renderer = TerminalRenderer(out=out)
    renderer.render(ev.Final(summary=ANSWER, steps_used=1, elapsed=1.0))
    assert ANSWER in out.getvalue()


def test_an_empty_summary_is_not_treated_as_streamed():
    out = streamed(ANSWER, "")
    assert "DONE" in out


# --- the read-only footer ----------------------------------------------------

def _final_with_mode(mode):
    out = io.StringIO()
    renderer = TerminalRenderer(out=out)
    renderer.render(ev.RunStarted(task="what does it do?", repo_root="/r",
                                  backend="ollama", model="phi4-mini", mode=mode))
    renderer.render(ev.Final(summary="It scores on path and content.",
                             steps_used=1, elapsed=1.0))
    return out.getvalue()


def test_answer_mode_does_not_warn_that_nothing_changed():
    """Changing nothing is the point; the warning would report success as a
    caveat and make a correct run look like a failed one."""
    assert "no files were changed" not in _final_with_mode("answer")


def test_edit_mode_still_warns():
    assert "no files were changed" in _final_with_mode("edit")


def test_answer_mode_says_so_up_front():
    text = _final_with_mode("answer")
    assert "question:" in text
    assert "read-only" in text


def test_edit_mode_header_is_unchanged():
    text = _final_with_mode("edit")
    assert "task:" in text
    assert "read-only" not in text


def test_answer_mode_says_how_to_get_the_change_applied():
    """
    The one way the routing can be wrong is a change request phrased as a
    question -- "why not return [] instead?". Answering it is the safe error,
    and this line is how the user un-makes it without knowing modes exist.
    """
    text = _final_with_mode("answer")
    assert "instruction" in text
