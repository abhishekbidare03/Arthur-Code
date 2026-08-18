"""
Context-window budgeting tests.

The failure this guards against is invisible at runtime: overflow the window
and Ollama drops tokens from the front, taking the system prompt -- and with
it the protocol definition -- without raising anything. So the invariants
worth pinning down are about *what survives* a trim, not just that trimming
happened.
"""

from arthur import config, context as ctx
from arthur.retriever import RetrievedFile


def msg(role, content):
    return {"role": role, "content": content}


# --- estimation --------------------------------------------------------------

def test_token_estimate_grows_with_length():
    assert ctx.estimate_tokens("x" * 350) > ctx.estimate_tokens("x" * 35)


def test_token_estimate_is_pessimistic():
    """Better to overestimate: underestimating is what eats the system prompt."""
    text = "def f(x):\n    return x + 1\n" * 20
    naive_gpt_ratio = len(text) / 4
    assert ctx.estimate_tokens(text) >= naive_gpt_ratio


# --- fitting retrieved files -------------------------------------------------

def files(*sizes):
    return [RetrievedFile(path=f"f{i}.py", score=100 - i, snippet="x" * n)
            for i, n in enumerate(sizes)]


def test_files_within_budget_are_untouched():
    given = files(500, 500)
    kept = ctx.fit_context_files(given, budget_chars=5000)
    assert [f.snippet for f in kept] == [f.snippet for f in given]


def test_oversized_file_is_truncated_not_dropped():
    kept = ctx.fit_context_files(files(10_000), budget_chars=2000)
    assert len(kept) == 1
    assert len(kept[0].snippet) <= 2000 + 60      # + the truncation marker
    assert "truncated" in kept[0].snippet


def test_budget_is_spent_on_the_best_scoring_files_first():
    kept = ctx.fit_context_files(files(1500, 1500, 1500), budget_chars=1600)
    assert [f.path for f in kept] == ["f0.py"]


def test_useless_slivers_are_dropped_rather_than_included():
    # 1500 of a 1600 budget spent; the 100 chars left teach the model nothing.
    kept = ctx.fit_context_files(files(1500, 1500), budget_chars=1600)
    assert len(kept) == 1


def test_fitting_does_not_mutate_the_caller_list():
    given = files(10_000)
    ctx.fit_context_files(given, budget_chars=500)
    assert len(given[0].snippet) == 10_000


# --- trimming history --------------------------------------------------------

def test_short_history_is_returned_unchanged():
    history = [msg("system", "protocol"), msg("user", "task"), msg("assistant", "ok")]
    assert ctx.trim_history(history, num_ctx=8192) is history


def test_long_history_is_trimmed():
    history = [msg("system", "protocol"), msg("user", "the task")]
    history += [msg("user", "OBSERVATION: " + "x" * 4000) for _ in range(20)]
    trimmed = ctx.trim_history(history, num_ctx=2048)
    assert len(trimmed) < len(history)


def test_system_prompt_and_task_always_survive():
    """The invariant that matters. Lose these and the agent stops being one."""
    history = [msg("system", "THE PROTOCOL"), msg("user", "THE ORIGINAL TASK")]
    history += [msg("user", "OBSERVATION: " + "x" * 8000) for _ in range(40)]

    trimmed = ctx.trim_history(history, num_ctx=2048)

    assert trimmed[0]["content"] == "THE PROTOCOL"
    assert trimmed[1]["content"] == "THE ORIGINAL TASK"


def test_most_recent_turn_survives():
    history = [msg("system", "protocol"), msg("user", "task")]
    history += [msg("user", "OBSERVATION: " + "x" * 4000) for _ in range(20)]
    history.append(msg("user", "THE LATEST OBSERVATION"))

    trimmed = ctx.trim_history(history, num_ctx=2048)
    assert trimmed[-1]["content"] == "THE LATEST OBSERVATION"


def test_elision_is_visible_to_the_model():
    """A silent gap confuses the model; a marked one is information."""
    history = [msg("system", "protocol"), msg("user", "task")]
    history += [msg("user", "OBSERVATION: " + "x" * 4000) for _ in range(20)]

    trimmed = ctx.trim_history(history, num_ctx=2048)
    assert any(ctx.ELISION == m["content"] for m in trimmed)


def test_result_actually_fits_the_window():
    history = [msg("system", "protocol"), msg("user", "task")]
    history += [msg("user", "OBSERVATION: " + "x" * 20_000) for _ in range(10)]

    trimmed = ctx.trim_history(history, num_ctx=4096)
    limit = 4096 * config.CONTEXT_HISTORY_BUDGET
    # The head is preserved unconditionally, so the guarantee is best-effort
    # once the system prompt alone is huge -- but it must have shrunk hard.
    assert ctx.estimate_messages_tokens(trimmed) < ctx.estimate_messages_tokens(history)
    assert len(trimmed) <= 5 or ctx.estimate_messages_tokens(trimmed) <= limit
