"""
Keeping the conversation inside the model's context window -- on purpose,
rather than letting the runtime do it silently.

Ollama (and llama.cpp underneath it) does not error when you overflow the
window. It drops tokens from the *front* and carries on. The front of our
conversation is the system prompt: the protocol definition, the tool list, the
repo tree. Lose that and the model stops emitting THOUGHT/ACTION/ARGS and
starts writing prose, with no error anywhere to explain why.

So we measure and trim ourselves, and when something has to go we choose what:
old observations from the middle, never the instructions at the front or the
most recent turns at the back.
"""

import dataclasses

from . import config


def estimate_tokens(text: str) -> int:
    """
    Cheap token estimate, deliberately pessimistic.

    A real tokenizer would mean shipping one per model family. Code tokenizes
    worse than prose -- punctuation, short identifiers and indentation all cost
    more per character -- so config.CHARS_PER_TOKEN is set low on purpose.
    Overestimating costs us a little unused window; underestimating costs us
    the system prompt.
    """
    return int(len(text) / config.CHARS_PER_TOKEN) + 1


def estimate_messages_tokens(messages: list[dict]) -> int:
    # ~4 tokens of per-message role/delimiter overhead in most chat templates.
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


def prompt_char_budget(num_ctx: int | None = None) -> int:
    """How many characters the system prompt may spend on repo context."""
    num_ctx = num_ctx or config.OLLAMA_NUM_CTX
    tokens = int(num_ctx * config.CONTEXT_PROMPT_BUDGET)
    return int(tokens * config.CHARS_PER_TOKEN)


def fit_context_files(files, budget_chars: int | None = None) -> list:
    """
    Trim retrieved files to fit the prompt budget, best-scoring first.

    Files are truncated rather than dropped where possible: half of a relevant
    file is usually more useful to the model than none of it. A file whose
    remaining budget is too small to be meaningful is dropped instead of
    included as a useless sliver.
    """
    budget = budget_chars if budget_chars is not None else prompt_char_budget()
    min_useful = 400  # below this a snippet teaches the model nothing

    kept, spent = [], 0
    for f in files:
        remaining = budget - spent
        if remaining < min_useful:
            break
        snippet = f.snippet
        if len(snippet) > remaining:
            snippet = snippet[:remaining] + "\n... [truncated to fit context]"
        # RetrievedFile is a dataclass; copy so the caller's list is untouched.
        # `replace` rather than a hand-built constructor call: a copy that
        # silently dropped the `mentioned` flag would strip exactly the files
        # the task named of the marking that says so.
        kept.append(dataclasses.replace(f, snippet=snippet))
        spent += len(snippet)
    return kept


ELISION = "[... earlier steps elided to stay within the context window ...]"

# Marks the per-task briefing agent.build_task_briefing produces. Trimming has
# to be able to recognise one: the briefing carries the current task and the
# contents of the file it is about, so losing it to a long tool history is how
# the model ends up editing from memory.
BRIEFING_PREFIX = "Current repository contents"


def _last_briefing(messages: list[dict]) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content", "").startswith(BRIEFING_PREFIX):
            return m
    return None


def trim_history(messages: list[dict], num_ctx: int | None = None,
                 keep_recent: int = 6) -> list[dict]:
    """
    Drop the oldest middle turns once the conversation gets too long.

    What is preserved, and why:
      - messages[0], the system prompt -- the protocol itself. Never dropped.
      - the newest task briefing -- the current goal and the current contents
        of the file it concerns. Note *newest*, not messages[1]: in a session
        that has run several tasks, messages[1] is the first task's briefing,
        which is both the wrong goal and a stale view of the repo.
      - the last `keep_recent` messages -- what it needs to pick the next action.

    Everything between is replaced by a single marker, so the model can see
    that history was removed rather than being quietly confused by a gap.
    """
    num_ctx = num_ctx or config.OLLAMA_NUM_CTX
    limit = int(num_ctx * config.CONTEXT_HISTORY_BUDGET)

    if estimate_messages_tokens(messages) <= limit:
        return messages

    head = messages[:1]                      # the protocol
    briefing = _last_briefing(messages)
    if briefing is None and len(messages) > 1:
        briefing = messages[1]               # pre-briefing callers: the raw task
    tail = messages[-keep_recent:] if keep_recent else []

    # Only prepend the briefing if the tail doesn't already carry it, otherwise
    # the model sees the same block twice.
    if briefing is not None and not any(m is briefing for m in tail):
        head = head + [briefing]
    if len(head) + len(tail) >= len(messages):
        return messages                      # nothing in the middle to drop

    trimmed = head + [{"role": "user", "content": ELISION}] + tail

    # Still too long even after elision: the recent turns themselves are huge
    # (a big file read back, usually). Shrink the tail until it fits.
    while estimate_messages_tokens(trimmed) > limit and len(tail) > 2:
        tail = tail[2:]
        trimmed = head + [{"role": "user", "content": ELISION}] + tail

    return trimmed
