"""
Deciding whether a line of input is a task at all.

Typing "hey" used to start a full agent run: index 55 files, retrieve context,
and hand a 3B model a briefing whose TASK line said `hey`. The model does the
only thing that prompt makes possible -- it looks for something plausible to
do, finds the prompt's own worked examples, and announces "the task is to add a
new function to an existing file". Five steps later it fails. Nothing in the
loop was broken; it was asked to plan a change and there was no change to plan.

So the fix belongs before the loop, not inside it. A greeting is answered here,
in Python, in no time at all, and never reaches the model.

The classifier is deliberately lopsided. Routing a task to chat is a real
failure -- the user asked for work and got a brush-off -- while routing chat to
the agent is only the status quo. So every rule below is a rule for proving
something is CHAT, the code-signal check overrides all of them, and anything
unproven is a task.
"""

from __future__ import annotations

import re


# Verbs that ask for the repository to be different afterwards. Their presence
# settles two questions at once: the line is real work, and it is work that
# writes. "hey can you fix the parser" is a task despite the hello, and "how do
# I add a route?" is a change request despite the question mark.
_WRITE_VERBS = {
    "add", "append", "build", "change", "clean", "convert", "create",
    "delete", "document", "edit", "extract", "fix", "generate", "implement",
    "improve", "insert", "install", "make", "move", "optimise", "optimize",
    "refactor", "remove", "rename", "replace", "rewrite", "simplify",
    "update", "write",
}

# Verbs that ask for work but leave the repository alone. A line built on one
# of these is a task, but it is not a reason to overrule a question mark.
_READ_VERBS = {
    "check", "debug", "describe", "explain", "find", "import", "list", "open",
    "read", "return", "review", "run", "show", "summarise", "summarize",
    "test", "trace", "walk",
}

_CODE_VERBS = _WRITE_VERBS | _READ_VERBS

# Openers that mark a line as something to answer rather than something to do.
_INTERROGATIVE = {"what", "whats", "why", "how", "hows", "where", "wheres",
                  "when", "which", "who", "whose", "does", "do", "did", "is",
                  "are", "was", "were", "can", "could", "should", "would",
                  "will", "explain", "describe", "tell"}

# "do" is the one opener that is as often an imperative as a question: "do you
# support X" asks, "do that again for the class below" instructs. English
# settles it by what follows -- a subject pronoun means a question, anything
# else (a demonstrative, a noun) means a command. "it" is left out on purpose,
# because "do it again" is the commonest instruction there is.
_DO_SUBJECTS = {"you", "we", "i", "they", "u"}

# Punctuation and structure that only appear when code is being discussed.
_CODE_SHAPE = re.compile(
    r"""
      \.(py|js|ts|md|txt|json|toml|ya?ml|html|css|c|h|cpp|rs|go|java|sh)\b
    | [/\\]                      # a path separator
    | \w+\(                      # a call: foo(
    | `                          # a backtick quote
    | \b\w+_\w+\b                # snake_case identifier
    | ^@                         # an @file mention
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_GREETING = {
    "hi", "hey", "hello", "yo", "hiya", "howdy", "sup", "greetings",
    "morning", "afternoon", "evening", "goodmorning",
}
_THANKS = {"thanks", "thank", "thx", "ty", "cheers", "appreciated", "nice",
           "cool", "great", "awesome", "perfect", "ok", "okay", "k", "good"}
_FAREWELL = {"bye", "goodbye", "exit", "quit", "later", "cya", "seeya"}

# Words that carry no intent of their own and so never block a chat match.
_FILLER = {"a", "am", "are", "arthur", "buddy", "can", "do", "doing", "dude",
           "everyone", "fine", "friend", "goin", "going", "how", "how's",
           "hows", "i", "is", "it", "its", "just", "man", "me", "my", "so",
           "testing", "there", "this", "u", "up", "well", "what", "whats",
           "you", "your", "youre", "yourself"}

# Asked in so many shapes that a word-set test is the wrong tool; these are
# matched on the normalized line as a whole.
_CAPABILITY = re.compile(
    r"^(?:"
    r"what (?:can|do) (?:you|u) (?:do|help with)"
    r"|what are (?:you|your) (?:tools|capabilities)"
    r"|who (?:are|r) (?:you|u)"
    r"|what(?: is|s) (?:this|arthur)"
    r"|how (?:do i|to) use (?:this|you|arthur)"
    r"|help me"
    r"|what should i (?:say|type|ask)"
    r")\b"
)

_PUNCT = re.compile(r"[^\w\s]+")


def _normalize(line: str) -> str:
    """
    Lowercase, and reduce to words. Apostrophes go rather than being kept as
    part of the word, so "what's" and "whats" -- both of which get typed --
    reach the vocabulary as the same token.
    """
    return " ".join(_PUNCT.sub(" ", line.lower().replace("'", "")).split())


def _is_question(text: str, words: list[str]) -> bool:
    """
    Is this asking to be told something, rather than asking for a change?

    A write verb overrules everything -- "how do I add a cache?" wants a cache,
    not an essay. Past that, an interrogative opener or a trailing question
    mark is enough.
    """
    if not words or any(w in _WRITE_VERBS for w in words):
        return False
    if words[0] == "do" and not text.endswith("?"):
        return len(words) > 1 and words[1] in _DO_SUBJECTS
    return words[0] in _INTERROGATIVE or text.endswith("?")


def classify(line: str) -> str:
    """
    One of "greeting", "thanks", "farewell", "capability", "question" or "task".

    The first four are answered here, without a model call. "question" and
    "task" both run the agent -- the difference is that a question runs it with
    the writing tools switched off, so it answers instead of finding something
    to change.
    """
    raw = line.strip()
    if not raw:
        return "task"

    text = _normalize(raw)
    words = text.split()

    # A code signal means real work, but says nothing about whether that work
    # writes -- "what does retriever.py score on?" is both code and a question.
    code = bool(_CODE_SHAPE.search(raw)) or any(w in _CODE_VERBS for w in words)

    if _CAPABILITY.match(text):
        return "capability"

    # Small talk is checked before questions, because the commonest greetings
    # in English are shaped like questions -- "how are you", "what's up" -- and
    # running the agent on one is the bug this module exists to stop.
    #
    # The five-word ceiling is the backstop for every phrasing the vocabulary
    # does not know: past a handful of words a line is making a request, however
    # harmless each word looks. Chat is short; requests are not.
    if not code and words and len(words) <= 5:
        chat = _small_talk(words)
        if chat:
            return chat

    if _is_question(raw.lower(), words):
        return "question"

    return "task"


def _small_talk(words: list[str]) -> str | None:
    """
    The kind of pure chat this line is, or None if it is not chat at all.

    Content words need not all come from the same list: "good morning" is one
    word of each, and "perfect thanks" and "ok bye" are the same shape. So
    require full coverage by the union, then let the first list that matches
    name the whole line -- which puts "thanks bye" under farewell, the more
    specific of the two.
    """
    content = [w for w in words if w not in _FILLER]
    if not content:
        return "greeting"          # "how are you", "what's up" -- filler only

    vocabularies = (("farewell", _FAREWELL), ("greeting", _GREETING),
                    ("thanks", _THANKS))
    if not all(any(w in vocab for _, vocab in vocabularies) for w in content):
        return None

    for kind, vocab in vocabularies:
        if any(w in vocab for w in content):
            return kind
    return None

    return "task"


_CAPABILITY_REPLY = """\
I change code in this repository. Describe the change in plain English and
I'll find the file, propose a patch, and show you the diff before anything is
written.

  arthur > add a docstring to retrieve() in arthur/retriever.py
  arthur > fix the off-by-one in the pagination helper
  arthur > create a tests/test_indexer.py with a case for empty repos

Naming the file makes me much more accurate; @-prefix it (@arthur/agent.py) to
force it into context. /help lists the commands."""

_REPLIES = {
    "greeting": ("Hi. Tell me what you'd like changed -- name the file if you "
                 "can, it makes me a lot more accurate.\nSomething like: "
                 "\"add a docstring to load_config in arthur/config.py\".\n"
                 "/help for commands."),
    "thanks": "Anytime. What's next?",
    "farewell": "Ctrl+D or /exit to leave.",
    "capability": _CAPABILITY_REPLY,
}


def reply(kind: str) -> str:
    """The canned answer for a non-task line. Empty string for "task"."""
    return _REPLIES.get(kind, "")
