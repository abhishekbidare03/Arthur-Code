"""
The orchestration loop. This is the part worth understanding line by line:

  1. Index the repo + retrieve relevant files      -> "read directory, understand code"
  2. Build a system prompt with that context        -> grounding
  3. Loop: model emits THOUGHT/ACTION/ARGS text      -> "plan"
  4. Destructive actions go through the safety gate  -> "trustworthy execution"
  5. Tool result is fed back as the next observation -> classic ReAct loop
  6. Model emits FINAL when done                     -> "tell what it did"

`run()` is a generator: it yields events (see events.py) and never prints. The
one event it expects an answer to is ApprovalNeeded -- the consumer sends back
a Decision. Keeping the human out of the loop body is what lets the same code
run interactively, headless, and under a test.
"""

import json
import re
import time
from typing import Callable, Iterator

from . import config
from . import context as ctx
from . import events as ev
from . import patcher
from . import tools
from . import safety_gate
from .indexer import build_index
from .retriever import retrieve
from .llm_backend import get_backend


KEYWORDS = ("THOUGHT", "CONFIDENCE", "ACTION", "ARGS", "PATH", "CONTENT",
            "FIND", "REPLACE", "AFTER", "FINAL")

# Real models decorate the protocol: **ACTION:**, ### THOUGHT:, `ARGS:`, and --
# phi4-mini, observed live -- plain Title Case: "Action:" / "Args:".
_DECORATED_KEYWORD = re.compile(
    # optional leading #/>/-/*/` decoration, the keyword, then a colon that
    # may sit either inside or outside the bold markers: **ACTION:** / **ACTION**:
    r"^[ \t]*(?:[#>\-*`]+[ \t]*)*(" + "|".join(KEYWORDS) + r")(?:\*\*|`)*[ \t]*:[ \t]*(?:\*\*|`)*",
    re.MULTILINE | re.IGNORECASE,
)

_FENCE_SPLIT = re.compile(r"(```)")


def _iter_regions(text: str):
    """
    Walk the text as (inside_a_fence, chunk) pairs.

    Normalization must not touch fenced content. The blocks now carry entire
    source files, and a perfectly ordinary line like `# Action: rename this`
    would otherwise be rewritten to `ACTION:` and land in the user's code --
    or worse, be read as a second action.
    """
    parts = _FENCE_SPLIT.split(text)
    inside = False
    for part in parts:
        if part == "```":
            inside = not inside
            yield True, part          # the fence itself is never normalized
            continue
        yield inside, part


def _normalize(text: str) -> str:
    """
    Strip the markdown a chatty model wraps the protocol in, and upper-case the
    keywords so everything downstream can assume one spelling.

    Case-insensitivity is what makes phi4-mini usable: it follows the protocol
    faithfully but writes `Thought:`/`Action:`/`Args:`, which a case-sensitive
    parser reads as pure prose -- the model looks broken when it is in fact
    doing exactly as it was told.
    """
    text = text.replace("\r\n", "\n")
    return "".join(
        chunk if fenced else _DECORATED_KEYWORD.sub(
            lambda m: f"{m.group(1).upper()}:", chunk)
        for fenced, chunk in _iter_regions(text)
    )


def _strip_fences(text: str) -> str:
    """Remove a leading ```json / ``` fence and its closing partner."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
    return text.strip()


def _first_json_object(text: str) -> str | None:
    """
    Extract the first balanced {...} block, tracking string state so braces
    and quotes inside the patch content don't fool the scanner. Needed
    because models append prose after the JSON ("...that should fix it!").
    """
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _escape_control_chars(blob: str) -> str:
    """
    Models emitting a whole rewritten file frequently put *literal* newlines
    inside the JSON string instead of \\n, which is invalid JSON. Re-escape
    control characters that appear inside string literals.
    """
    out, in_string, escaped = [], False, False
    for ch in blob:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


# GREEDY on purpose -- see _normalize_triple_quotes.
_TRIPLE_QUOTED = re.compile(r'"""(.*)"""', re.DOTALL)


def _normalize_triple_quotes(text: str) -> str:
    """
    Rewrite a Python triple-quoted value into a real JSON string.

    Observed from qwen2.5-coder:3b: asked for a whole rewritten file as JSON,
    it reaches for the Python syntax it has seen a million times and emits

        ARGS: {"path": "a.py", "new_content": \"\"\"
        def f(): ...
        \"\"\"}

    which is not JSON at all, and whose three consecutive quotes also
    desynchronise the balanced-brace scanner.

    The match is greedy because the *file being written* frequently contains
    triple quotes of its own -- any Python file with a module docstring does.
    A non-greedy match stops at that inner docstring and silently truncates
    everything after it, which is far worse than not parsing at all: it
    produces a plausible-looking patch that deletes most of the file. Greedy
    takes the outermost pair, so inner docstrings ride along as content.
    """
    return _TRIPLE_QUOTED.sub(lambda m: json.dumps(m.group(1)), text)


def _candidates(args_text: str):
    """
    Progressively more aggressive repairs, cheapest and safest first.

    Order matters. Well-formed JSON must be parsed as-is: the triple-quote
    rewrite is a guess about malformed input, and applying it to valid JSON
    that happens to contain \"\"\" inside a string corrupts it.
    """
    text = _strip_fences(args_text)

    blob = _first_json_object(text)
    if blob is not None:
        yield blob
        yield _escape_control_chars(blob)

    if '"""' in text:
        repaired = _normalize_triple_quotes(text)
        blob = _first_json_object(repaired)
        if blob is not None:
            yield blob
            yield _escape_control_chars(blob)


def parse_args_block(args_text: str) -> dict:
    """Best-effort JSON recovery from whatever the model put after ARGS:."""
    for candidate in _candidates(args_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


# --- the block form ----------------------------------------------------------
#
# Asking a 3B to embed an entire source file inside a JSON string is asking for
# the one thing it is worst at. Across live runs the same model failed
# differently every time: Python triple quotes one run, single-quoted values
# with a nested object the next, unescaped newlines the run after. Each fix is
# a regex, and there is always another malformation.
#
# So for apply_patch -- the only tool whose argument is large -- the protocol
# also accepts a form with no escaping at all:
#
#     ACTION: apply_patch
#     PATH: calculator.py
#     CONTENT:
#     ```python
#     ...the whole file, verbatim...
#     ```
#
# Small models are heavily trained on fenced code blocks and produce them
# reliably. JSON args still work and are still preferred for the short-argument
# tools; this is a second accepted spelling, not a replacement.

_PATH_LINE = re.compile(r"^PATH:[ \t]*[`'\"]?([^\r\n`'\"]+)", re.MULTILINE)

_BARE_FENCE = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$")

# Where a fenced block must stop even if its closing fence never arrived.
_BLOCK_BOUNDARY = re.compile(
    r"^(?:FIND|REPLACE|CONTENT|AFTER):[ \t]*\r?$|^(?:ACTION|FINAL|ARGS):",
    re.MULTILINE,
)


def _fenced_after(keyword: str, text: str) -> str | None:
    """
    Pull the fenced block that follows `KEYWORD:` on its own line.

    Shared by CONTENT:, FIND: and REPLACE:. Kept tolerant about the fence --
    models vary the language tag, indent it, or occasionally omit it -- but
    strict about the keyword being at the start of a line, so a FIND: mentioned
    inside the code being edited can't be mistaken for the protocol.
    """
    # Everything after the keyword is optional: the newline, the fence, both.
    # The protocol shows the value in a fence on the line below, and phi4-mini
    # writes all three layouts --
    #
    #     AFTER:\n```\ndef total_count(self):\n```     the documented one
    #     AFTER: ```\ndef total_count(self):\n```      fence on the keyword line
    #     AFTER: total_count:                          no fence at all
    #
    # -- and the last two used to yield None. That is worse than it sounds,
    # because parse_block_form drops the WHOLE call when a required block is
    # missing: a turn carrying a perfectly good PATH arrived as `insert_after()`
    # with no arguments, and the observation said "missing: path, after" about a
    # turn that had supplied the path. The model apologised and re-sent it
    # identically until the stall detector fired. Two of the four eval tasks
    # were lost to this.
    #
    # The keyword still has to start a line, which is the check that stops a
    # `FIND:` inside the code being edited from being read as protocol.
    start = re.search(rf"^{keyword}:[ \t]*(?:\r?\n|(?=```))?", text, re.MULTILINE)
    if not start:
        return None
    rest = text[start.end():]

    # Markdown INLINE code -- the whole value between backtick runs on one
    # line: `AFTER: ```def total_count(self):``` `. Checked before the
    # multi-line fence logic below, which handles this shape catastrophically:
    # its opening-fence pattern needs a newline, so it runs past the value,
    # resumes on the NEXT line, and closes on the fence that OPENS the
    # following block -- parsing the anchor as the literal string "CONTENT:".
    # Captured live. An anchor of "CONTENT:" matches nothing, forever, and
    # every check upstream passed it.
    #
    # An opening fence proper cannot match here: "```" has nothing between its
    # backtick runs, and "```python" does not end in one.
    #
    # Only the triple-backtick form reaches this; the single-backtick form is
    # half-stripped by _normalize first and is dealt with in
    # _strip_loose_fences. Two spellings of one mistake, undone in two places
    # because normalization sees one of them and not the other.
    first_line, _, remainder = rest.partition("\n")
    inline = re.match(r"[ \t]*`+(?P<value>.*?)`+[ \t]*$", first_line)
    if inline and inline.group("value").strip() and _ends_the_block(remainder):
        return inline.group("value")

    # A block never runs past the keyword that starts the NEXT block. Models
    # forget the closing fence far more often than the opening one, and without
    # this the search for ``` sails straight through `REPLACE:` and closes on
    # the fence that OPENS the replacement -- welding the literal text
    # "REPLACE:" onto the end of FIND, where it can never match anything, and
    # losing the replacement entirely. Observed live on phi4-mini, on the
    # commonest edit there is: adding a docstring.
    #
    # Deliberately narrower than KEYWORDS. Code can contain a line beginning
    # "PATH:" or "THOUGHT:" -- this project's own prompt strings do -- so the
    # boundary is only the spellings that would be freakish inside real source:
    # a block keyword alone on its line, or the keywords that end the block
    # section outright.
    boundary = _BLOCK_BOUNDARY.search(rest)
    scope = rest[:boundary.start()] if boundary else rest

    # The body is everything up to the closing fence. Captured loosely and
    # trimmed afterwards so that an EMPTY block (```\n```) still matches --
    # an empty REPLACE is a legitimate deletion, not a malformed turn.
    fenced = re.match(r"[ \t]*```[^\r\n]*\r?\n(.*?)[ \t]*```", scope, re.DOTALL)
    if fenced:
        body = fenced.group(1)
        return _trim_blank_edges(body[:-1] if body.endswith("\n") else body)

    # An opening fence with nothing closing it before the next keyword: take
    # the rest of the block. Better a slightly over-long FIND that the exact
    # matcher can reject cleanly than one with "REPLACE:" welded onto the end.
    unclosed = re.match(r"[ \t]*```[^\r\n]*\r?\n(.*)", scope, re.DOTALL)
    if unclosed:
        return _trim_blank_edges(unclosed.group(1).rstrip("\n")) or None

    # No fence: run to the next protocol keyword, or the end of the message.
    stop = re.search(r"^(?:" + "|".join(KEYWORDS) + r"):", rest, re.MULTILINE)
    body = rest[:stop.start()] if stop else rest
    return _trim_blank_edges(_strip_loose_fences(body)) or None


def _trim_blank_edges(body: str) -> str:
    """
    Drop blank lines from the top and bottom of a block.

    Captured live, and the reason `docstring` failed two runs in five:

        AFTER: ```
        <- a blank line here
        def total_count(self):
        ```

    which parses as the two lines ["", "def total_count(self):"]. As an anchor
    that is over-specified -- it now demands a blank line ABOVE the def as well
    -- so it stops matching, and the model is told its anchor "does not appear
    in the file" when the line it named is right there.

    A blank line against a fence is layout in every block this parser has: in
    AFTER and FIND it breaks the match, and in CONTENT and REPLACE it inserts a
    stray empty line nobody asked for. An empty block stays empty, because an
    empty REPLACE is a deletion rather than a malformed turn.
    """
    lines = body.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _ends_the_block(remainder: str) -> bool:
    """
    Was that first line the whole block?

    Guards the inline-code shortcut. Taking a first line that merely *looks*
    like inline code would silently truncate a real multi-line block -- and in
    a Markdown file ``code`` on its own line is ordinary content, so this
    is reachable rather than theoretical. The shortcut only applies when
    nothing follows but the next keyword.
    """
    if not remainder.strip():
        return True
    return bool(re.match(r"[ \t]*(?:" + "|".join(KEYWORDS) + r"):", remainder))


def _strip_loose_fences(body: str) -> str:
    """
    Drop fence lines from a block that was never properly fenced to begin with.

    Models fence some blocks and not others in the same turn -- observed live:
    FIND fenced and closed correctly, REPLACE opening with no fence at all and
    then ending with a stray ```. Left in, that fence becomes the first line of
    the code written to the file. It is markup, never content, so it goes.
    """
    lines = body.split("\n")
    while lines and _BARE_FENCE.match(lines[0]):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _BARE_FENCE.match(lines[-1])):
        lines.pop()

    # Markdown INLINE code -- `AFTER: \`def total_count(self):\`` -- arrives
    # here already half-stripped, because _normalize eats decoration between
    # the keyword and its value but has no reason to look at the end of the
    # line. What is left is a value with a stray backtick run welded to it,
    # and that run goes into the file: an anchor of "def total_count(self):`"
    # matches nothing, and a docstring ending in a backtick is a SyntaxError.
    #
    # Captured live. The insertion was refused as "no longer valid Python",
    # which is true and gives the model nothing to work with -- it re-sent the
    # same turn until the stall detector fired, three runs in a row.
    # Stripped from both ends, because which end survives depends on whether
    # the backtick runs happened to balance: _iter_regions treats an odd run as
    # opening a fence and leaves everything after it alone, so the LAST inline
    # value in a turn keeps its opening run instead of its closing one. Either
    # way it is markup. A line of Python cannot begin or end with a backtick.
    if len(lines) == 1:
        lines[0] = re.sub(r"^[ \t]*`+|`+[ \t]*$", "", lines[0]).strip()
    return "\n".join(lines)


def parse_edit_form(text: str) -> dict:
    """
    Recover {path, find, replace} from the FIND:/REPLACE: spelling.

    An empty REPLACE block is legitimate -- that's a deletion -- so it is
    distinguished from a missing one rather than being treated as absent.
    """
    path_m = _PATH_LINE.search(text)
    if not path_m:
        return {}

    find = _fenced_after("FIND", text)
    if find is None:
        return {}

    replace = _fenced_after("REPLACE", text)
    return {
        "path": path_m.group(1).strip(),
        "find": find,
        "replace": "" if replace is None else replace,
    }
def parse_block_form(text: str) -> dict:
    """Recover {path, new_content} from the PATH:/CONTENT: spelling."""
    path_m = _PATH_LINE.search(text)
    if not path_m:
        return {}

    content = _fenced_after("CONTENT", text)
    if content is None:
        return {}

    if not content.endswith("\n"):
        content += "\n"          # files end with a newline; models forget
    return {"path": path_m.group(1).strip(), "new_content": content}


def parse_response(text: str) -> dict:
    result = {"thought": None, "confidence": None, "action": None, "args": None, "final": None}
    text = _normalize(text)

    # Keywords are only keywords at the start of a line -- otherwise a patch
    # whose contents mention "FINAL:" would end the run.
    stop = r"(?=^(?:" + "|".join(KEYWORDS) + r"):|\Z)"

    thought_m = re.search(r"^THOUGHT:[ \t]*(.+?)" + stop, text, re.DOTALL | re.MULTILINE)
    if thought_m:
        result["thought"] = thought_m.group(1).strip()

    conf_m = re.search(r"^CONFIDENCE:[ \t]*\"?([0-9]*\.?[0-9]+)", text, re.MULTILINE)
    if conf_m:
        value = float(conf_m.group(1))
        # Some models answer "85" meaning 85%.
        result["confidence"] = value / 100.0 if value > 1.0 else value

    final_m = re.search(r"^FINAL:[ \t]*(.+)", text, re.DOTALL | re.MULTILINE)
    action_m = re.search(r"^ACTION:[ \t]*`?(\w+)", text, re.MULTILINE)

    # Whichever comes FIRST wins. Small models routinely emit an action and
    # then, without waiting, invent the observation and a triumphant FINAL in
    # the same breath. Treating that as a finished run silently discards the
    # patch and reports success having changed nothing -- observed live. If an
    # action precedes the FINAL, the action is what the model actually decided
    # and the rest is it getting ahead of itself.
    if final_m and (action_m is None or final_m.start() < action_m.start()):
        result["final"] = final_m.group(1).strip()
        return result

    if action_m:
        result["action"] = action_m.group(1).strip()

    args_m = re.search(r"^ARGS:", text, re.MULTILINE)
    if args_m:
        result["args"] = parse_args_block(text[args_m.end():])

    # Fall back to the fenced spellings when JSON either wasn't there or didn't
    # survive parsing. Only for the two tools whose arguments are big enough
    # for escaping to defeat a small model.
    args = result["args"] or {}
    if result["action"] == "edit_file" and not args.get("find"):
        block = parse_edit_form(text)
        if block:
            result["args"] = block
    elif result["action"] == "apply_patch" and not args.get("new_content"):
        block = parse_block_form(text)
        if block:
            result["args"] = block
    elif result["action"] == "append_file" and not args.get("content"):
        block = parse_block_form(text)
        if block:
            # Same PATH:/CONTENT: shape, different argument name.
            result["args"] = {"path": block["path"], "content": block["new_content"]}
    elif result["action"] == "insert_after" and not args.get("content"):
        path_m = _PATH_LINE.search(text)
        after = _fenced_after("AFTER", text)
        if path_m and after is not None:
            # CONTENT is accepted as absent on purpose. Models merge the two
            # blocks -- writing the anchor and the new lines together under
            # AFTER, because that is what the finished code looks like. Taking
            # the turn as-is lets compute_insert split it, which it can do
            # unambiguously; rejecting it here would only produce another
            # "missing: content" the model has already shown it cannot act on.
            result["args"] = {"path": path_m.group(1).strip(),
                              "after": after,
                              "content": _fenced_after("CONTENT", text) or ""}

    return result


def _summarize_args(args: dict) -> str:
    """Print args without dumping an entire rewritten file into the terminal."""
    shown = {
        k: (v if not isinstance(v, str) or len(v) <= 60 else f"{v[:60]}... [{len(v)} chars]")
        for k, v in args.items()
    }
    return ", ".join(f"{k}={v!r}" for k, v in shown.items())


def build_system_prompt(repo_root: str) -> str:
    """
    The protocol half of the prompt: laid down once per session and never
    rewritten.

    Deliberately contains no repo state. The system prompt is the one message
    context.trim_history is forbidden to drop, so anything put here survives
    for the whole session -- which is right for the protocol and wrong for a
    file listing, because a stale listing is worse than none. Repo state goes
    in the per-task briefing instead, where each task gets a fresh one.
    """
    return f"""You are a careful coding agent working inside the repository at {repo_root}.

Respond using EXACTLY this protocol, ONE action per turn, nothing else:

THOUGHT: <your reasoning>
ACTION: <tool name>
ARGS: <json args on one line>

Available tools:
  list_dir(path)       - list a directory
  read_file(path)      - read a file's contents
  search_code(query)   - grep-like search across the repo
  edit_file(...)       - change lines that are already there
  insert_after(...)    - add lines below an existing line
  append_file(...)     - add a new function/class at the end of a file
  apply_patch(...)     - create a file that does not exist yet
  run_command(command) - run a shell command in the repo (e.g. tests)

WHICH WRITING TOOL -- get this right and the rest is easy:
  the file does not exist                    -> apply_patch
  ADDING a docstring, comment, guard or line -> insert_after
  ADDING a whole new function or class       -> append_file
  CHANGING code that is already there        -> edit_file

Only edit_file can delete anything. If you are adding, do not use it.

TO ADD A LINE -- a docstring, a comment, a guard, a print -- use insert_after.
AFTER holds an existing line, copied exactly. CONTENT holds only what is new.
The indentation is worked out for you:

THOUGHT: <your reasoning>
CONFIDENCE: <0-1>
ACTION: insert_after
PATH: example.py
AFTER:
```
def total_count(self):
```
CONTENT:
```
\"\"\"Return the total number of items.\"\"\"
```

This is the tool for "add a docstring to X". It cannot delete anything, so it
is always the safe choice when you are adding rather than replacing.

AFTER must hold the WHOLE line as it appears in the file, not just the name of
the thing. `def total_count(self):` -- not `total_count:`. Copy it.

TO CHANGE AN EXISTING FILE, use edit_file with this exact form:

THOUGHT: <your reasoning>
CONFIDENCE: <0-1, how sure you are the change is correct>
ACTION: edit_file
PATH: <path to the file>
FIND:
```
<the exact lines you want to replace, copied from the file>
```
REPLACE:
```
<what those lines should become>
```

Keep FIND as SHORT as possible -- just the lines that change, plus the minimum
needed to make them unique. Do NOT copy the whole function. Everything you do
not mention is left exactly as it is.

The FIND block must match the file EXACTLY, character for character, as it
appears right now. Copy it; do not retype or paraphrase it. If those lines
appear more than once in the file, include a few surrounding lines so the
match is unique.

Close EVERY fence. FIND's ``` must be closed before you write REPLACE:.

WORKED EXAMPLE. Say the file contains:

def divide(a, b):
    return a / b

and you must guard against b being zero. The whole function does NOT go in
FIND -- only the line you are changing around:

ACTION: edit_file
PATH: example.py
FIND:
```
def divide(a, b):
```
REPLACE:
```
def divide(a, b):
    if b == 0:
        raise ValueError("cannot divide by zero")
```

Note what REPLACE contains: the FIND line again, unchanged, then the new lines.

This would be WRONG, because it deletes the function:

REPLACE:
```
    if b == 0:
        raise ValueError("cannot divide by zero")
```

REPLACE is not "what to insert" -- it is what those exact lines BECOME. Every
line in FIND that you are keeping must appear in REPLACE as well. A line in
FIND and missing from REPLACE is DELETED.

Never end a Python block with a closing brace. Python uses indentation only.

TO ADD A NEW FUNCTION OR CLASS to a file that already exists, use append_file.
Write ONLY the new code -- nothing that is already in the file:

THOUGHT: <your reasoning>
CONFIDENCE: <0-1>
ACTION: append_file
PATH: <the existing file>
CONTENT:
```
def the_new_function(x):
    return x
```

Do NOT use edit_file to add a new function. Putting the old function in FIND
and the new one in REPLACE deletes the old one, which is never what you want.

TO CREATE A FILE THAT DOES NOT EXIST YET, use apply_patch:

THOUGHT: <your reasoning>
CONFIDENCE: <0-1, how sure you are the change is correct>
ACTION: apply_patch
PATH: <the new file's name, e.g. example.py>
CONTENT:
```
<the entire new file>
```

PATH must always be a FILENAME, never a directory and never ".". If the user
asked for a file by name, use exactly that name.

Write the working code FIRST. Keep docstrings to one short line, or leave them
out. A function that has a signature and a docstring but no body is not a
solution, and it will be rejected.

apply_patch on a file that already exists will be REFUSED. When you are asked
to fix, change, correct, rename, add to or remove from something, the file
already exists: do not write a new one, and do not invent a second file with a
similar name. Change the file you were given.

If you have not been shown the contents of the file you need to change, call
read_file on it first. Never edit a file you have not read.

Do exactly what the TASK line asks and nothing more. Do not borrow a task from
the examples above -- they are only there to show the format. If the task is a
question, answer it with FINAL once you have read enough to answer. If it asks
for no change at all, reply with FINAL immediately and change nothing.

Rules:
  - Exactly one ACTION per turn. Never emit two.
  - Every turn that writes to a file must include a CONFIDENCE line. Give a
    bare number and nothing else -- "CONFIDENCE: 0.8", not "CONFIDENCE: high".
  - Write code plainly inside the fences: real newlines, real quotes, no
    JSON, no escaping, no \\n.
  - For list_dir, read_file, search_code and run_command, use
    ARGS: <one line of raw JSON> instead.
  - Paths are relative to the repository root. Never use absolute paths.
  - Never invent tools outside the list above.
  - Do not write OBSERVATION: yourself. Stop after your ACTION and wait.

When the task is fully done, respond instead with:
THOUGHT: <reasoning>
FINAL: <plain-English summary of every file you changed and why>
"""


ANSWER_BRIEFING = """\
This is a QUESTION, not a change request. Answer it.

Read whatever you need with read_file, search_code and list_dir. When you know
the answer, reply with THOUGHT and then FINAL containing the answer itself --
in plain English, quoting the relevant lines if that helps.

Change NOTHING. edit_file, apply_patch, append_file and insert_after are
switched off for this turn and will be refused. Do not go looking for something
to improve; the user asked to be told something, not to have code rewritten.
"""


def build_task_briefing(task: str, index, context_files, mode: str = "edit") -> str:
    """
    The state half of the prompt, rebuilt for every task in the session.

    This is what makes a follow-up work. Previously the repo tree and file
    contents lived in the system prompt, which is written once -- so the second
    task in a session ("now fix the off-by-one in it") was handed nothing but
    the sentence itself, against a tree snapshotted before the first task had
    written anything. The model could not see the file it was being asked to
    change, so it did the only thing it could and wrote a new one.

    Files the task named by name are labelled as such, because a small model
    reading a wall of context needs telling which part of it is the target.
    """
    named = [f for f in context_files if getattr(f, "mentioned", False)]

    if mode == "answer":
        # The targeting paragraph below is all about which file to CHANGE, and
        # handing it to a question is how "what does the retriever score on?"
        # became four turns of edit_file with no FIND block. Answering needs
        # the contents and nothing else.
        blocks = "\n\n".join(f"--- {f.path} ---\n{f.snippet}"
                             for f in context_files) or "(nothing matched yet)"
        return (f"Current repository contents:\n{index.tree_str()}\n\n"
                f"Files that look relevant:\n{blocks}\n\n"
                f"{ANSWER_BRIEFING}\nQUESTION: {task}")

    blocks = []
    for f in context_files:
        label = "EXISTS -- change it with edit_file, do NOT recreate it"
        blocks.append(f"--- {f.path} ({label}) ---\n{f.snippet}")
    context_block = "\n\n".join(blocks) or (
        "(nothing matched automatically -- use list_dir, search_code and "
        "read_file to find the right file before you change anything)"
    )

    if named:
        listed = ", ".join(f.path for f in named)
        targeting = (
            f"\nThe task refers to {listed}, shown above with its current "
            "contents. That file already exists. Edit it with edit_file; do "
            "not create a new file.\n"
        )
    else:
        targeting = (
            "\nThe task does not name a file. Work out which existing file it "
            "is about from the tree and the contents above -- use read_file or "
            "search_code if you are unsure. Only create a new file if the task "
            "genuinely asks for one that is not there.\n"
        )

    return f"""Current repository contents (this is up to date, including any
changes made earlier in this session):
{index.tree_str()}

Files that look relevant to this task:
{context_block}
{targeting}
TASK: {task}"""


_LEADING_THOUGHT = re.compile(r"^\s*THOUGHT\s*:\s*", re.IGNORECASE)


def _prose_answer(text: str) -> str:
    """
    The answer inside a turn that forgot to say FINAL.

    Only the keyword itself is stripped. What follows is all answer -- the
    model was writing an explanation and labelled it THOUGHT, which is a
    labelling mistake and not a reason to throw any of it away.
    """
    return _LEADING_THOUGHT.sub("", text.strip()).strip()


PROTOCOL_REMINDER = (
    "Your last message did not follow the protocol. Reply with exactly:\n"
    "THOUGHT: <reasoning>\nACTION: <tool name>\nARGS: <one-line JSON>\n"
    "or THOUGHT: <reasoning>\nFINAL: <summary>"
)


def run(task: str,
        repo_root: str,
        backend: str | None = None,
        auto_approve: bool = False,
        messages: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
        mode: str = "edit") -> Iterator[ev.Event]:
    """
    Drive one task to completion, yielding events as it goes.

    `messages`, when passed, is a conversation carried over from earlier tasks
    in the same session -- this is what makes "now add a test for it" work. It
    is mutated in place so the caller keeps the updated history.

    `on_token` is handed straight to the backend for live streaming. It stays
    out of the event stream on purpose: a replayed run should not re-enact
    typing, and the assembled text arrives in Thought/ActionProposed anyway.
    """
    backend = backend or config.BACKEND
    model = {"gemini": config.GEMINI_MODEL,
             "ollama": config.OLLAMA_MODEL}.get(backend, "scripted")
    started = time.time()
    files_changed: list[str] = []

    yield ev.RunStarted(task=task, repo_root=repo_root, backend=backend,
                        model=model, mode=mode)

    index = build_index(repo_root)
    yield ev.IndexBuilt(file_count=len(index.files), tree=index.tree_str())

    retrieved = retrieve(task, index)
    fitted = ctx.fit_context_files(retrieved)
    yield ev.ContextRetrieved(
        files=[{"path": f.path, "score": f.score, "chars": len(f.snippet)} for f in fitted],
        dropped=len(retrieved) - len(fitted),
    )

    llm = get_backend(backend)

    # A fresh task in an existing session appends to the history rather than
    # rebuilding it; the system prompt is only laid down once.
    if messages is None:
        messages = []
    if not messages:
        messages.append({"role": "system", "content": build_system_prompt(repo_root)})

    # The briefing goes in on EVERY task, not just the first. The repo has
    # usually changed since the last one -- often because this agent changed it
    # -- and a follow-up like "now fix the bug in it" is unanswerable without a
    # current view of the file it refers to.
    messages.append({"role": "user",
                     "content": build_task_briefing(task, index, fitted, mode)})

    recent: list[str] = []
    calls: list[str] = []

    for step in range(1, config.MAX_STEPS + 1):
        yield ev.StepStarted(step=step, max_steps=config.MAX_STEPS)

        # Trim before the call, never after: this is the whole point of
        # context.py -- decide what to lose ourselves rather than letting the
        # runtime silently drop the system prompt off the front.
        sent = ctx.trim_history(messages)
        try:
            response = llm.chat(sent, on_token=on_token)
        except Exception as e:
            yield ev.RunFailed(error=str(e), step=step)
            return

        # A local model at temperature 0.1 is close to deterministic, so a turn
        # it got wrong it will get wrong again, identically, until the step cap.
        # Burning ten more steps to prove that helps nobody.
        recent.append(response.text.strip())
        if len(recent) >= config.REPEAT_LIMIT and len(set(recent[-config.REPEAT_LIMIT:])) == 1:
            yield ev.RunFailed(
                error=(f"the model repeated the same reply {config.REPEAT_LIMIT} times "
                       "and is not making progress. Usually this means it can't "
                       "produce the ARGS format for this task -- try a larger "
                       "model, or a smaller, more specific task."),
                step=step,
            )
            return

        parsed = parse_response(response.text)

        # In answer mode a turn with no ACTION is not a protocol violation, it
        # is the answer. Observed live: asked what the retriever scores on,
        # phi4-mini wrote a correct and detailed reply in plain prose, got
        # PROTOCOL_REMINDER for it, and produced "No action taken" on the
        # retry -- a good answer discarded for want of a five-character prefix.
        # Nothing can be written in this mode, so prose can be taken at face
        # value. Checked before the Thought event, because the parser reads an
        # unstructured turn as one long THOUGHT and rendering it as both a
        # thought and an answer would print the same paragraph twice.
        if (mode == "answer" and not parsed["action"] and not parsed["final"]
                and response.text.strip()):
            yield ev.Final(summary=_prose_answer(response.text),
                           steps_used=step, files_changed=[],
                           elapsed=time.time() - started)
            messages.append({"role": "assistant", "content": response.text})
            return

        if parsed["thought"]:
            yield ev.Thought(text=parsed["thought"])

        if parsed["final"]:
            yield ev.Final(summary=parsed["final"], steps_used=step,
                           files_changed=files_changed,
                           elapsed=time.time() - started)
            messages.append({"role": "assistant", "content": response.text})
            return

        action = parsed["action"]
        args = parsed["args"] or {}

        # The text-level check above only catches a byte-identical reply. A
        # model can also stall while rewording its THOUGHT every turn and
        # sending the same broken call underneath -- same cost, same outcome,
        # and invisible to a string comparison.
        if action is not None:
            calls.append(f"{action}|{sorted((k, str(v)) for k, v in args.items())}")
            if len(calls) >= config.REPEAT_LIMIT and \
                    len(set(calls[-config.REPEAT_LIMIT:])) == 1:
                yield ev.RunFailed(
                    error=(f"the model called {action} with the same arguments "
                           f"{config.REPEAT_LIMIT} times and got the same result "
                           "each time. It is stuck rather than working -- try a "
                           "more specific task, or name the file directly."),
                    step=step,
                )
                return

        if action is None:
            yield ev.ProtocolViolation(step=step, raw=response.text[:500],
                                       detail="no ACTION and no FINAL")
            observation = PROTOCOL_REMINDER
        else:
            # Keep the unparsed turn whenever the call came back with a hole in
            # it -- no arguments at all, or a block that parsed to nothing.
            # Both are the same question and neither can be answered from the
            # transcript: did the model omit that block, or did the parser fail
            # to read it? Four times in Phase 9 the answer was the parser, and
            # each one had to be reproduced live to find out.
            lossy = not args or any(v == "" for v in args.values())
            yield ev.ActionProposed(action=action, args=args,
                                    confidence=parsed["confidence"],
                                    raw=response.text[:2000] if lossy else "")

            if action in WRITE_TOOLS and mode == "answer":
                # Not a hint -- a wall. Telling a 3B model not to edit gets it
                # to agree and then edit anyway; refusing the call is the only
                # version of "read-only" that is actually read-only.
                observation = (
                    f"{action} is switched off: this task is a question, not a "
                    "change request. Nothing is to be modified. Read what you "
                    "need with read_file or search_code, then answer with:\n"
                    "THOUGHT: <reasoning>\nFINAL: <the answer to the question>"
                )
            elif action in WRITE_TOOLS:
                outcome = yield from _handle_patch(
                    task=task, args=args, repo_root=repo_root, backend=backend,
                    response_text=response.text, auto_approve=auto_approve,
                    tool=action,
                )
                observation, applied, always = outcome
                if applied:
                    files_changed.append(args["path"])
                if always:
                    auto_approve = True
            elif action in tools.DISPATCH:
                missing = [k for k in READ_REQUIRED_ARGS.get(action, ())
                           if not str(args.get(k, "")).strip()]
                if missing:
                    observation = (
                        f"{action} needs {', '.join(missing)}. You sent no ARGS "
                        f"at all. Copy this line and fill it in:\n"
                        f"{READ_USAGE_HINT[action]}"
                    )
                else:
                    try:
                        observation = tools.DISPATCH[action](args, repo_root)
                    except Exception as e:
                        # Defence in depth. Every tool already converts its own
                        # expected failures into observations, but `args` is a
                        # dict a language model made up -- a value of the wrong
                        # *type* can still raise something no tool anticipated,
                        # and one bad turn must not end the run.
                        observation = (f"ERROR: {action} failed with "
                                       f"{type(e).__name__}: {e}")
            else:
                observation = (f"Unknown tool: {action}. Available tools: "
                               f"{', '.join(sorted(tools.DISPATCH))}.")

        yield ev.Observation(text=observation, action=action or "")
        messages.append({"role": "assistant", "content": response.text})
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    yield ev.StepLimitReached(max_steps=config.MAX_STEPS)


REQUIRED_ARGS = {
    "apply_patch": ("path", "new_content"),
    "edit_file": ("path", "find"),
    "append_file": ("path", "content"),
    # `content` is deliberately not required: see compute_insert, which
    # recovers it from a merged AFTER block.
    "insert_after": ("path", "after"),
}

# The read tools take one line of JSON, so their whole call fits in the hint.
# Naming the missing key alone is not enough: told "missing required argument
# 'query'", phi4-mini re-sent a bare `search_code()` three turns running. It
# knew what was missing and not how to supply it, and no amount of repeating
# the diagnosis was going to teach it. Handing back a line it can copy does.
READ_REQUIRED_ARGS = {
    "read_file": ("path",),
    "search_code": ("query",),
    "run_command": ("command",),
}

READ_USAGE_HINT = {
    "read_file": 'ACTION: read_file\nARGS: {"path": "arthur/agent.py"}',
    "search_code": 'ACTION: search_code\nARGS: {"query": "def retrieve"}',
    "run_command": 'ACTION: run_command\nARGS: {"command": "python -m pytest -q"}',
    "list_dir": 'ACTION: list_dir\nARGS: {"path": "."}',
}

USAGE_HINT = {
    "apply_patch": "ACTION: apply_patch\nPATH: <file>\nCONTENT:\n```\n<entire file>\n```",
    "edit_file": ("ACTION: edit_file\nPATH: <file>\nFIND:\n```\n<exact lines to replace>\n```"
                  "\nREPLACE:\n```\n<the new lines>\n```"),
    "append_file": ("ACTION: append_file\nPATH: <file>\nCONTENT:\n```\n"
                    "<the new function, on its own>\n```"),
    "insert_after": ("ACTION: insert_after\nPATH: <file>\nAFTER:\n```\n"
                     "<the existing line to insert below>\n```\nCONTENT:\n```\n"
                     "<the new lines>\n```"),
}

# The tools that reach disk, and so go through the gate rather than DISPATCH.
WRITE_TOOLS = ("apply_patch", "edit_file", "append_file", "insert_after")


def _echo_file(path: str, content: str) -> str:
    """The current text of a file, for an observation that has to teach."""
    if not content or len(content) > config.EDIT_ECHO_MAX_CHARS:
        return ""
    return f"\nHere is {path} exactly as it is right now:\n```\n{content}```\n"


def _edit_error(error: str, args: dict, repo_root: str) -> str:
    """Report a failed edit alongside the text the model should have matched."""
    parts = [f"ERROR: {error}"]
    try:
        content = tools.current_content(args, repo_root)
    except (tools.PathEscape, tools.BadTarget, OSError):
        return parts[0]

    echo = _echo_file(args["path"], content)
    if echo:
        parts.append(echo + "Copy the lines you want to change from this, "
                            "character for character, into a new FIND block.")
    return "\n".join(parts)


def _overwrite_refusal(path: str, content: str) -> str:
    """
    Refuse a whole-file rewrite of a file that already exists.

    The failure this closes is the one that showed up most in real use: asked
    to correct a file, the model reaches for apply_patch and re-emits the file
    from memory. Sometimes that is fine. Often it drops a function it wasn't
    thinking about, or invents a near-duplicate file, and the user's actual
    edit is buried in a rewrite nobody asked for.

    Refusing outright rather than warning is the point. The structural gate
    already catches rewrites that delete things, but it catches them by
    stopping to ask a human -- which turns every correction into an approval
    prompt. Closing the path entirely means the model has to name the lines it
    is changing, and a change it can name is one it cannot accidentally make.

    Rewriting the whole file is still reachable: edit_file with the entire
    current text in FIND does exactly that, deliberately.
    """
    return (
        f"ERROR: {path} already exists, so apply_patch is refused -- it would "
        "overwrite the file with a version written from memory.\n"
        "Use edit_file and change only the lines that need to change."
        + _echo_file(path, content) +
        "Copy the lines you want to change from this, character for "
        "character, into FIND, and put what they should become in REPLACE."
    )


# Deleting one definition while adding another is the signature of "add a
# second function" done with the wrong tool. Observed live three turns running,
# each blocked, never recovering -- because the advice it got was about REPLACE
# blocks when the real answer was a different tool.
_APPEND_INSTEAD = (
    "It looks like you are ADDING a function, not changing one. edit_file "
    "replaces the lines in FIND, so putting the old function in FIND and a new "
    "one in REPLACE deletes the old one. Use append_file instead, with ONLY "
    "the new function in CONTENT."
)


def _replace_hint(args: dict, tool: str, gate) -> str:
    """
    Say what to do differently, in terms of what actually went wrong.

    Generic advice does not land on a small model. Told only that a patch was
    rejected, phi4-mini repeated the identical turn until the repeat detector
    stopped the run -- twice, for two different reasons. So each structural
    signal gets its own concrete instruction, and where possible the model's
    own text is quoted back at it rather than paraphrased.
    """
    shape_warnings = " ".join(gate.structural_warnings)

    if "no code in the body" in shape_warnings:
        # Nothing was lost -- the write just stopped after the docstring.
        # Telling this one to "keep existing definitions" would be answering a
        # question it did not ask.
        return (f"You wrote a signature and a docstring but no working code. "
                f"Send the SAME {tool} again with the function body filled in: "
                "the actual statements that compute the answer and return it. "
                "Keep the docstring to one line.")

    if tool != "edit_file" or not args.get("find"):
        return ("Try again, keeping every function and class that already "
                "exists in the file.")

    # Ordered most specific first. Each of these is a different mistake with a
    # different fix, and naming the wrong one is worse than saying nothing:
    # told to check its indentation, the model corrected the indentation and
    # dropped the docstring it was supposed to be adding.
    if "removes existing definitions" in shape_warnings and "def " in args.get("replace", ""):
        return _APPEND_INSTEAD

    find_anchor = next((ln.strip() for ln in args["find"].split("\n") if ln.strip()), "")
    replace_lines = {ln.strip() for ln in args.get("replace", "").split("\n")}
    if find_anchor and find_anchor not in replace_lines:
        # Almost always an INSERT written as a replace. Rather than explaining
        # the difference again, hand over the tool that means what the model
        # already meant -- and hand it back its own two blocks, ready to send.
        return (f"Your REPLACE block does not contain the line `{find_anchor}`, "
                "so that line would be deleted. It looks like you want to ADD "
                "lines, not replace them. Use insert_after:\n"
                f"ACTION: insert_after\nPATH: {args['path']}\nAFTER:\n"
                f"```\n{find_anchor}\n```\nCONTENT:\n```\n"
                f"{args.get('replace', '').strip()}\n```")

    if "valid Python" in shape_warnings:
        # Nothing was deleted -- the result just doesn't parse, and by far the
        # commonest reason is indentation: the REPLACE block written flush left
        # when the code it replaces sits inside a class or a loop.
        return ("The result does not parse. Check the INDENTATION of your "
                "REPLACE block: every line must be indented to match the code "
                "around it. If you are editing a method inside a class, its "
                "body is indented by 8 spaces, not 4.")

    # The other live failure: asked to add a docstring, the model put
    # `def two_sum(nums, target):` in FIND and ONLY the docstring in REPLACE,
    # deleting the signature. Showing the exact lines it must repeat is a
    # concrete instruction rather than a principle to apply.
    return (
        "Your REPLACE block must contain everything from FIND that you are "
        "KEEPING, plus your change. Anything in FIND but missing from REPLACE "
        "is deleted. You searched for:\n"
        f"```\n{args['find']}\n```\n"
        "so REPLACE must contain those lines too, unless you meant to remove "
        "them."
    )


def _handle_patch(task: str, args: dict, repo_root: str, backend: str,
                  response_text: str, auto_approve: bool, tool: str = "apply_patch"):
    """
    The gated write path, shared by both writing tools. Returns
    (observation, applied, set_auto_approve).

    Split out of `run` because it is the part that matters: everything else in
    the loop is plumbing, and this is where the project's actual claim lives.
    Both tools converge here on (old_content, new_content), so the critic and
    the structural veto behave identically regardless of how the model chose
    to express the change.
    """
    missing = [k for k in REQUIRED_ARGS[tool] if k not in args]
    if missing:
        return (f"{tool} is missing: {', '.join(missing)}. Use this form:\n"
                f"{USAGE_HINT[tool]}", False, False)

    try:
        if tool == "insert_after":
            old_content, new_content, error = tools.preview_insert(args, repo_root)
            if not old_content:
                return (f"ERROR: {args['path']} does not exist yet. Use "
                        "apply_patch to create it.", False, False)
            if error:
                return _edit_error(error, args, repo_root), False, False
        elif tool == "append_file":
            old_content, new_content = tools.preview_append(args, repo_root)
            if not old_content:
                return (f"ERROR: {args['path']} does not exist yet, so there is "
                        "nothing to append to. Use apply_patch to create it.",
                        False, False)
        elif tool == "edit_file":
            old_content, new_content, error = tools.preview_edit(args, repo_root)
            if not old_content:
                # Editing a file that isn't there. Without this the empty file
                # falls through to the matcher and comes back "the FIND text
                # does not appear in the file" -- true, and useless: the model
                # reads it as a matching problem, rewrites the FIND block, and
                # gets the identical answer until the repeat detector fires.
                # Observed live, five turns in a row. Name the actual problem.
                return (f"ERROR: {args['path']} does not exist yet, so there is "
                        "nothing to edit. Use apply_patch to create it, with "
                        "the whole file in CONTENT.", False, False)
            if error:
                # A failed match is not a patch -- it never reaches the gate.
                # Hand the file back with the error: the commonest miss is the
                # model writing what it wants the code to BECOME instead of
                # copying what is there, and it cannot fix that from a scolding
                # alone. Showing the current text turns an unrecoverable loop
                # into a correctable mistake.
                return _edit_error(error, args, repo_root), False, False
        else:
            old_content = tools.current_content(args, repo_root)
            if old_content.strip():
                return _overwrite_refusal(args["path"], old_content), False, False
            new_content = args["new_content"]
    except (tools.PathEscape, tools.BadTarget) as e:
        return f"ERROR: {e}", False, False
    except OSError as e:
        # Every argument here was invented by a language model, so unreadable
        # targets and permission errors are ordinary inputs, not exceptional
        # ones. They belong in the observation, not in a traceback.
        return f"ERROR: cannot read {args.get('path')!r}: {e}", False, False

    if new_content == old_content:
        # An edit that changes nothing is not a success, and left alone it
        # reads as one: the tool reports EDITED, the model says the task is
        # complete, and the file is untouched. Observed live -- told its
        # REPLACE block had an indentation problem, the model fixed the
        # indentation and dropped the docstring it was supposed to be adding,
        # producing an exact copy of what was already there.
        return (f"ERROR: that edit would leave {args['path']} exactly as it is "
                "-- the REPLACE block is identical to the FIND block. Nothing "
                "was applied. Make REPLACE contain the change you intend.",
                False, False)

    diff_text = patcher.unified_diff(old_content, new_content, args["path"])
    yield ev.DiffReady(path=args["path"], diff=diff_text)

    gate = safety_gate.evaluate(
        task=task,
        coder_response_text=response_text,
        diff_text=diff_text,
        backend_name=backend,
        old_content=old_content,
        new_content=new_content,
        path=args["path"],
    )
    yield ev.GateDecision(
        self_confidence=gate.self_confidence,
        critic_verdict=gate.critic_verdict,
        critic_reason=gate.critic_reason,
        score=gate.score,
        threshold=config.CONFIDENCE_THRESHOLD,
        needs_human=gate.needs_human,
        structural_warnings=gate.structural_warnings,
        blocked_structurally=gate.blocked_structurally,
    )

    if not gate.needs_human:
        yield ev.ApprovalResolved(decision=ev.Decision.APPLY.value,
                                  reason="cleared the gate", automatic=True)
        return tools.DISPATCH[tool](args, repo_root), True, False

    # auto-approve covers borderline confidence, NOT active objections. `-y`
    # means "stop asking me about judgement calls", not "turn the safety system
    # off". Two things it deliberately cannot override:
    #
    #   - a structural block: deleting definitions or breaking the file needs a
    #     human, and if nobody is there to look, the answer is no;
    #   - an outright critic REJECT: that is a reviewer actively saying the
    #     change is wrong, which is different from a score sitting just under a
    #     threshold. Observed live -- with -y on, a REJECTed edit (score 0.25)
    #     was applied and corrupted a file the agent had just written correctly.
    vetoed = gate.blocked_structurally or gate.critic_verdict == "REJECT"
    if auto_approve and not vetoed:
        yield ev.ApprovalResolved(decision=ev.Decision.APPLY.value,
                                  reason="auto-approve is on", automatic=True)
        return tools.DISPATCH[tool](args, repo_root), True, False

    reasons = list(gate.structural_warnings)
    if gate.score < config.CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {gate.score:.2f} is below the "
                       f"{config.CONFIDENCE_THRESHOLD} threshold")

    reply = yield ev.ApprovalNeeded(
        path=args["path"], diff=diff_text,
        score=gate.score, threshold=config.CONFIDENCE_THRESHOLD,
        reasons=reasons, blocked_structurally=gate.blocked_structurally,
    )

    decision, reason = _unpack_reply(reply)
    yield ev.ApprovalResolved(decision=decision.value, reason=reason)

    if decision is ev.Decision.REJECT:
        # The reason goes back to the model as the observation. That is the
        # difference between a roadblock and a conversation: it gets to try
        # again knowing what was wrong with the first attempt.
        parts = ["The patch was NOT applied."]
        if gate.structural_warnings:
            # These are machine-checked facts, not a human's paraphrase, and
            # they name exactly what went wrong -- which is far more useful to
            # the model than "rejected". Dropped definitions are the failure a
            # small model makes most often and is most able to fix when told.
            parts.append("Automatic check found: " + "; ".join(gate.structural_warnings) + ".")
            parts.append(_replace_hint(args, tool, gate))
        if reason:
            parts.append(f"Reviewer said: {reason}")
        return " ".join(parts), False, False

    return tools.DISPATCH[tool](args, repo_root), True, decision is ev.Decision.ALWAYS


def _unpack_reply(reply) -> tuple[ev.Decision, str]:
    """
    Accept what a consumer might plausibly send back: a Decision, a
    (Decision, reason) pair, a bare string, or nothing at all.

    Nothing at all means a consumer that isn't answering approvals -- a
    transcript replay, or a test. Failing closed (reject) is the only safe
    reading of silence when the question is "may I write to disk".
    """
    if reply is None:
        return ev.Decision.REJECT, "no answer from the caller"
    if isinstance(reply, tuple):
        decision, reason = (list(reply) + [""])[:2]
        return ev.Decision(decision), reason or ""
    return ev.Decision(reply), ""
