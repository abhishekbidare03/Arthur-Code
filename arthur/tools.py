"""
Tool implementations. Each tool takes a dict of args and the repo root,
and returns a plain string "observation" that gets fed back to the model.

Two-step write: `apply_patch` does NOT touch disk directly -- it goes
through the safety gate in agent.py first. By the time apply_patch's
underlying `_write` runs, the gate has already approved it.
"""

import ast
import os
import re
import subprocess
import textwrap

from . import config
from .patcher import unified_diff


class PathEscape(Exception):
    """A tool argument pointed outside the repository."""


def resolve(root: str, path: str) -> str:
    """
    Resolve a model-supplied path against the repo root, refusing to leave it.

    `os.path.join(root, path)` is not containment: given an absolute path it
    discards the root entirely, and given "../.." it walks straight out. Both
    show up in practice -- the very first live local run had the model echo
    back an absolute path it had seen in the prompt. That was harmless here,
    but the same code path is how an agent ends up writing to C:\\Windows.
    """
    full = os.path.abspath(os.path.join(root, path))
    root_abs = os.path.abspath(root)
    if os.path.normcase(full) != os.path.normcase(root_abs) and \
            not os.path.normcase(full).startswith(os.path.normcase(root_abs) + os.sep):
        raise PathEscape(
            f"path {path!r} resolves outside the repository. "
            "Use a path relative to the repo root."
        )
    return full


def _safe(fn):
    """
    Turn any tool failure into an observation instead of a crash.

    A tool is handed arguments invented by a language model, so every one of
    these is reachable: paths outside the repo, missing keys, directories where
    a filename belongs, permission errors. The model should be told what went
    wrong and get another turn -- none of it is worth killing an interactive
    session the user has been building context in.
    """
    def wrapped(args: dict, root: str) -> str:
        try:
            return fn(args, root)
        except (PathEscape, BadTarget) as e:
            return f"ERROR: {e}"
        except KeyError as e:
            return f"ERROR: missing required argument {e}"
        except OSError as e:
            return f"ERROR: {e}"
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


@_safe
def list_dir(args: dict, root: str) -> str:
    path = resolve(root, args.get("path", "."))
    try:
        entries = sorted(os.listdir(path))
    except OSError as e:
        return f"ERROR: {e}"
    return "\n".join(entries) if entries else "(empty directory)"


@_safe
def read_file(args: dict, root: str) -> str:
    path = resolve(root, args["path"])
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    except OSError as e:
        return f"ERROR: {e}"
    return content[: config.MAX_FILE_CHARS]


@_safe
def search_code(args: dict, root: str) -> str:
    query = args["query"]          # KeyError here is handled by @_safe
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS]
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if query.lower() in line.lower():
                            rel = os.path.relpath(full, root)
                            hits.append(f"{rel}:{lineno}: {line.strip()}")
            except OSError:
                continue
    return "\n".join(hits[:50]) if hits else "No matches."


class BadTarget(Exception):
    """The path is not something we can write a file to."""


def check_target(root: str, path: str) -> str:
    """
    Resolve a write target, refusing anything that isn't a file path.

    Models name a directory more often than you'd think -- phi4-mini answered
    "create twosum.py" with `path: "."`. Opening a directory raises
    PermissionError on Windows and IsADirectoryError on POSIX, and an
    unguarded one takes the whole interactive session down with it.
    """
    if not path or not path.strip():
        raise BadTarget("no path was given. Provide a filename, e.g. example.py")

    full = resolve(root, path)
    if os.path.isdir(full):
        raise BadTarget(
            f"{path!r} is a directory, not a file. Give the full filename you "
            "want to write, e.g. example.py"
        )
    return full


def current_content(args: dict, root: str) -> str:
    """The file as it stands, or "" if it doesn't exist yet."""
    path = check_target(root, args.get("path", ""))
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError as e:
        raise BadTarget(f"cannot read {args.get('path')!r}: {e}") from e


def preview_patch(args: dict, root: str) -> str:
    """Compute the diff WITHOUT writing -- used by the safety gate/critic."""
    return unified_diff(current_content(args, root), args["new_content"], args["path"])


@_safe
def apply_patch(args: dict, root: str) -> str:
    """Actually write to disk. Only called after the safety gate approves."""
    path = check_target(root, args["path"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(args["new_content"])
    rel = os.path.relpath(path, os.path.abspath(root))
    return f"WROTE {rel} ({len(args['new_content'])} chars)"


def _normalize_ws(text: str) -> str:
    """Line-wise trailing-whitespace strip, for the second matching attempt."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))


_CLOSER_LINE = re.compile(r"^[ \t]*[}\])]+[ \t]*;?[ \t]*$")


def strip_stray_closers(find: str) -> str:
    """
    Drop trailing brace-only lines that FIND opened nothing to justify.

    Models trained mostly on C-family code end a Python block with a `}`. It
    is one character, it is invisible in a diff summary, and it makes the FIND
    text unmatchable forever -- observed live, phi4-mini closing a Python
    function with a brace and then repeating the identical mistake until the
    step limit, because "the FIND text does not appear in the file" gives it no
    clue which character is wrong.

    Only UNBALANCED closers are dropped, and only from the end. A `}` that
    closes a `{` opened inside the same FIND block is real code -- the last
    line of a multi-line dict literal is exactly that -- and is left alone. So
    this only ever removes a bracket that could not have been valid Python in
    the first place.
    """
    lines = find.replace("\r\n", "\n").split("\n")
    while lines:
        candidate = lines[-1]
        if not candidate.strip():
            break
        if not _CLOSER_LINE.match(candidate):
            break
        body = "\n".join(lines[:-1])
        closers = candidate.strip().rstrip(";").strip()
        opens = sum(body.count(c) for c in "{[(")
        closes = sum(body.count(c) for c in "}])")
        if opens >= closes + len(closers):
            break                    # it closes something; that's real code
        lines = lines[:-1]
    return "\n".join(lines)


def _split(text: str) -> list[str]:
    """Lines of `text`, without the empty string a trailing newline produces."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _dedent(lines: list[str]) -> list[str]:
    """Strip the deepest indentation common to every non-blank line."""
    widths = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    common = min(widths) if widths else 0
    return [ln[common:] if ln.strip() else "" for ln in lines]


def _anchored_matches(old_lines: list[str], find_lines: list[str],
                      exact: bool) -> list[tuple[int, str]]:
    """
    Where `find_lines` occurs as whole lines. Returns (start, indent) per hit.

    With `exact`, the lines must match verbatim and `indent` is always "".
    Otherwise every line may be shifted right by the SAME prefix -- which is
    how a model's mental image of a method ("def total_count(self):") lines up
    with the file's reality ("    def total_count(self):"). One uniform shift,
    never a per-line guess: the block's internal shape has to be right already.
    """
    n = len(find_lines)
    if not n or n > len(old_lines):
        return []

    hits = []
    for i in range(len(old_lines) - n + 1):
        indent, ok = None, True
        for actual, wanted in zip(old_lines[i:i + n], find_lines):
            actual, wanted = actual.rstrip(), wanted.rstrip()
            if not wanted:
                ok = not actual
                if not ok:
                    break
                continue
            if not actual.endswith(wanted):
                ok = False
                break
            prefix = actual[: len(actual) - len(wanted)]
            if prefix.strip() or (exact and prefix):
                ok = False
                break
            if indent is None:
                indent = prefix
            elif prefix != indent:
                ok = False
                break
        if ok:
            hits.append((i, indent or ""))
    return hits


def _splice_lines(old: str, old_lines: list[str], start: int, length: int,
                  replace: str, indent: str, shift: bool) -> str:
    """
    Put `replace` in at `start`.

    `shift` only when the match itself was shifted -- i.e. FIND was written
    without the indentation the file actually has. Then REPLACE is dedented to
    its own margin and moved to where the match sat, so the model's block keeps
    its internal shape and lands at the right depth. When FIND matched
    verbatim, REPLACE already carries the right indentation and re-indenting it
    would push a correct block one level too far right.
    """
    block = _split(replace)
    if shift:
        block = [indent + ln if ln.strip() else "" for ln in _dedent(block)]
    merged = "\n".join(old_lines[:start] + block + old_lines[start + length:])
    # _split dropped the empty string a trailing newline leaves behind; put the
    # newline back, because a file that ended with one still should.
    return merged + "\n" if old.endswith("\n") else merged


def compute_edit(old: str, find: str, replace: str) -> tuple[str | None, str]:
    """
    Apply a search/replace edit. Returns (new_content, error).

    Exact, on purpose. The whole reason this tool exists is that a small model
    cannot be trusted to reproduce code it did not intend to change -- so the
    model supplies only the fragment it *is* changing, and we refuse anything
    we cannot locate. There is no "close enough" path, which is what makes
    silent corruption structurally impossible rather than something we detect
    afterwards.

    Matching is LINE-oriented first and substring-oriented only as a fallback,
    and that order is load-bearing. A plain `old.count(find)` treats
    "def total_count(self):" as a hit inside "    def total_count(self):" --
    a match in the middle of a line -- and splices the replacement after four
    spaces that are no longer accounted for. Observed live: every line of the
    replacement then landed one level short, the file stopped parsing, and the
    model was told to fix a REPLACE block that was already correct. It could
    not see the four spaces, so it never recovered.

    Matching whole lines instead lets us record the indentation the match sat
    at and put the replacement back at the same depth. The model gets to think
    about the code, not about the column it happens to live in.

    Failures are reported distinctly because the model can act on them
    differently: not found (it invented or paraphrased the snippet) versus
    ambiguous (the snippet is real but appears more than once, so it needs to
    include more surrounding context).
    """
    if not find:
        return None, "FIND block is empty. It must contain the exact text to replace."

    old_lines = _split(old)
    find_lines = _split(find)

    def ambiguous(count):
        return None, (
            f"the FIND text appears {count} times, so the edit is ambiguous. "
            "Include more surrounding lines to make it unique."
        )

    # 1. Whole lines, verbatim. The model reproduced the file exactly.
    # 2. Whole lines, shifted right by one uniform indent.
    #
    # Tried in this order so that a FIND which already matches verbatim can
    # never be re-read as an indented match somewhere else -- "    return x"
    # is unique as written but would collide with "        return x" once
    # dedented, and turning a working edit into "ambiguous" is a regression.
    for exact in (True, False):
        wanted = find_lines if exact else _dedent(find_lines)
        hits = _anchored_matches(old_lines, wanted, exact=exact)
        if len(hits) == 1:
            start, indent = hits[0]
            return _splice_lines(old, old_lines, start, len(find_lines),
                                 replace, indent, shift=not exact), ""
        if len(hits) > 1:
            return ambiguous(len(hits))

    # 3. Substring, for an edit that genuinely sits inside a line.
    count = old.count(find)
    if count == 1:
        return old.replace(find, replace, 1), ""
    if count > 1:
        return ambiguous(count)

    # 4. Last tolerance: drop a stray closing brace the model appended out of
    # C-family habit, and try the line matchers again. Applied to REPLACE as
    # well -- it is one habit, not two mistakes, and repairing only the search
    # half would splice the brace into the file and guarantee a syntax error.
    destrayed = strip_stray_closers(find)
    if destrayed != find and destrayed.strip():
        destrayed_lines = _split(destrayed)
        for exact in (True, False):
            wanted = destrayed_lines if exact else _dedent(destrayed_lines)
            hits = _anchored_matches(old_lines, wanted, exact=exact)
            if len(hits) == 1:
                start, indent = hits[0]
                return _splice_lines(old, old_lines, start, len(destrayed_lines),
                                     strip_stray_closers(replace), indent,
                                     shift=not exact), ""

    return None, (
        "the FIND text does not appear in the file. It must match the file's "
        "CURRENT contents exactly -- copy the lines as they are now, not as "
        "you want them to end up."
    )


@_safe
def edit_file(args: dict, root: str) -> str:
    """
    Replace one exact fragment of a file. The preferred way to change code:
    the model writes only what it is changing, so it cannot delete the rest of
    the file by failing to retype it.
    """
    path = check_target(root, args["path"])
    if not os.path.exists(path):
        return (f"ERROR: {args['path']} does not exist. "
                "Use apply_patch to create a new file.")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        old = fh.read()

    new, error = compute_edit(old, args.get("find", ""), args.get("replace", ""))
    if new is None:
        return f"ERROR: {error}"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    rel = os.path.relpath(path, os.path.abspath(root))
    return f"EDITED {rel}"


def compute_insert(old: str, after: str, content: str) -> tuple[str | None, str]:
    """
    Put `content` in immediately below the line(s) `after`. Returns (new, error).

    The mental model this exists to serve is "insert this text after that
    line", and it is the one a model reaches for constantly -- adding a
    docstring, a guard, a log line, a comment. edit_file cannot express it
    without repeating the anchor inside REPLACE, and models will not reliably
    do that: told in as many words that REPLACE must contain the `def` line,
    phi4-mini went on writing REPLACE blocks holding a docstring and nothing
    else, run after run, failing a different way each time. append_file settled
    the equivalent argument for "add a new function" the moment it existed.

    Safe by construction, like append_file: every existing line is kept and the
    new lines go between them. There is no argument this tool can be given that
    removes a single character of what is already in the file.

    The insertion depth comes from the line BELOW the anchor rather than from
    the anchor itself, which needs no knowledge of Python: text going in under
    `def f():` lands at the depth of that function's body, and text going in
    under an ordinary statement lands beside it.
    """
    if not after.strip():
        return None, "AFTER block is empty. It must contain the line to insert below."

    old_lines = _split(old)
    after_lines = _split(after)

    if not content.strip():
        # A merged block: the model wrote the anchor and the new lines together
        # under AFTER, because that is what the finished code looks like.
        # Observed live, five turns running, each answered with "missing:
        # content" -- which it plainly could not act on. The split is not a
        # guess: the longest run of leading lines that is actually IN the file
        # is the anchor, and what follows cannot be, or it would have matched
        # too. The alternative reading -- that the model wanted to insert
        # nothing -- is never what anyone means.
        for cut in range(len(after_lines) - 1, 0, -1):
            head = after_lines[:cut]
            if any(len(_anchored_matches(old_lines, w, exact=e)) == 1
                   for e, w in ((True, head), (False, _dedent(head)))):
                after_lines, content = head, "\n".join(after_lines[cut:])
                break
        else:
            # A single-line AFTER with nothing to insert. There is no merged
            # block to split and no way to invent the missing lines, so the
            # only useful reply is the model's own turn handed back with the
            # gap marked -- the same move that fixed the dropped anchor. Simply
            # restating the rule does not work: observed live, the model
            # re-sent the identical turn until the stall detector fired.
            anchor = "\n".join(after_lines)
            return None, (
                "CONTENT is empty, so there is nothing to insert. AFTER is the "
                "line you are inserting BELOW; CONTENT is the new text. Send "
                "this again with CONTENT filled in:\n"
                f"ACTION: insert_after\nPATH: <the file>\nAFTER:\n```\n{anchor}"
                "\n```\nCONTENT:\n```\n<the line to add -- this part was "
                "missing>\n```"
            )

    for exact in (True, False):
        wanted = after_lines if exact else _dedent(after_lines)
        hits = _anchored_matches(old_lines, wanted, exact=exact)
        if len(hits) > 1:
            return None, (
                f"the AFTER text appears {len(hits)} times, so the insertion "
                "point is ambiguous. Include more surrounding lines."
            )
        if len(hits) == 1:
            start, anchor_indent = hits[0]
            end = start + len(after_lines)

            # The deeper of the last anchor line and the line below it, both
            # as they appear in the FILE. Two cases, one rule: after
            # `def f():` the line below is the function body and is deeper, so
            # the insertion joins the body; after the last statement of a
            # block the line below dedents out of it, so the insertion stays
            # at the anchor's own depth. Taking the deeper of the two is right
            # both times, and needs no idea of what the code means.
            def _indent_of(line):
                return line[: len(line) - len(line.lstrip())]

            anchored = [ln for ln in old_lines[start:end] if ln.strip()]
            indent = _indent_of(anchored[-1]) if anchored else anchor_indent

            following = next((ln for ln in old_lines[end:] if ln.strip()), None)
            if following is not None:
                below = _indent_of(following)
                if len(below) > len(indent):
                    indent = below

            block = [indent + ln if ln.strip() else ""
                     for ln in _dedent(_split(content))]
            merged = "\n".join(old_lines[:end] + block + old_lines[end:])
            return (merged + "\n" if old.endswith("\n") else merged), ""

    return _abbreviated_anchor(old, old_lines, after_lines, content)


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _abbreviated_anchor(old, old_lines, after_lines, content):
    """
    Last resort for a one-line anchor that names its target without quoting it.

    Captured live: asked to document `total_count`, phi4-mini sent
    `AFTER: total_count:` -- the method's name and a colon, where the file says
    `    def total_count(self):`. Not a matching problem so much as an
    abbreviation, and answering "the AFTER text does not appear in the file"
    told it nothing it could act on.

    Accepting an abbreviation is only safe because of what insert_after is:
    every existing line survives whatever this resolves to, so the worst case
    is a new line in the wrong place, not lost code. Even then it is bounded --
    the identifiers must ALL appear on one line, and that line must be the only
    one in the file where they do. Two candidates is not a near-miss, it is a
    guess, and it gets named back instead.
    """
    wanted = _IDENTIFIER.findall(" ".join(after_lines))
    missed = (
        "the AFTER text does not appear in the file. It must match the file's "
        "CURRENT contents -- copy the line as it is now."
    )
    if len(after_lines) != 1 or not wanted:
        return None, missed

    # Whole words, not substrings: an anchor of `n` must not match every line
    # holding `name`, and `count` must not match `total_count`.
    patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in wanted]
    hits = [ln for ln in old_lines
            if ln.strip() and all(p.search(ln) for p in patterns)]
    if not hits:
        return None, missed
    if len(set(hits)) > 1:
        listed = "\n".join(f"  {ln.strip()}" for ln in dict.fromkeys(hits))
        return None, (
            f"'{after_lines[0].strip()}' matches {len(set(hits))} lines, so the "
            f"insertion point is ambiguous:\n{listed}\n"
            "Put ONE of them in AFTER, copied exactly."
        )

    return compute_insert(old, hits[0], content)


@_safe
def insert_after(args: dict, root: str) -> str:
    """Insert lines below an anchor. Cannot remove anything that is there."""
    path = check_target(root, args["path"])
    if not os.path.exists(path):
        return (f"ERROR: {args['path']} does not exist. "
                "Use apply_patch to create a new file.")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        old = fh.read()

    new, error = compute_insert(old, args.get("after", ""), args.get("content", ""))
    if new is None:
        return f"ERROR: {error}"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    rel = os.path.relpath(path, os.path.abspath(root))
    return f"INSERTED into {rel} ({len(new) - len(old)} chars added)"


def preview_insert(args: dict, root: str) -> tuple[str, str, str]:
    """Compute an insertion WITHOUT writing: returns (old, new, error)."""
    old = current_content(args, root)
    new, error = compute_insert(old, args.get("after", ""), args.get("content", ""))
    return old, (new or ""), error


def _trailing_class_indent(old: str) -> str | None:
    """
    The body indentation of a class the file ENDS inside, or None.

    `ast.parse().body[-1]` being a ClassDef is the whole test: if anything
    followed the class at module level it would be last instead. So a match
    means an append lands inside that class's body -- if it is indented to.
    """
    try:
        tree = ast.parse(old)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None
    last = tree.body[-1]
    if not isinstance(last, ast.ClassDef) or not last.body:
        return None

    lines = old.split("\n")
    first = last.body[0]
    if first.lineno > len(lines):
        return None
    line = lines[first.lineno - 1]
    return line[:len(line) - len(line.lstrip())] or None


def _is_method(addition: str) -> bool:
    """
    Does this code only make sense inside a class?

    `self` as the first parameter is the signal, and it is a reliable one --
    the model writes it precisely when it means a method. Everything in the
    block has to qualify: a mix of methods and plain functions is not a thing
    that can be indented into a class wholesale.
    """
    try:
        tree = ast.parse(textwrap.dedent(addition))
    except (SyntaxError, ValueError):
        return False
    if not tree.body:
        return False
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        params = node.args.posonlyargs + node.args.args
        if not params or params[0].arg not in ("self", "cls"):
            return False
    return True


def compute_append(old: str, addition: str) -> str:
    """
    Old content plus new content at the end, with the blank lines PEP 8 wants.

    Adding a whole new function to an existing file is the one common task that
    search/replace handles badly. To append with edit_file the model has to
    FIND the last lines of the file and repeat them in REPLACE before its new
    code -- and observed live, phi4-mini instead puts the OLD function in FIND
    and the NEW one in REPLACE, which deletes the old one. It did that three
    times running, each time blocked by the gate, never recovering.

    Appending is not a matching problem, so it should not be expressed as one.
    Note what this makes impossible: the old content is a strict prefix of the
    new, so this tool cannot delete or alter a single existing byte, whatever
    the model puts in CONTENT.
    """
    if not old:
        return addition if addition.endswith("\n") else addition + "\n"
    body = old if old.endswith("\n") else old + "\n"
    blank = len(body) - len(body.rstrip("\n")) - 1
    tail = addition.lstrip("\n")

    # "add a count method to cart.py" -- observed live. The model wrote
    # `def count(self):` and the file ended inside `class Cart`, so appending
    # at column zero produced a module-level function taking `self`: it
    # compiles, it keeps every existing line, and it is not a method. Both the
    # structural veto and the 1.5B critic passed it, the critic reporting in as
    # many words that a method had been added to the class.
    #
    # Nothing about the model's intent was unclear -- `self` says method -- so
    # this is a placement bug, and placement is decidable here without asking
    # anyone. PEP 8 wants one blank line between methods rather than two.
    indent = _trailing_class_indent(body)
    if indent and _is_method(tail):
        tail = textwrap.indent(textwrap.dedent(tail), indent)
        separator = "\n" * max(0, 1 - blank)
    else:
        separator = "\n" * max(0, 2 - blank)

    return body + separator + (tail if tail.endswith("\n") else tail + "\n")


@_safe
def append_file(args: dict, root: str) -> str:
    """Add code to the end of an existing file. Cannot touch what is there."""
    path = check_target(root, args["path"])
    if not os.path.exists(path):
        return (f"ERROR: {args['path']} does not exist. "
                "Use apply_patch to create a new file.")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        old = fh.read()

    new = compute_append(old, args.get("content", ""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    rel = os.path.relpath(path, os.path.abspath(root))
    return f"APPENDED to {rel} ({len(new) - len(old)} chars added)"


def preview_append(args: dict, root: str) -> tuple[str, str]:
    """Compute an append WITHOUT writing: returns (old, new)."""
    old = current_content(args, root)
    return old, compute_append(old, args.get("content", ""))


def preview_edit(args: dict, root: str) -> tuple[str, str, str]:
    """
    Compute an edit WITHOUT writing: returns (old, new, error).
    Used by the gate, which has to see the result before it is allowed to land.
    """
    old = current_content(args, root)
    new, error = compute_edit(old, args.get("find", ""), args.get("replace", ""))
    return old, (new or ""), error


@_safe
def run_command(args: dict, root: str) -> str:
    """Restricted shell execution: user's own repo, short timeout, no shell=True."""
    cmd = args["command"]          # KeyError here is handled by @_safe
    try:
        result = subprocess.run(
            cmd.split(),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return f"exit={result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
    except Exception as e:
        return f"ERROR running command: {e}"


DISPATCH = {
    "list_dir": list_dir,
    "read_file": read_file,
    "search_code": search_code,
    "edit_file": edit_file,
    "insert_after": insert_after,
    "append_file": append_file,
    "apply_patch": apply_patch,
    "run_command": run_command,
}
