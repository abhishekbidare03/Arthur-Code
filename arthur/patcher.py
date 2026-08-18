"""
Turns a (old_content, new_content) pair into a human-readable unified diff.

Design choice: the coder agent proposes a FULL new file, not a hand-written
diff. Small models are bad at emitting syntactically valid unified diffs;
they're much better at "here is the whole corrected file." We recompute the
diff ourselves for display/review, which is more reliable than trusting the
model to format a patch correctly.
"""

import ast
import difflib
from dataclasses import dataclass, field


def unified_diff(old_content: str, new_content: str, path: str) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )
    text = "".join(diff)
    return text if text else "(no textual changes)"


# --- structural analysis -----------------------------------------------------
#
# Asking a small model for a whole rewritten file (rather than a diff it cannot
# reliably format) buys robustness at a price: when the model's attention runs
# out, it just stops writing, and the "patch" silently deletes the tail of the
# file. Observed on the first live eval run -- qwen2.5-coder:3b was asked to add
# one docstring and returned a file with two functions missing. The LLM critic
# approved it, because the diff genuinely does contain the requested docstring.
#
# So the gate needs one signal that isn't a language model's opinion. These are
# deterministic facts about the change, and they are what catch the failure an
# LLM reviewer talks itself past.


@dataclass
class PatchShape:
    removed_symbols: list[str] = field(default_factory=list)
    added_symbols: list[str] = field(default_factory=list)
    gutted_symbols: list[str] = field(default_factory=list)   # kept the name, lost the body
    hollow_additions: list[str] = field(default_factory=list)  # arrived with no body
    old_lines: int = 0
    new_lines: int = 0
    parse_broken: bool = False       # was valid Python, no longer is

    @property
    def shrink_ratio(self) -> float:
        """How much of the file disappeared. 0.0 = nothing, 1.0 = everything."""
        if not self.old_lines:
            return 0.0
        return max(0.0, (self.old_lines - self.new_lines) / self.old_lines)

    @property
    def destructive(self) -> bool:
        return bool(self.removed_symbols) or bool(self.gutted_symbols) or self.parse_broken

    @property
    def unfinished(self) -> bool:
        """Not destructive -- nothing was lost -- but not a solution either."""
        return bool(self.hollow_additions)


_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_hollow(node) -> bool:
    """
    True if a function has a name and a docstring but no actual code.

    Every other check in this module asks whether a symbol still EXISTS. This
    one asks whether it still DOES anything, which is a different question and
    the one that caught us live: asked to add a docstring to two_sum, the model
    replaced the whole function with a signature and a docstring. The name was
    still there, the file compiled, and the file got LONGER -- so the missing
    symbol check, the compile check and the shrink check all passed a function
    that now returns None.
    """
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                       # a docstring is not an implementation
    if not body:
        return True
    return all(
        isinstance(stmt, ast.Pass)
        or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis)
        for stmt in body
    )


def _top_level_symbols(source: str) -> tuple[dict[str, bool], bool]:
    """
    ({name: is_hollow}, valid_ok). Nested defs are ignored -- top level is the
    contract. Classes map to False; hollowness is only meaningful for functions.

    Uses compile(), not ast.parse(). ast.parse only builds a tree; a whole
    class of errors is raised later, when the tree is compiled to bytecode.
    `return` outside a function is the one that caught us: a model produced a
    file ending in a module-level `return None`, ast.parse accepted it happily,
    the structural check reported "valid Python", and a file that cannot even
    be imported was written to disk. compile() rejects it.
    """
    try:
        tree = ast.parse(source)
        compile(source, "<patch>", "exec")
    except (SyntaxError, ValueError):
        # ValueError covers source containing null bytes, which compile()
        # rejects separately from SyntaxError.
        return {}, False

    symbols: dict[str, bool] = {}
    for node in tree.body:
        if isinstance(node, _FUNC):
            symbols[node.name] = _is_hollow(node)
        elif isinstance(node, ast.ClassDef):
            symbols[node.name] = False
            # Methods count: losing one is losing API surface just the same.
            for sub in node.body:
                if isinstance(sub, _FUNC):
                    symbols[f"{node.name}.{sub.name}"] = _is_hollow(sub)
    return symbols, True


def analyze(old_content: str, new_content: str, path: str) -> PatchShape:
    """
    Compare before and after. Python files get symbol-level analysis; anything
    else falls back to line counts, which still catches wholesale truncation.
    """
    shape = PatchShape(
        old_lines=len(old_content.splitlines()),
        new_lines=len(new_content.splitlines()),
    )

    if not path.endswith(".py"):
        return shape

    old_symbols, old_ok = _top_level_symbols(old_content)
    new_symbols, new_ok = _top_level_symbols(new_content)

    # A function that ARRIVES with no body. This is the truncation failure seen
    # from the other side: asked to write two_sum, phi4-mini produced a
    # signature and a twelve-line docstring and stopped -- no algorithm at all.
    # The file was valid Python, deleted nothing, and grew, so every check
    # above passed a function that returns None. Nothing was lost here, which
    # is why it is tracked apart from the destructive signals; it just isn't
    # finished.
    if new_ok:
        shape.hollow_additions = sorted(
            name for name, hollow in new_symbols.items()
            if hollow and name not in old_symbols and "." not in name
        )

    # Only meaningful if the original parsed: we can't claim a symbol was
    # removed from a file we never understood in the first place.
    if old_ok:
        shape.parse_broken = not new_ok
        if new_ok:
            shape.removed_symbols = sorted(set(old_symbols) - set(new_symbols))
            shape.added_symbols = sorted(set(new_symbols) - set(old_symbols))
            # Survivors that stopped doing anything. Going the other way --
            # a stub gaining an implementation -- is exactly what we want, so
            # only the substantive -> hollow direction counts.
            shape.gutted_symbols = sorted(
                name for name in set(old_symbols) & set(new_symbols)
                if new_symbols[name] and not old_symbols[name]
            )

    return shape
