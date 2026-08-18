"""
Recovering from the two mistakes phi4-mini actually makes on an edit.

Both were captured from a live turn. Asked to add a docstring to two_sum, the
model produced this, verbatim:

    FIND:
    ```
    def two_sum(nums, target):
        ...
            seen[num] = i
    }
    REPLACE:
    ```python
    # Given a list of numbers ...
    def two_sum(nums, target):
        ...
            seen[num] = i
    }
    ```

Two independent slips in one turn:

  1. FIND's closing fence is missing, so the search for ``` ran on and closed
     on the fence that OPENS the replacement. FIND came out ending in the
     literal text "REPLACE:", and the replacement was swallowed whole.
  2. The Python block is closed with a `}`. One character, invisible in a diff
     summary, and it makes the FIND text unmatchable forever.

Neither is recoverable by the model from the error message it gets back --
"the FIND text does not appear in the file" does not say which character is
wrong -- so it repeated the identical turn until the repeat detector killed
the run. Both are trivially recoverable HERE.
"""

from arthur import agent, tools


LIVE_TURN = """THOUGHT: I will add documentation.

CONFIDENCE: 1 (high confidence)

ACTION: edit_file
PATH: twosum.py
FIND:
```
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        seen[num] = i
}
REPLACE:
```python
def two_sum(nums, target):
    \"\"\"Return the indices of the two numbers adding to target.\"\"\"
    seen = {}
    for i, num in enumerate(nums):
        seen[num] = i
}
```
"""

FILE = ("def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, num in enumerate(nums):\n"
        "        seen[num] = i\n")


# --- the unclosed fence ------------------------------------------------------

def test_an_unclosed_find_fence_does_not_swallow_the_replacement():
    parsed = agent.parse_response(LIVE_TURN)
    args = parsed["args"]

    assert parsed["action"] == "edit_file"
    assert "REPLACE:" not in args["find"]        # the bug, in one assertion
    assert args["replace"].strip()               # and the replacement survived
    assert "Return the indices" in args["replace"]


def test_a_properly_closed_fence_is_unaffected():
    turn = ("ACTION: edit_file\nPATH: m.py\n"
            "FIND:\n```\nold line\n```\nREPLACE:\n```\nnew line\n```")
    args = agent.parse_response(turn)["args"]
    assert args["find"] == "old line"
    assert args["replace"] == "new line"


def test_code_containing_a_keyword_like_line_still_parses():
    """
    The boundary must not be so eager that real source trips it. A line
    beginning "PATH:" or "THOUGHT:" inside a fenced block is legitimate -- this
    project's own prompt strings contain exactly that -- so only a block
    keyword ALONE on its line ends a block early.
    """
    turn = ("ACTION: apply_patch\nPATH: p.py\nCONTENT:\n```\n"
            'PROMPT = "PATH: give a filename"\n'
            'HINT = "THOUGHT: reason first"\n'
            "```")
    args = agent.parse_response(turn)["args"]
    assert "PATH: give a filename" in args["new_content"]
    assert "THOUGHT: reason first" in args["new_content"]


# --- the stray brace ---------------------------------------------------------

def test_a_trailing_brace_that_closes_nothing_is_dropped():
    assert tools.strip_stray_closers("def f():\n    return 1\n}") == "def f():\n    return 1"


def test_a_brace_that_closes_something_is_kept():
    """The last line of a multi-line dict literal is real code, not a slip."""
    literal = 'config = {\n    "a": 1,\n}'
    assert tools.strip_stray_closers(literal) == literal


def test_several_stray_closers_are_dropped():
    assert tools.strip_stray_closers("x = 1\n}\n)") == "x = 1"


def test_ordinary_code_is_untouched():
    for source in ("def f():\n    return 1", "x = [1, 2]\ny = (3, 4)", ""):
        assert tools.strip_stray_closers(source) == source


def test_the_edit_lands_despite_the_stray_brace():
    find = "def two_sum(nums, target):\n    seen = {}\n"
    new, error = tools.compute_edit(
        FILE,
        find + "}",                      # the model's habit
        find + '    """Docstring."""\n',
    )
    assert error == ""
    assert '"""Docstring."""' in new
    assert "}" not in new.replace("seen = {}", "")     # no brace spliced in
    compile(new, "<test>", "exec")                     # and it is valid Python


def test_a_stray_brace_in_replace_is_dropped_too():
    """
    Repairing only the search half would splice the brace into the file and
    guarantee a syntax error -- a worse outcome than not matching at all.
    """
    new, error = tools.compute_edit(
        FILE,
        "        seen[num] = i\n}",
        "        seen[num] = i\n        continue\n}",
    )
    assert error == ""
    compile(new, "<test>", "exec")


def test_a_deliberate_closer_in_replace_is_kept():
    """When FIND matched as written, REPLACE is taken at its word."""
    old = 'config = {\n    "a": 1,\n}\n'
    new, error = tools.compute_edit(old, '    "a": 1,\n', '    "a": 1,\n    "b": 2,\n')
    assert error == ""
    assert new.count("}") == 1
    compile(new, "<test>", "exec")


def test_exact_matches_still_win_over_the_repair():
    """A file that really does end in a brace-only line must match as written."""
    old = 'x = {\n    "a": 1,\n}\ny = 2\n'
    new, error = tools.compute_edit(old, '    "a": 1,\n}', '    "a": 2,\n}')
    assert error == ""
    assert '"a": 2' in new
    compile(new, "<test>", "exec")


# --- end to end --------------------------------------------------------------

def test_the_whole_live_turn_now_applies_cleanly():
    parsed = agent.parse_response(LIVE_TURN)
    args = parsed["args"]
    new, error = tools.compute_edit(FILE, args["find"], args["replace"])

    assert error == "", error
    assert "Return the indices" in new
    assert "seen[num] = i" in new          # the body survived
    compile(new, "<test>", "exec")
