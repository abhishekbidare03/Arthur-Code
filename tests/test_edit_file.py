"""
Search/replace editing — the fix for the measured local-model failure.

A 3B cannot reproduce a file it did not intend to change, so every whole-file
rewrite it produced deleted something. edit_file removes that job entirely: the
model writes only the fragment it is changing, and anything it does not mention
is untouched by construction.

The matching is exact on purpose. There is no fuzzy path, so a model that
invents or paraphrases a snippet gets a clear error instead of a corrupted
file — the failure is loud, and recoverable by the model itself.
"""

import pytest

from arthur import tools
from arthur.agent import parse_edit_form, parse_response


ORIGINAL = """def add(a, b):
    return a + b


def divide(a, b):
    return a / b
"""


# --- the matching rules ------------------------------------------------------

def test_unique_match_is_replaced():
    new, err = tools.compute_edit(ORIGINAL, "    return a / b", "    return a / b if b else None")
    assert err == ""
    assert "if b else None" in new


def test_untouched_code_survives_verbatim():
    """The whole point: what the model doesn't mention cannot be lost."""
    new, _ = tools.compute_edit(ORIGINAL, "def divide(a, b):\n    return a / b",
                                "def divide(a, b):\n    if b == 0:\n        raise ValueError\n    return a / b")
    assert "def add(a, b):\n    return a + b" in new


def test_missing_text_is_refused():
    new, err = tools.compute_edit(ORIGINAL, "def multiply(a, b):", "x")
    assert new is None
    assert "does not appear" in err


def test_ambiguous_match_is_refused_with_a_count():
    """Real risk: `return a + b` style lines recur. Refuse, don't guess."""
    src = "def f():\n    return 1\n\n\ndef g():\n    return 1\n"
    new, err = tools.compute_edit(src, "    return 1", "    return 2")
    assert new is None
    assert "2 times" in err
    assert "more surrounding lines" in err


def test_empty_find_is_refused():
    """Otherwise "" matches everywhere and would prepend to the file."""
    new, err = tools.compute_edit(ORIGINAL, "", "x")
    assert new is None
    assert "empty" in err


def test_empty_replace_is_a_deletion_not_an_error():
    new, err = tools.compute_edit(ORIGINAL, "def add(a, b):\n    return a + b\n\n\n", "")
    assert err == ""
    assert "def add" not in new
    assert "def divide" in new


def test_trailing_whitespace_mismatch_is_tolerated():
    """Models get invisible characters wrong while getting the code right."""
    src = "def f():   \n    return 1\n"
    new, err = tools.compute_edit(src, "def f():\n    return 1", "def f():\n    return 2")
    assert err == ""
    assert "return 2" in new


def test_multiline_replacement_grows_the_file():
    new, _ = tools.compute_edit(ORIGINAL, "    return a / b",
                                "    if b == 0:\n        raise ValueError('no')\n    return a / b")
    assert new.count("\n") > ORIGINAL.count("\n")


# --- the tool ----------------------------------------------------------------

def test_edit_file_writes_to_disk(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(ORIGINAL, encoding="utf-8")

    out = tools.edit_file({"path": "m.py", "find": "    return a / b",
                           "replace": "    return 0"}, str(tmp_path))
    assert out.startswith("EDITED")
    assert "return 0" in f.read_text(encoding="utf-8")


def test_failed_match_leaves_the_file_alone(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(ORIGINAL, encoding="utf-8")

    out = tools.edit_file({"path": "m.py", "find": "not in the file", "replace": "x"},
                          str(tmp_path))
    assert out.startswith("ERROR:")
    assert f.read_text(encoding="utf-8") == ORIGINAL


def test_editing_a_missing_file_points_at_apply_patch(tmp_path):
    out = tools.edit_file({"path": "nope.py", "find": "x", "replace": "y"}, str(tmp_path))
    assert "does not exist" in out
    assert "apply_patch" in out


def test_edit_file_respects_repo_containment(tmp_path):
    out = tools.edit_file({"path": "../escape.py", "find": "x", "replace": "y"},
                          str(tmp_path))
    assert out.startswith("ERROR:")
    assert "outside the repository" in out


def test_preview_does_not_write(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(ORIGINAL, encoding="utf-8")

    old, new, err = tools.preview_edit(
        {"path": "m.py", "find": "    return a / b", "replace": "    return 0"}, str(tmp_path))
    assert err == ""
    assert "return 0" in new
    assert f.read_text(encoding="utf-8") == ORIGINAL      # untouched


# --- parsing -----------------------------------------------------------------

def test_parse_find_replace_form():
    raw = (
        "THOUGHT: guard it.\n"
        "CONFIDENCE: 0.9\n"
        "ACTION: edit_file\n"
        "PATH: calculator.py\n"
        "FIND:\n```python\ndef divide(a, b):\n    return a / b\n```\n"
        "REPLACE:\n```python\ndef divide(a, b):\n    if b == 0:\n"
        "        raise ValueError('no')\n    return a / b\n```"
    )
    p = parse_response(raw)
    assert p["action"] == "edit_file"
    assert p["args"]["path"] == "calculator.py"
    assert p["args"]["find"] == "def divide(a, b):\n    return a / b"
    assert "raise ValueError" in p["args"]["replace"]
    assert p["confidence"] == 0.9


def test_parse_tolerates_missing_language_tag():
    raw = ("ACTION: edit_file\nPATH: m.py\n"
           "FIND:\n```\nold line\n```\nREPLACE:\n```\nnew line\n```")
    assert parse_response(raw)["args"]["find"] == "old line"


def test_parse_empty_replace_block():
    raw = "ACTION: edit_file\nPATH: m.py\nFIND:\n```\ngone\n```\nREPLACE:\n```\n```"
    args = parse_response(raw)["args"]
    assert args["find"] == "gone"
    assert args["replace"] == ""


def test_parse_keeps_code_containing_the_word_find():
    """A FIND: inside the code must not be read as protocol."""
    raw = ("ACTION: edit_file\nPATH: m.py\n"
           "FIND:\n```\nx = 1  # FIND: something\n```\nREPLACE:\n```\nx = 2\n```")
    assert parse_edit_form(raw)["find"] == "x = 1  # FIND: something"


def test_edit_without_a_find_block_yields_nothing():
    assert parse_edit_form("ACTION: edit_file\nPATH: m.py\n") == {}


def test_json_args_still_work_for_edit_file():
    raw = 'ACTION: edit_file\nARGS: {"path": "m.py", "find": "a", "replace": "b"}'
    assert parse_response(raw)["args"] == {"path": "m.py", "find": "a", "replace": "b"}
