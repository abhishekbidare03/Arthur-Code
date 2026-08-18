"""
The deterministic half of the safety gate, and repo containment.

Both of these exist because of things that actually happened on the first live
run against qwen2.5-coder:3b, not because they seemed like good ideas:

  - asked to add one docstring, the model returned a "patched" file with two
    top-level functions silently missing, and the LLM critic approved it;
  - the model echoed back an absolute path, which os.path.join honours by
    discarding the repo root entirely.
"""

import os

import pytest

from arthur import config, patcher, safety_gate, tools


ORIGINAL = '''"""A small module."""


def alpha(x):
    return x + 1


def beta(x):
    return x * 2


class Gamma:
    def method(self):
        return 3
'''


# --- structural analysis -----------------------------------------------------

def test_pure_addition_is_not_destructive():
    new = ORIGINAL + "\n\ndef delta(x):\n    return x - 1\n"
    shape = patcher.analyze(ORIGINAL, new, "m.py")
    assert shape.removed_symbols == []
    assert shape.added_symbols == ["delta"]
    assert not shape.destructive


def test_docstring_edit_is_not_destructive():
    new = ORIGINAL.replace("return x + 1", '"""Add one."""\n    return x + 1')
    assert not patcher.analyze(ORIGINAL, new, "m.py").destructive


def test_truncation_is_caught():
    """The exact failure observed live: the tail of the file just stops."""
    truncated = '"""A small module."""\n\n\ndef alpha(x):\n    return x + 1\n'
    shape = patcher.analyze(ORIGINAL, truncated, "m.py")
    assert "beta" in shape.removed_symbols
    assert "Gamma" in shape.removed_symbols
    assert shape.destructive


def test_removing_a_method_counts_as_removing_a_symbol():
    new = ORIGINAL.replace("    def method(self):\n        return 3\n", "    pass\n")
    assert "Gamma.method" in patcher.analyze(ORIGINAL, new, "m.py").removed_symbols


def test_renaming_reads_as_remove_plus_add():
    new = ORIGINAL.replace("def beta(", "def beta_renamed(")
    shape = patcher.analyze(ORIGINAL, new, "m.py")
    assert shape.removed_symbols == ["beta"]
    assert shape.added_symbols == ["beta_renamed"]


def test_broken_syntax_is_caught():
    shape = patcher.analyze(ORIGINAL, "def alpha(x\n    return", "m.py")
    assert shape.parse_broken
    assert shape.destructive


def test_return_outside_a_function_is_caught():
    """
    ast.parse accepts this; only compile() rejects it. Observed live: a model
    wrote a file ending in a module-level `return None`, the check reported
    "valid Python", and a file that cannot be imported was written to disk.
    """
    broken = "def alpha(x):\n    return x\n\nreturn None\n"
    import ast
    ast.parse(broken)                                   # the lenient check passes
    assert patcher.analyze(ORIGINAL, broken, "m.py").parse_broken


def test_yield_outside_a_function_is_caught():
    broken = "def alpha(x):\n    return x\n\nyield 1\n"
    assert patcher.analyze(ORIGINAL, broken, "m.py").parse_broken


def test_compile_level_break_blocks_the_gate():
    broken = ORIGINAL + "\nreturn None\n"
    result = evaluate(ORIGINAL, broken)
    assert result.blocked_structurally
    assert any("valid Python" in w for w in result.structural_warnings)


def test_unparseable_original_makes_no_claims():
    """We can't say a symbol vanished from a file we never understood."""
    shape = patcher.analyze("this is not python <<<", "def a(): pass", "m.py")
    assert shape.removed_symbols == []
    assert not shape.parse_broken


def test_non_python_falls_back_to_line_counts():
    shape = patcher.analyze("a\nb\nc\nd\n", "a\n", "notes.txt")
    assert shape.removed_symbols == []
    assert shape.shrink_ratio == 0.75


def test_shrink_ratio_ignores_growth():
    assert patcher.analyze("a\n", "a\nb\nc\n", "m.txt").shrink_ratio == 0.0


# --- keeping the name, losing the body ---------------------------------------
#
# Every check above asks whether a symbol still EXISTS. These ask whether it
# still DOES anything. Observed live, and the reason they exist: asked to add a
# docstring to two_sum, phi4-mini replaced the whole function with a signature
# and a docstring. The name survived, the file compiled, and the file got
# LONGER -- so symbol-removal, compile and shrink checks all passed a function
# that now returns None.

GUTTED = '''"""A small module."""


def alpha(x):
    """Add one to x and return it."""


def beta(x):
    return x * 2


class Gamma:
    def method(self):
        return 3
'''


def test_a_function_reduced_to_a_docstring_is_caught():
    shape = patcher.analyze(ORIGINAL, GUTTED, "m.py")
    assert shape.gutted_symbols == ["alpha"]
    assert shape.removed_symbols == []          # the name is still there
    assert shape.shrink_ratio == 0.0            # and the file did not shrink
    assert shape.destructive


def test_gutting_blocks_the_gate_despite_two_approvals():
    result = evaluate(ORIGINAL, GUTTED)
    assert result.score > config.CONFIDENCE_THRESHOLD
    assert result.blocked_structurally
    assert any("alpha" in w for w in result.structural_warnings)


def test_a_body_replaced_by_pass_is_caught():
    new = ORIGINAL.replace("    return x + 1", "    pass")
    assert patcher.analyze(ORIGINAL, new, "m.py").gutted_symbols == ["alpha"]


def test_a_body_replaced_by_ellipsis_is_caught():
    new = ORIGINAL.replace("    return x + 1", "    ...")
    assert patcher.analyze(ORIGINAL, new, "m.py").gutted_symbols == ["alpha"]


def test_a_gutted_method_is_caught():
    new = ORIGINAL.replace("        return 3", '        """Return three."""')
    assert patcher.analyze(ORIGINAL, new, "m.py").gutted_symbols == ["Gamma.method"]


def test_adding_a_docstring_properly_is_not_gutting():
    """The change the model was actually asked for must still sail through."""
    new = ORIGINAL.replace("def alpha(x):\n    return x + 1",
                           'def alpha(x):\n    """Add one."""\n    return x + 1')
    shape = patcher.analyze(ORIGINAL, new, "m.py")
    assert shape.gutted_symbols == []
    assert not shape.destructive


def test_filling_in_a_stub_is_not_gutting():
    """Hollow -> substantive is the direction we want; only the reverse counts."""
    stub = "def alpha(x):\n    ...\n"
    filled = "def alpha(x):\n    return x + 1\n"
    assert patcher.analyze(stub, filled, "m.py").gutted_symbols == []


def test_a_stub_that_stays_a_stub_is_not_flagged():
    stub = "def alpha(x):\n    ...\n\n\ndef beta(x):\n    return 1\n"
    new = stub.replace("def beta(x):\n    return 1", "def beta(x):\n    return 2")
    assert patcher.analyze(stub, new, "m.py").gutted_symbols == []


# --- arriving with no body ---------------------------------------------------
#
# The same failure from the other side. Asked to WRITE two_sum from scratch,
# phi4-mini produced a signature and a twelve-line docstring and stopped -- no
# algorithm at all. Nothing was deleted (there was nothing there), the file was
# valid Python, and it only grew, so every check above passed a function that
# returns None.

TRUNCATED_NEW_FILE = '''def two_sum(nums, target):
    """
    Given a list of integers and a target, return the indices of the two
    numbers that add up to the target.

    Args:
        nums (List[int]): A list of integers.
        target (int): The target sum.

    Returns:
        Tuple[int, int]: The indices of the two numbers.
    """
'''


def test_a_new_function_with_no_body_is_caught():
    shape = patcher.analyze("", TRUNCATED_NEW_FILE, "twosum.py")
    assert shape.hollow_additions == ["two_sum"]
    assert shape.unfinished
    assert not shape.destructive          # nothing was lost; it just isn't done
    assert not shape.parse_broken         # and it is perfectly valid Python


def test_an_unfinished_new_file_blocks_the_gate():
    result = evaluate("", TRUNCATED_NEW_FILE, path="twosum.py")
    assert result.blocked_structurally
    assert any("no code in the body" in w for w in result.structural_warnings)


def test_a_complete_new_file_passes():
    complete = ('def two_sum(nums, target):\n'
                '    """Return the indices of the pair summing to target."""\n'
                '    seen = {}\n'
                '    for i, n in enumerate(nums):\n'
                '        if target - n in seen:\n'
                '            return [seen[target - n], i]\n'
                '        seen[n] = i\n')
    shape = patcher.analyze("", complete, "twosum.py")
    assert shape.hollow_additions == []
    assert not evaluate("", complete, path="twosum.py").blocked_structurally


def test_an_untouched_hollow_function_is_not_re_flagged():
    """Only NEW stubs count -- an existing one is the user's business."""
    stub = 'def planned():\n    """Not written yet."""\n'
    new = stub + "\n\ndef done():\n    return 1\n"
    assert patcher.analyze(stub, new, "m.py").hollow_additions == []


def test_a_new_abstract_method_is_not_flagged():
    """Methods are exempt: `...` in a Protocol or ABC is normal, deliberate code."""
    old = "class Base:\n    pass\n"
    new = "class Base:\n    def handle(self):\n        ...\n"
    assert patcher.analyze(old, new, "m.py").hollow_additions == []


# --- the gate's veto ---------------------------------------------------------

class ApprovingCritic:
    """Stands in for the 1.5B that waved the truncation through."""
    def chat(self, messages, on_token=None):
        from arthur.llm_backend import LLMResponse
        return LLMResponse(text="VERDICT: APPROVE\nREASON: looks good to me.")


def evaluate(old, new, path="m.py", confidence="0.99"):
    return safety_gate.evaluate(
        task="add a docstring",
        coder_response_text=f"CONFIDENCE: {confidence}",
        diff_text=patcher.unified_diff(old, new, path),
        critic_backend=ApprovingCritic(),
        old_content=old, new_content=new, path=path,
    )


def test_high_confidence_plus_approving_critic_still_blocked_on_deletion():
    """
    The headline case. Both LLM signals say yes -- self-confidence 0.99 and an
    APPROVE from the critic, scoring a clean 1.0 -- and the patch is still
    stopped, because the deletion is a fact rather than an opinion.
    """
    truncated = '"""A small module."""\n\n\ndef alpha(x):\n    return x + 1\n'
    result = evaluate(ORIGINAL, truncated)

    # Both LLM signals are maxed out: 0.5*0.99 + 0.5*1.0, way over threshold.
    assert result.score > config.CONFIDENCE_THRESHOLD
    assert result.critic_verdict == "APPROVE"
    assert result.blocked_structurally
    assert result.needs_human
    assert not result.approved
    assert any("beta" in w for w in result.structural_warnings)


def test_clean_patch_passes_the_structural_check():
    new = ORIGINAL.replace("return x + 1", '"""Add one."""\n    return x + 1')
    result = evaluate(ORIGINAL, new)
    assert not result.blocked_structurally
    assert result.structural_warnings == []
    assert result.approved


def test_new_file_is_not_treated_as_deletion():
    result = evaluate("", "def brand_new():\n    return 1\n")
    assert not result.blocked_structurally


def test_gate_without_content_still_works():
    """Callers that don't pass content just skip the structural half."""
    result = safety_gate.evaluate(
        task="t", coder_response_text="CONFIDENCE: 0.9",
        diff_text="--- a\n+++ b\n", critic_backend=ApprovingCritic(),
    )
    assert result.structural_warnings == []
    assert not result.blocked_structurally


def test_massive_shrink_blocks_even_without_symbol_loss():
    old = "\n".join(f"# comment line {i}" for i in range(100)) + "\n"
    result = evaluate(old, "# comment line 0\n", path="notes.txt")
    assert result.blocked_structurally
    assert any("deletes" in w for w in result.structural_warnings)


# --- repo containment --------------------------------------------------------

def test_relative_path_resolves_inside_the_repo(tmp_path):
    resolved = tools.resolve(str(tmp_path), "sub/file.py")
    assert resolved.startswith(os.path.abspath(str(tmp_path)))


def test_dotdot_escape_is_refused(tmp_path):
    with pytest.raises(tools.PathEscape):
        tools.resolve(str(tmp_path), "../../../etc/passwd")


def test_absolute_path_outside_repo_is_refused(tmp_path):
    outside = os.path.abspath(os.path.join(str(tmp_path), "..", "elsewhere.py"))
    with pytest.raises(tools.PathEscape):
        tools.resolve(str(tmp_path), outside)


def test_absolute_path_inside_repo_is_allowed(tmp_path):
    """What the model actually did -- echoing back a full path it had seen."""
    inside = os.path.join(str(tmp_path), "inventory.py")
    assert tools.resolve(str(tmp_path), inside) == os.path.abspath(inside)


def test_sibling_directory_with_shared_prefix_is_refused(tmp_path):
    """`/repo-backup` must not pass a naive startswith check against `/repo`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "repo-backup").mkdir()
    with pytest.raises(tools.PathEscape):
        tools.resolve(str(repo), str(tmp_path / "repo-backup" / "x.py"))


def test_escape_becomes_an_observation_not_a_crash(tmp_path):
    """The model should be told off and retry, not kill the run."""
    result = tools.read_file({"path": "../../secrets.txt"}, str(tmp_path))
    assert result.startswith("ERROR:")
    assert "outside the repository" in result


def test_patch_outside_repo_never_reaches_disk(tmp_path):
    target = tmp_path.parent / "should_not_exist.py"
    out = tools.apply_patch({"path": str(target), "new_content": "x = 1\n"}, str(tmp_path))
    assert out.startswith("ERROR:")
    assert not target.exists()


def test_written_path_is_reported_relative(tmp_path):
    out = tools.apply_patch({"path": "a/b.py", "new_content": "x = 1\n"}, str(tmp_path))
    assert "a" in out and str(tmp_path) not in out
