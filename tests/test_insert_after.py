"""
Inserting lines below an existing one.

"Add a docstring to X" is the most common small task there is, and it was the
last one still failing. The reason is a mismatch of mental models, not of
capability: the model thinks *insert this text after that line*, and edit_file
can only express *replace these lines with those lines*. Bridging the two means
repeating the anchor inside REPLACE, and models will not reliably do it. Told
in as many words that REPLACE had to contain the `def` line, phi4-mini went on
sending REPLACE blocks holding a docstring and nothing else -- failing a
different way each run, and once "fixing" the complaint by dropping the
docstring instead and reproducing the file unchanged.

append_file settled the same argument for "add a new function" the moment it
existed. This is the general version, and it shares the property that matters:
every existing line is kept, so no argument it can be given removes a single
character of what is already there.
"""

import pytest

from arthur import agent, tools
from arthur.llm_backend import MockBackend


CLASS_FILE = '''class Inventory:
    def __init__(self):
        self.items = {}

    def total_count(self):
        return sum(self.items.values())
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "inventory.py").write_text(CLASS_FILE, encoding="utf-8")
    return str(tmp_path)


def scripted(monkeypatch, coder, critic="VERDICT: APPROVE\nREASON: fine."):
    from arthur import llm_backend, safety_gate
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(coder))
    for module in (llm_backend, safety_gate):
        monkeypatch.setattr(module, "get_critic_backend",
                            lambda name=None: MockBackend([critic]))


def collect(gen, answer=None):
    out, reply = [], None
    while True:
        try:
            event = gen.send(reply)
        except StopIteration:
            return out
        reply = None
        if event.kind == "approval_needed":
            reply = answer
        out.append(event)


def inserted(old, after, content):
    new, error = tools.compute_insert(old, after, content)
    assert error == "", error
    return new


# --- what it cannot do -------------------------------------------------------

def test_every_existing_line_survives_whatever_it_is_given():
    """The safety property. Nothing else about this tool needs checking."""
    for content in ("x = 1", "}}}nonsense", "def f():\n    pass"):
        new = inserted(CLASS_FILE, "    def total_count(self):", content)
        for line in CLASS_FILE.split("\n"):
            assert line in new.split("\n"), (content, line)


def test_no_symbol_can_be_lost():
    from arthur import patcher
    new = inserted(CLASS_FILE, "    def total_count(self):", '"""Total."""')
    shape = patcher.analyze(CLASS_FILE, new, "inventory.py")
    assert shape.removed_symbols == []
    assert shape.gutted_symbols == []
    assert not shape.destructive


# --- the depth it inserts at -------------------------------------------------

def test_a_docstring_lands_inside_the_method_it_names():
    """
    Taken from the line BELOW the anchor, which needs no knowledge of Python:
    text going in under `def f():` lands at the depth of that function's body.
    """
    import ast
    new = inserted(CLASS_FILE, "def total_count(self):",
                   '"""Return the total number of items."""')
    compile(new, "<test>", "exec")

    method = next(n for n in ast.walk(ast.parse(new))
                  if isinstance(n, ast.FunctionDef) and n.name == "total_count")
    assert ast.get_docstring(method) == "Return the total number of items."


def test_the_method_still_works_afterwards():
    new = inserted(CLASS_FILE, "def total_count(self):", '"""Total."""')
    ns = {}
    exec(compile(new, "<test>", "exec"), ns)
    inv = ns["Inventory"]()
    inv.items = {"a": 2, "b": 5}
    assert inv.total_count() == 7


def test_inserting_beside_an_ordinary_statement():
    new = inserted(CLASS_FILE, "self.items = {}", "self.count = 0")
    compile(new, "<test>", "exec")
    assert "        self.count = 0" in new


def test_the_anchor_may_be_written_without_its_indentation():
    """How the model writes it, every time."""
    a = inserted(CLASS_FILE, "def total_count(self):", '"""Total."""')
    b = inserted(CLASS_FILE, "    def total_count(self):", '"""Total."""')
    assert a == b


def test_content_indentation_is_normalized_not_trusted():
    """Whatever the model indents by, the result lands at the right depth."""
    results = {inserted(CLASS_FILE, "def total_count(self):", c)
               for c in ('"""Total."""', '    """Total."""', '        """Total."""')}
    assert len(results) == 1
    compile(results.pop(), "<test>", "exec")


def test_inserting_after_the_last_statement_of_a_block_stays_in_the_block():
    """
    The line below dedents out of the block, so it cannot be the guide here --
    the anchor's own depth is. The same rule covers both cases by taking the
    deeper of the two.
    """
    new = inserted(CLASS_FILE, "        self.items = {}", "self.count = 0")
    compile(new, "<test>", "exec")
    assert "        self.count = 0" in new

    ns = {}
    exec(compile(new, "<test>", "exec"), ns)
    assert ns["Inventory"]().count == 0          # it really is in __init__


def test_inserting_after_the_last_line_of_a_file():
    new = inserted("x = 1\n", "x = 1", "y = 2")
    assert new == "x = 1\ny = 2\n"


def test_the_trailing_newline_is_preserved():
    assert inserted("x = 1\n", "x = 1", "y = 2").endswith("\n")
    assert not inserted("x = 1", "x = 1", "y = 2").endswith("\n")


# --- refusals ----------------------------------------------------------------

def test_a_missing_anchor_is_refused():
    new, error = tools.compute_insert(CLASS_FILE, "def nonexistent(self):", "x = 1")
    assert new is None
    assert "does not appear" in error


def test_an_ambiguous_anchor_is_refused_with_a_count():
    old = "def a():\n    return 1\n\n\ndef b():\n    return 1\n"
    new, error = tools.compute_insert(old, "return 1", "print('hi')")
    assert new is None
    assert "appears 2 times" in error


def test_an_empty_anchor_is_refused():
    new, error = tools.compute_insert(CLASS_FILE, "", "x = 1")
    assert new is None
    assert "AFTER block is empty" in error


def test_a_single_anchor_line_with_nothing_to_insert_is_refused():
    new, error = tools.compute_insert(CLASS_FILE, "    def total_count(self):", "")
    assert new is None
    assert "nothing to insert" in error


# --- the two blocks written as one -------------------------------------------

def test_a_merged_block_is_split_at_the_line_that_still_matches():
    """
    Observed live, five turns running. The model writes the anchor and the new
    lines together under AFTER, because that is what the finished code looks
    like, and omits CONTENT entirely. Answering "missing: content" achieved
    nothing -- it sent the identical turn until the repeat detector fired.

    The split is not a guess: the longest run of leading lines that is actually
    IN the file is the anchor, and what follows cannot be, or it would have
    matched too.
    """
    import ast
    merged = ('def total_count(self):\n'
              '    """Return the sum of all item quantities."""')
    new = inserted(CLASS_FILE, merged, "")
    compile(new, "<test>", "exec")

    method = next(n for n in ast.walk(ast.parse(new))
                  if isinstance(n, ast.FunctionDef) and n.name == "total_count")
    assert ast.get_docstring(method) == "Return the sum of all item quantities."
    assert "return sum(self.items.values())" in new


def test_a_merged_block_splits_after_several_matching_lines():
    merged = ("    def __init__(self):\n"
              "        self.items = {}\n"
              "        self.count = 0")
    new = inserted(CLASS_FILE, merged, "")
    compile(new, "<test>", "exec")
    assert "        self.count = 0" in new
    assert new.count("self.items = {}") == 1        # not duplicated


def test_splitting_never_duplicates_the_anchor():
    merged = 'def total_count(self):\n    """Total."""'
    new = inserted(CLASS_FILE, merged, "")
    assert new.count("def total_count(self):") == 1


def test_an_explicit_content_block_is_never_second_guessed():
    """The split only runs when CONTENT is genuinely absent."""
    new = inserted(CLASS_FILE, "def total_count(self):", '"""Total."""')
    assert new.count("def total_count(self):") == 1
    assert '"""Total."""' in new


# --- through the agent -------------------------------------------------------

INSERT_TURN = (
    'THOUGHT: adding the docstring.\n'
    'CONFIDENCE: 0.9\n'
    'ACTION: insert_after\n'
    'PATH: inventory.py\n'
    'AFTER:\n```\n'
    'def total_count(self):\n'
    '```\n'
    'CONTENT:\n```\n'
    '"""Return the total number of items."""\n'
    '```'
)


def test_the_insert_form_parses():
    parsed = agent.parse_response(INSERT_TURN)
    assert parsed["action"] == "insert_after"
    assert parsed["args"]["path"] == "inventory.py"
    assert parsed["args"]["after"] == "def total_count(self):"
    assert '"""Return the total number of items."""' in parsed["args"]["content"]


def test_insert_lands_through_the_gate(repo, monkeypatch):
    scripted(monkeypatch, [INSERT_TURN, "FINAL: added the docstring."])
    stream = collect(agent.run("add a docstring to total_count", repo,
                               backend="mock", auto_approve=True))

    assert "gate_decision" in [e.kind for e in stream]
    written = open(f"{repo}/inventory.py").read()
    compile(written, "<test>", "exec")
    assert "Return the total number of items" in written
    assert "return sum(self.items.values())" in written    # body intact


def test_inserting_into_a_missing_file_is_refused(repo, monkeypatch):
    turn = INSERT_TURN.replace("PATH: inventory.py", "PATH: nope.py")
    scripted(monkeypatch, [turn, "FINAL: gave up."])
    stream = collect(agent.run("add a docstring", repo, backend="mock",
                               auto_approve=True))

    observation = next(e for e in stream if e.kind == "observation")
    assert "does not exist" in observation.text
    assert "apply_patch" in observation.text


def test_a_dropped_anchor_is_handed_the_insert_form_ready_to_send(repo, monkeypatch):
    """
    The recovery that matters. Rather than explaining the difference between
    replacing and inserting for a third time, give back the model's own two
    blocks arranged as the tool that means what it meant.
    """
    dropped = (
        'ACTION: edit_file\nPATH: inventory.py\n'
        'FIND:\n```\ndef total_count(self):\n```\n'
        'REPLACE:\n```\n"""Return the total."""\n```'
    )
    scripted(monkeypatch, [dropped, "FINAL: stopped."])
    stream = collect(agent.run("add a docstring", repo, backend="mock",
                               auto_approve=True), answer=None)

    observation = next(e for e in stream if e.kind == "observation")
    assert "ACTION: insert_after" in observation.text
    assert "AFTER:" in observation.text
    assert "def total_count(self):" in observation.text
    assert '"""Return the total."""' in observation.text


def test_insert_after_is_the_headline_tool_in_the_prompt():
    prompt = agent.build_system_prompt("/repo")
    assert "insert_after" in prompt
    assert "add a docstring" in prompt.lower()


# --- an anchor that names its target instead of quoting it -------------------

def test_a_method_named_without_its_def_line_still_resolves():
    """
    Captured live. Asked to document `total_count`, phi4-mini sent
    `AFTER: total_count:` -- the name and a colon, where the file says
    `    def total_count(self):`. An abbreviation, not a matching problem, and
    "the AFTER text does not appear in the file" told it nothing it could act
    on: it re-sent the same turn until the stall detector fired.
    """
    import ast
    new = inserted(CLASS_FILE, "total_count:", '"""Return the total."""')
    compile(new, "<test>", "exec")

    method = next(n for n in ast.walk(ast.parse(new))
                  if isinstance(n, ast.FunctionDef) and n.name == "total_count")
    assert ast.get_docstring(method) == "Return the total."
    assert "return sum(self.items.values())" in new


def test_a_bare_name_works_too():
    new = inserted(CLASS_FILE, "total_count", "x = 1")
    assert "        x = 1" in new


def test_nothing_existing_is_lost_by_an_abbreviated_anchor():
    """The safety property has to survive the convenience."""
    new = inserted(CLASS_FILE, "total_count:", '"""Total."""')
    for line in CLASS_FILE.rstrip("\n").split("\n"):
        assert line in new.split("\n"), line


def test_an_ambiguous_abbreviation_is_refused_with_the_candidates():
    """
    Two candidates is a guess, not a near-miss. The refusal names both, so the
    next turn is a choice rather than another blind attempt.
    """
    old = ("def load(self, name):\n    return name\n\n\n"
           "def reload(self, name):\n    return name\n")
    new, error = tools.compute_insert(old, "name", "x = 1")
    assert new is None
    assert "ambiguous" in error
    assert "def load(self, name):" in error
    assert "def reload(self, name):" in error


def test_a_name_that_is_nowhere_gets_the_plain_refusal():
    new, error = tools.compute_insert(CLASS_FILE, "nonexistent_thing", "x = 1")
    assert new is None
    assert "does not appear" in error


def test_a_multi_line_anchor_is_never_abbreviated():
    """
    Several lines that do not match is a copying mistake, and guessing at it
    would be resolving a whole block from scattered identifiers.
    """
    new, error = tools.compute_insert(
        CLASS_FILE, "def total_count(self):\n    return WRONG", "x = 1")
    assert new is None
    assert "does not appear" in error


def test_an_exact_anchor_never_reaches_the_fallback():
    """The abbreviation path must not change what already worked."""
    a = inserted(CLASS_FILE, "    def total_count(self):", '"""Total."""')
    b = inserted(CLASS_FILE, "total_count:", '"""Total."""')
    assert a == b


def test_the_anchor_matches_whole_words_not_substrings():
    """
    `count` must not resolve against `total_count`, and `n` must not match
    every line containing `name`. Substring containment would make the
    fallback fire far more often than it should, and each extra firing is a
    line inserted somewhere nobody asked for.
    """
    old = ("class Basket:\n"
           "    def total_count(self):\n"
           "        return 0\n\n"
           "    def count(self):\n"
           "        return 1\n")
    new = inserted(old, "count", "x = 1")
    lines = new.split("\n")
    at = lines.index("        x = 1")
    assert lines[at - 1] == "    def count(self):"


def test_the_prompt_says_the_anchor_is_a_whole_line():
    """
    The abbreviated-anchor fallback recovers `AFTER: total_count:`, but a
    recovery is a second-best outcome -- it resolves by heuristic where an
    exact line resolves by construction. Say it in the prompt too.
    """
    prompt = agent.build_system_prompt("/repo")
    assert "WHOLE line" in prompt
    assert "total_count:" in prompt          # the wrong version, shown as wrong


def test_a_missing_content_block_is_handed_back_as_a_form():
    """
    Observed live, and the commonest remaining docstring failure: AFTER parses
    correctly and CONTENT arrives empty. There is no merged block to split and
    no way to invent the missing text, so the only useful reply is the model's
    own turn with the gap marked -- the move that fixed the dropped anchor.
    Restating the rule does not work; it re-sent the identical turn until the
    stall detector fired.
    """
    new, error = tools.compute_insert(CLASS_FILE, "    def total_count(self):", "")
    assert new is None
    assert "ACTION: insert_after" in error
    assert "def total_count(self):" in error      # its own anchor, ready to send
    assert "CONTENT:" in error
    assert "was missing" in error


def test_the_handed_back_form_parses_as_a_real_call():
    """A template the agent would reject teaches the model a broken shape."""
    _, error = tools.compute_insert(CLASS_FILE, "    def total_count(self):", "")
    form = error[error.index("ACTION: insert_after"):].replace("<the file>",
                                                              "inventory.py")
    parsed = agent.parse_response(form)
    assert parsed["action"] == "insert_after"
    # The anchor comes back exactly as the model sent it, indentation included,
    # so re-sending the form is a copy rather than a retype.
    assert parsed["args"]["after"] == "    def total_count(self):"


def test_a_merged_block_is_still_split_rather_than_refused():
    """The form is the last resort, not the first answer."""
    merged = 'def total_count(self):\n    """Total."""'
    new, error = tools.compute_insert(CLASS_FILE, merged, "")
    assert error == ""
    assert new is not None
