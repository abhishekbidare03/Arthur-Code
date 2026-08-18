"""
Splicing a REPLACE block into indented code.

The failure, live and reproducible: asked to add a docstring to `total_count`
-- a METHOD, four spaces inside a class -- phi4-mini anchored FIND on
`def total_count(self):` and wrote a REPLACE block flush against the left
margin, because that is how the code looks when you think about it alone. The
first line lands correctly, inheriting the anchor's indentation from the text
already on that line; every line after it is short by exactly four spaces, and
the method body that follows is then over-indented relative to its docstring.
IndentationError.

The model corrected its REPLACE block exactly as instructed, three turns
running, and was rejected every time, because the real problem was four spaces
it could not see. Repairing it here is the difference between a task that
passes and one that burns the step limit.

The repair is outcome-driven rather than unconditional: reindenting always
would double the indentation of a model that got it right. So the naive splice
is tried first, and the reindented one is used only when the naive one breaks
a file that used to parse.
"""

from arthur import tools


CLASS_FILE = '''class Inventory:
    def __init__(self):
        self.items = {}

    def total_count(self):
        return sum(self.items.values())
'''


def edited(old, find, replace):
    new, error = tools.compute_edit(old, find, replace)
    assert error == "", error
    return new


# --- the live failure --------------------------------------------------------

def test_a_flush_left_replace_is_reindented_into_a_method():
    new = edited(
        CLASS_FILE,
        "def total_count(self):",
        # Self-consistent on its own terms -- the docstring IS indented under
        # its def. It is only the base indent, the four spaces that put the
        # method inside the class, that the model left off. This is verbatim
        # the shape phi4-mini produced.
        'def total_count(self):\n    """Return the total number of items."""',
    )
    compile(new, "<test>", "exec")
    assert "def total_count" in new
    assert "def __init__" in new


def test_the_docstring_actually_attaches_to_the_method():
    """Valid Python isn't enough -- it has to be the docstring OF the method."""
    import ast
    new = edited(
        CLASS_FILE,
        "def total_count(self):",
        # Self-consistent on its own terms -- the docstring IS indented under
        # its def. It is only the base indent, the four spaces that put the
        # method inside the class, that the model left off. This is verbatim
        # the shape phi4-mini produced.
        'def total_count(self):\n    """Return the total number of items."""',
    )
    tree = ast.parse(new)
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "total_count")
    assert ast.get_docstring(method)
    assert "total number of items" in ast.get_docstring(method)


def test_the_method_body_survives():
    new = edited(
        CLASS_FILE,
        "def total_count(self):",
        'def total_count(self):\n    """Total items."""',
    )
    ns = {}
    exec(compile(new, "<test>", "exec"), ns)
    inv = ns["Inventory"]()
    inv.items = {"a": 2, "b": 3}
    assert inv.total_count() == 5


# --- not firing when it shouldn't --------------------------------------------

def test_correctly_indented_replace_is_left_alone():
    """The naive splice parses, so nothing is touched -- no double indent."""
    new = edited(
        CLASS_FILE,
        "    def total_count(self):\n        return sum(self.items.values())\n",
        "    def total_count(self):\n"
        '        """Total items."""\n'
        "        return sum(self.items.values())\n",
    )
    assert '        """Total items."""' in new
    assert '            """' not in new          # not pushed out again
    compile(new, "<test>", "exec")


def test_top_level_edits_are_unaffected():
    old = "def f():\n    return 1\n"
    new = edited(old, "def f():", "def f():\n    return 2")
    assert new == "def f():\n    return 2\n    return 1\n"


def test_a_single_line_replacement_is_never_reindented():
    new = edited(CLASS_FILE, "self.items = {}", "self.items = dict()")
    assert "self.items = dict()" in new
    compile(new, "<test>", "exec")


def test_indentation_is_matched_even_in_a_file_that_does_not_parse():
    """
    The alignment is textual, not semantic -- it reads the file's own layout
    rather than asking Python what the code means. So it still works on a file
    that is mid-edit and temporarily broken, which is exactly when an agent is
    most likely to be looking at one.
    """
    broken = "class Inventory:\n    def f(self:\n"
    new, error = tools.compute_edit(broken, "def f(self:", "def f(self:\nx = 1")
    assert error == ""
    assert new == "class Inventory:\n    def f(self:\n    x = 1\n"


def test_indentation_is_preserved_in_non_python_files_too():
    """Nothing here is Python-specific: it is about columns, not syntax."""
    old = "  item one\n  item two\n"
    new = edited(old, "item one", "item one\nitem one and a half")
    assert new == "  item one\n  item one and a half\n  item two\n"


def test_an_unfixable_break_stays_broken_rather_than_being_guessed_at():
    """
    Reindenting is a repair, not a rescue. When it does not produce something
    that parses, the model's own text is what goes to the gate -- which then
    blocks it and says so, instead of us inventing a third version.
    """
    new = edited(CLASS_FILE, "def total_count(self):", "def total_count(self:")
    assert "def total_count(self:" in new
    try:
        compile(new, "<test>", "exec")
        raise AssertionError("expected this to stay broken")
    except SyntaxError:
        pass


# --- every spelling the model actually produces -------------------------------

def test_all_three_anchor_spellings_produce_the_same_clean_result():
    """
    Across live turns phi4-mini wrote the same edit three different ways: FIND
    without the base indent and REPLACE without it either; FIND without it but
    REPLACE with it; and both fully verbatim. All three mean the same thing to
    a human. All three now mean the same thing here.

    Before, the first two went through substring matching and came out either
    unparseable or with total_count nested inside the method above it -- which
    the gate then reported as "removes existing definitions", sending the model
    off to fix a problem it did not have.
    """
    from arthur import patcher
    import ast

    spellings = [
        ("def total_count(self):",
         'def total_count(self):\n    """Return the total."""'),
        ("def total_count(self):",
         '    def total_count(self):\n        """Return the total."""'),
        ("    def total_count(self):\n        return sum(self.items.values())",
         '    def total_count(self):\n        """Return the total."""\n'
         "        return sum(self.items.values())"),
    ]

    for find, replace in spellings:
        new = edited(CLASS_FILE, find, replace)
        compile(new, "<test>", "exec")

        shape = patcher.analyze(CLASS_FILE, new, "inventory.py")
        assert shape.removed_symbols == [], (find, shape.removed_symbols)
        assert shape.gutted_symbols == [], (find, shape.gutted_symbols)

        method = next(n for n in ast.walk(ast.parse(new))
                      if isinstance(n, ast.FunctionDef) and n.name == "total_count")
        assert ast.get_docstring(method) == "Return the total."


# --- deeper nesting ----------------------------------------------------------

def test_a_replace_block_that_is_broken_on_its_own_terms_is_not_rescued():
    """
    The repair assumes REPLACE's INTERNAL structure is right and only its base
    indent is missing -- shifting a self-consistent block into place. A body
    that isn't indented under its own `def` is not that; it is wrong as
    standalone Python too, and inventing the missing structure would be
    guessing. It goes to the gate, which blocks it and says so.
    """
    new = edited(
        CLASS_FILE,
        "def total_count(self):",
        'def total_count(self):\n"""No indent at all."""',
    )
    try:
        compile(new, "<test>", "exec")
        raise AssertionError("expected this to stay broken")
    except IndentationError:
        pass


def test_a_replacement_inside_a_loop_inside_a_method():
    old = ('class A:\n'
           '    def run(self, xs):\n'
           '        for x in xs:\n'
           '            print(x)\n')
    new = edited(old, "print(x)", "print(x)\nprint(x * 2)")
    compile(new, "<test>", "exec")
    assert new.count("print(x") == 2
