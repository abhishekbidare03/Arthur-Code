"""
Appending a method to a class rather than beside it.

Observed live, on `now add a count method to cart.py`:

    +def count(self):
    +    return len(self.items)

At column zero, after a file that ended inside `class Cart`. So it is a
module-level function taking a parameter called `self`, which is not a method
and which nothing will ever call correctly.

Every check passed it. It compiles, so the structural veto had nothing to say;
no symbol was removed or hollowed; every existing line survived, which is the
property append_file is built around. The 1.5B critic approved it and reported
that "a `count` method has been added to the `Cart` class" -- reading the diff
the way a person skims it, by the words in it rather than the columns.

That is the shape of every failure left in this project: correct-looking wrong
answers that no structural rule catches. This one is an exception worth taking,
because the model's intent is not ambiguous at all. It wrote `self`. `self`
means method. Where the method goes is then a question about the file, and the
file can be asked.
"""

import ast
import pytest

from arthur import tools


CART = '''class Cart:
    def __init__(self):
        self.items = []

    def total(self):
        return sum(item["price"] for item in self.items)
'''

COUNT = ('def count(self):\n'
         '    """Return the number of items."""\n'
         '    return len(self.items)\n')


def methods_of(source, class_name):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == class_name)
    return [f.name for f in node.body if isinstance(f, ast.FunctionDef)]


def top_level_functions(source):
    return [n.name for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)]


# --- the fix -----------------------------------------------------------------

def test_a_method_lands_inside_the_class():
    new = tools.compute_append(CART, COUNT)
    assert "count" in methods_of(new, "Cart")
    assert top_level_functions(new) == []


def test_the_method_actually_works():
    """The test that would have caught it: call the thing."""
    new = tools.compute_append(CART, COUNT)
    ns = {}
    exec(compile(new, "<test>", "exec"), ns)
    cart = ns["Cart"]()
    cart.items = [{"price": 2}, {"price": 3}]
    assert cart.count() == 2
    assert cart.total() == 5


def test_nothing_existing_is_lost():
    """append_file's one guarantee, which the indentation must not break."""
    new = tools.compute_append(CART, COUNT)
    for line in CART.rstrip("\n").split("\n"):
        assert line in new.split("\n"), line
    assert methods_of(new, "Cart")[:2] == ["__init__", "total"]


def test_cls_counts_too():
    new = tools.compute_append(CART, "def build(cls):\n    return cls()\n")
    assert "build" in methods_of(new, "Cart")


def test_several_methods_at_once():
    two = ("def count(self):\n    return len(self.items)\n\n\n"
           "def clear(self):\n    self.items = []\n")
    new = tools.compute_append(CART, two)
    assert methods_of(new, "Cart") == ["__init__", "total", "count", "clear"]


def test_content_arriving_already_indented_is_not_double_indented():
    new = tools.compute_append(CART, "    def count(self):\n        return 1\n")
    compile(new, "<test>", "exec")
    assert "count" in methods_of(new, "Cart")


def test_pep8_spacing_inside_a_class():
    """One blank line between methods, not the two that separate top-level
    definitions."""
    new = tools.compute_append(CART, COUNT)
    body = new[len(CART):]
    assert body.startswith("\n    def count")


# --- where it must NOT indent ------------------------------------------------

def test_a_plain_function_is_not_dragged_into_the_class():
    """
    No `self`, so it is not a method, whatever the file happens to end with.
    Guessing here would put a helper somewhere the user never asked for.
    """
    new = tools.compute_append(CART, "def discount(price):\n    return price * 0.9\n")
    assert top_level_functions(new) == ["discount"]
    assert "discount" not in methods_of(new, "Cart")


def test_a_file_that_does_not_end_in_a_class_appends_at_the_margin():
    source = CART + "\n\ndef helper():\n    return 1\n"
    new = tools.compute_append(source, COUNT)
    assert "count" in top_level_functions(new)
    assert "count" not in methods_of(new, "Cart")


def test_a_file_with_no_class_at_all():
    new = tools.compute_append("def a():\n    return 1\n", COUNT)
    assert top_level_functions(new) == ["a", "count"]


def test_an_appended_class_is_never_nested():
    new = tools.compute_append(CART, "class Order:\n    pass\n")
    names = [n.name for n in ast.parse(new).body if isinstance(n, ast.ClassDef)]
    assert names == ["Cart", "Order"]


def test_an_empty_class_body_is_left_alone():
    """`class X: pass` has a body, but nothing to match indentation against
    that would be safe to guess from."""
    new = tools.compute_append("class Empty:\n    pass\n", COUNT)
    compile(new, "<test>", "exec")


# --- files that do not parse -------------------------------------------------

def test_unparseable_source_falls_back_to_a_plain_append():
    """
    The placement question cannot be answered without a parse, and refusing
    would be worse: the file is broken already and appending still cannot make
    it more broken, since every existing byte is kept either way.
    """
    broken = "class Cart:\n    def total(self)\n        return 1\n"
    new = tools.compute_append(broken, COUNT)
    assert new.startswith(broken)
    assert COUNT.rstrip("\n") in new


def test_unparseable_addition_falls_back_too():
    new = tools.compute_append(CART, "def count(self)\n    return 1\n")
    assert new.startswith(CART)


@pytest.mark.parametrize("source,addition", [
    ("", COUNT),
    (CART, ""),
    ("", ""),
    ("x = 1", COUNT),
])
def test_the_degenerate_cases_do_not_raise(source, addition):
    tools.compute_append(source, addition)


# --- through the tool --------------------------------------------------------

def test_append_file_writes_the_indented_version(tmp_path):
    path = tmp_path / "cart.py"
    path.write_text(CART, encoding="utf-8")

    result = tools.append_file({"path": "cart.py", "content": COUNT}, str(tmp_path))

    assert "APPENDED" in result
    assert "count" in methods_of(path.read_text(encoding="utf-8"), "Cart")


def test_the_preview_matches_what_gets_written(tmp_path):
    """The gate judges the preview; a mismatch would approve one thing and
    write another."""
    path = tmp_path / "cart.py"
    path.write_text(CART, encoding="utf-8")
    args = {"path": "cart.py", "content": COUNT}

    _, previewed = tools.preview_append(args, str(tmp_path))
    tools.append_file(args, str(tmp_path))

    assert path.read_text(encoding="utf-8") == previewed
