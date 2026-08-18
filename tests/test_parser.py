"""
Parser tests.

    pytest                 (if you have it)
    python tests/test_parser.py    (if you don't -- there is a runner at the bottom)

Every case below is a real failure mode observed from chat-tuned models against
the THOUGHT/ACTION/ARGS protocol. The mock backend never produces any of them --
which is exactly why the parser needs its own test set before you trust a live
run against a 3B model.
"""

from arthur.agent import parse_response

CASES = []


def case(name):
    """Register for the no-pytest runner. Functions are still named test_* so
    pytest collects them the normal way."""
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


@case("clean protocol output")
def test_clean_protocol_output():
    p = parse_response('THOUGHT: Look first.\nACTION: read_file\nARGS: {"path": "calculator.py"}')
    assert p["thought"] == "Look first."
    assert p["action"] == "read_file"
    assert p["args"] == {"path": "calculator.py"}


@case("markdown-bolded keywords")
def test_markdown_bolded_keywords():
    p = parse_response('**THOUGHT:** Reading it.\n**ACTION:** read_file\n**ARGS:** {"path": "a.py"}')
    assert p["action"] == "read_file", p
    assert p["args"] == {"path": "a.py"}


@case("ARGS wrapped in a json code fence")
def test_args_in_code_fence():
    p = parse_response(
        'THOUGHT: Fix it.\nACTION: apply_patch\nARGS:\n```json\n{"path": "a.py", "new_content": "x = 1\\n"}\n```'
    )
    assert p["args"]["new_content"] == "x = 1\n", p


@case("literal newlines inside the JSON string")
def test_literal_newlines_in_json_string():
    # Invalid JSON per spec -- models do this constantly with full-file rewrites.
    p = parse_response(
        'THOUGHT: Rewrite.\nACTION: apply_patch\nARGS: {"path": "a.py", "new_content": "def f():\n    return 1\n"}'
    )
    assert p["args"]["new_content"] == "def f():\n    return 1\n", p


@case("trailing prose after the JSON")
def test_trailing_prose_after_json():
    p = parse_response(
        'THOUGHT: Go.\nACTION: read_file\nARGS: {"path": "a.py"}\n\nThat should do it! Let me know.'
    )
    assert p["args"] == {"path": "a.py"}, p


@case("confidence given as a percentage")
def test_confidence_as_percentage():
    p = parse_response('THOUGHT: Sure.\nCONFIDENCE: 85\nACTION: apply_patch\nARGS: {"path": "a.py", "new_content": ""}')
    assert p["confidence"] == 0.85, p


@case("confidence as a quoted decimal")
def test_confidence_quoted_decimal():
    p = parse_response('CONFIDENCE: "0.4"\nACTION: apply_patch\nARGS: {}')
    assert p["confidence"] == 0.4, p


@case("FINAL ends the turn")
def test_final_ends_the_turn():
    p = parse_response("THOUGHT: Done.\nFINAL: Added a guard to divide().")
    assert p["final"] == "Added a guard to divide()."
    assert p["action"] is None


@case("'FINAL:' inside patch content does not end the turn")
def test_final_inside_patch_content():
    # The killer bug in a naive parser: the agent 'finishes' mid-edit.
    p = parse_response(
        'THOUGHT: Add a log line.\nCONFIDENCE: 0.9\nACTION: apply_patch\n'
        'ARGS: {"path": "a.py", "new_content": "print(\\"FINAL: done\\")\\n"}'
    )
    assert p["final"] is None, p
    assert p["action"] == "apply_patch"


@case("braces inside the patch content don't truncate the JSON")
def test_braces_inside_patch_content():
    p = parse_response(
        'ACTION: apply_patch\nARGS: {"path": "a.py", "new_content": "d = {\\"k\\": {\\"n\\": 1}}\\n"}'
    )
    assert p["args"]["new_content"] == 'd = {"k": {"n": 1}}\n', p


@case("pure prose -> no action, caller re-prompts")
def test_pure_prose_yields_nothing():
    p = parse_response("Sure! I'd be happy to help you fix that division bug.")
    assert p["action"] is None and p["final"] is None and p["args"] is None


@case("unparseable ARGS degrades to empty dict, not a crash")
def test_unparseable_args_degrades():
    p = parse_response("ACTION: read_file\nARGS: path=calculator.py")
    assert p["action"] == "read_file"
    assert p["args"] == {}


# --- captured from live qwen2.5-coder:3b runs --------------------------------

@case("python triple-quoted string as a JSON value (qwen2.5-coder:3b)")
def test_triple_quoted_new_content():
    """
    Verbatim from the first live local run. The model reached for Python's
    triple-quote syntax instead of a JSON string, which both breaks json.loads
    and desynchronises the brace scanner -- and because a local model at low
    temperature is near-deterministic, it repeated it until the step cap.
    """
    raw = (
        'THOUGHT: I will add a guard.\n'
        'CONFIDENCE: 0.9\n'
        'ACTION: apply_patch\n'
        'ARGS: {"path": "calculator.py", "new_content": """\n'
        'def divide(a, b):\n'
        '    try:\n'
        '        return a / b\n'
        '    except ZeroDivisionError:\n'
        '        return "Error: Division by zero"\n'
        '"""}'
    )
    p = parse_response(raw)
    assert p["action"] == "apply_patch"
    assert p["args"].get("path") == "calculator.py"
    assert "def divide(a, b):" in p["args"].get("new_content", "")
    assert "ZeroDivisionError" in p["args"]["new_content"]


@case("triple-quoted value whose content has its own docstring")
def test_triple_quoted_content_containing_triple_quotes():
    """
    The nasty one. The file being written starts with a module docstring, so
    there are THREE pairs of triple quotes in the line. A non-greedy match
    stops at the inner docstring and truncates the rest of the file -- which
    produces a plausible patch that silently deletes most of the module.
    """
    raw = (
        'ACTION: apply_patch\n'
        'ARGS: {"path": "m.py", "new_content": """"""A module."""\n'
        '\n'
        'def alpha():\n'
        '    return 1\n'
        '\n'
        'def omega():\n'
        '    return 2\n'
        '"""}'
    )
    content = parse_response(raw)["args"].get("new_content", "")
    assert "def alpha" in content
    assert "def omega" in content, "content was truncated at the inner docstring"


@case("valid JSON containing triple quotes is not mangled")
def test_valid_json_with_triple_quotes_is_left_alone():
    """The repair must not fire on input that already parses."""
    raw = (
        'ACTION: apply_patch\n'
        r'ARGS: {"path": "m.py", "new_content": "\"\"\"Doc.\"\"\"\ndef f():\n    return 1\n"}'
    )
    content = parse_response(raw)["args"]["new_content"]
    assert content == '"""Doc."""\ndef f():\n    return 1\n'


@case("block form: PATH/CONTENT with a fenced file")
def test_block_form_basic():
    """
    The escape-free spelling for apply_patch. Embedding a whole file in JSON is
    the single thing a 3B is worst at -- it failed differently on every live
    run -- while fenced code blocks are what it produces most reliably.
    """
    raw = (
        "THOUGHT: add a guard.\n"
        "CONFIDENCE: 0.8\n"
        "ACTION: apply_patch\n"
        "PATH: calculator.py\n"
        "CONTENT:\n"
        "```python\n"
        "def divide(a, b):\n"
        '    if b == 0:\n'
        '        raise ValueError("no")\n'
        "    return a / b\n"
        "```"
    )
    p = parse_response(raw)
    assert p["action"] == "apply_patch"
    assert p["args"]["path"] == "calculator.py"
    assert p["args"]["new_content"].startswith("def divide(a, b):")
    assert p["args"]["new_content"].endswith("return a / b\n")
    assert p["confidence"] == 0.8


@case("block form: content keeps its own quotes and braces verbatim")
def test_block_form_no_escaping_needed():
    raw = (
        "ACTION: apply_patch\nPATH: m.py\nCONTENT:\n```python\n"
        '"""A module."""\n'
        'd = {"k": {"n": 1}}\n'
        "```"
    )
    content = parse_response(raw)["args"]["new_content"]
    assert '"""A module."""' in content
    assert 'd = {"k": {"n": 1}}' in content


@case("block form: unfenced content runs to end of message")
def test_block_form_without_a_fence():
    raw = "ACTION: apply_patch\nPATH: m.py\nCONTENT:\nx = 1\ny = 2\n"
    assert parse_response(raw)["args"]["new_content"] == "x = 1\ny = 2\n"


@case("JSON args still win when they parse")
def test_json_args_take_precedence_over_block_form():
    raw = ('ACTION: apply_patch\n'
           'ARGS: {"path": "a.py", "new_content": "from_json\\n"}')
    assert parse_response(raw)["args"]["new_content"] == "from_json\n"


@case("action before FINAL wins (model inventing its own success)")
def test_action_preceding_final_is_not_discarded():
    """
    Observed live: the 3B emits a patch, then immediately writes a second
    THOUGHT and a FINAL congratulating itself on a tool call that never ran.
    Honouring the FINAL silently drops the patch and reports a success that
    changed nothing.
    """
    raw = (
        "THOUGHT: patching.\n"
        "ACTION: apply_patch\n"
        "PATH: m.py\n"
        "CONTENT:\n```python\nx = 1\n```\n\n"
        "THOUGHT: that worked.\n"
        "FINAL: The file has been updated."
    )
    p = parse_response(raw)
    assert p["action"] == "apply_patch", "the patch must not be discarded"
    assert p["final"] is None
    assert p["args"]["new_content"] == "x = 1\n"


@case("a genuine FINAL with no action still ends the run")
def test_final_alone_still_terminates():
    p = parse_response("THOUGHT: nothing to do.\nFINAL: No changes were needed.")
    assert p["final"] == "No changes were needed."
    assert p["action"] is None


@case("Title Case keywords (phi4-mini)")
def test_title_case_keywords():
    """
    Captured live from phi4-mini, which follows the protocol faithfully but
    writes it in Title Case. A case-sensitive parser reads this as prose and
    the model looks broken while doing exactly as instructed.
    """
    raw = ('Thought: I need to read the file first.\n'
           'Action: read_file\n'
           'Args: {"path": "calculator.py"}')
    p = parse_response(raw)
    assert p["action"] == "read_file"
    assert p["args"] == {"path": "calculator.py"}
    assert p["thought"].startswith("I need to read")


@case("lower case keywords")
def test_lower_case_keywords():
    p = parse_response('thought: go\naction: read_file\nargs: {"path": "a.py"}')
    assert p["action"] == "read_file"


@case("mixed case with markdown decoration")
def test_mixed_case_and_decoration():
    p = parse_response('**Action:** read_file\n**Args:** {"path": "a.py"}')
    assert p["action"] == "read_file"


@case("keywords inside a fenced block are left alone")
def test_keywords_inside_fences_are_not_rewritten():
    """
    The blocks carry whole source files. A line like `# Action: rename this` is
    ordinary code -- rewriting it would corrupt what gets written to disk, and
    could be read as a second action.
    """
    raw = ("ACTION: apply_patch\nPATH: m.py\nCONTENT:\n```python\n"
           "# Action: tidy this up later\n"
           "# Final: ship it\n"
           "x = 1\n```")
    content = parse_response(raw)["args"]["new_content"]
    assert "# Action: tidy this up later" in content
    assert "# Final: ship it" in content


@case("a Title Case FINAL inside content does not end the turn")
def test_title_case_final_in_content():
    raw = ("ACTION: apply_patch\nPATH: m.py\nCONTENT:\n```python\n"
           'print("Final: done")\n```')
    p = parse_response(raw)
    assert p["final"] is None
    assert p["action"] == "apply_patch"


@case("blank lines between protocol keywords")
def test_blank_lines_between_keywords():
    # The 3B pads its output with blank lines between every section.
    p = parse_response(
        'THOUGHT: Reading it.\n\nACTION: read_file\n\nARGS: {"path": "a.py"}'
    )
    assert p["action"] == "read_file", p
    assert p["args"] == {"path": "a.py"}


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- the fence on the keyword's own line -------------------------------------

INLINE_FENCE = (
    'THOUGHT: adding the docstring.\n'
    'ACTION: insert_after\n'
    'PATH: inventory.py\n'
    'AFTER: ```\n'
    'def total_count(self):\n'
    '```\n'
    'CONTENT: ```\n'
    '"""Return the total number of items."""\n'
    '```'
)


def test_a_fence_on_the_keyword_line_still_parses():
    """
    Captured verbatim from phi4-mini. The protocol shows the fence on the line
    BELOW the keyword and the model put it on the same line -- a formatting
    slip, not a different intention. Every block came back None, so the turn
    arrived as `insert_after()` with no arguments at all, and the observation
    said "missing: path, after" about a turn that had supplied both. The model
    apologised and re-sent it identically twice more.
    """
    parsed = parse_response(INLINE_FENCE)
    assert parsed["action"] == "insert_after"
    assert parsed["args"]["path"] == "inventory.py"
    assert parsed["args"]["after"] == "def total_count(self):"
    assert parsed["args"]["content"] == '"""Return the total number of items."""'


def test_the_two_layouts_parse_identically():
    """Whichever line the fence lands on, the same call comes out."""
    onto_next_line = INLINE_FENCE.replace(": ```", ":\n```")
    assert parse_response(onto_next_line) == parse_response(INLINE_FENCE)


def test_it_works_for_find_and_replace_too():
    text = ('ACTION: edit_file\nPATH: a.py\n'
            'FIND: ```\nold_line = 1\n```\n'
            'REPLACE: ```\nnew_line = 2\n```')
    parsed = parse_response(text)
    assert parsed["args"]["find"] == "old_line = 1"
    assert parsed["args"]["replace"] == "new_line = 2"


def test_a_language_tag_on_the_keyword_line_is_fine():
    text = 'ACTION: append_file\nPATH: a.py\nCONTENT: ```python\nx = 1\n```'
    # append_file's content is destined for the end of a file, so it is
    # normalized to end with a newline; the tag itself must not survive.
    assert parse_response(text)["args"]["content"].strip() == "x = 1"
    assert "python" not in parse_response(text)["args"]["content"]


def test_text_after_the_keyword_that_is_not_a_fence_is_unaffected():
    """
    The relaxation only fires when a fence follows immediately. `CONTENT: x = 1`
    on one line must keep whatever behaviour it had, not be swallowed as a
    block opener.
    """
    text = 'ACTION: append_file\nPATH: a.py\nCONTENT: x = 1\n'
    parsed = parse_response(text)
    assert parsed["action"] == "append_file"


# --- blank lines against a fence ---------------------------------------------

BLANK_EDGED = (
    'THOUGHT: adding the docstring.\n'
    'ACTION: insert_after\n'
    'PATH: inventory.py\n'
    'AFTER: ```\n'
    '\n'
    'def total_count(self):\n'
    '```\n'
    'CONTENT: ```\n'
    '\n'
    '"""Return the total number of items."""\n'
    '```'
)


def test_a_blank_line_after_the_fence_is_not_part_of_the_block():
    """
    Captured live, and the reason `docstring` failed two runs in five. The
    blank line makes the anchor the two lines ["", "def total_count(self):"],
    which demands a blank line ABOVE the def as well -- so it stops matching,
    and the model is told its anchor "does not appear in the file" when the
    line it named is sitting right there.
    """
    args = parse_response(BLANK_EDGED)["args"]
    assert args["after"] == "def total_count(self):"
    assert args["content"] == '"""Return the total number of items."""'


def test_the_blank_lines_change_nothing_about_the_result():
    without = BLANK_EDGED.replace("```\n\n", "```\n")
    assert parse_response(without) == parse_response(BLANK_EDGED)


def test_trailing_blank_lines_go_too():
    text = ('ACTION: edit_file\nPATH: a.py\n'
            'FIND:\n```\nx = 1\n\n\n```\n'
            'REPLACE:\n```\nx = 2\n\n```')
    args = parse_response(text)["args"]
    assert args["find"] == "x = 1"
    assert args["replace"] == "x = 2"


def test_blank_lines_INSIDE_a_block_are_untouched():
    """Only the edges are layout. A blank line between two statements is code."""
    text = ('ACTION: append_file\nPATH: a.py\n'
            'CONTENT:\n```\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n```')
    assert "return 1\n\n\ndef b" in parse_response(text)["args"]["content"]


def test_an_empty_block_stays_empty():
    """An empty REPLACE is a deletion, not a malformed turn -- it must survive
    the trimming as an empty string and not become a missing block."""
    text = 'ACTION: edit_file\nPATH: a.py\nFIND:\n```\nx = 1\n```\nREPLACE:\n```\n```'
    args = parse_response(text)["args"]
    assert args["find"] == "x = 1"
    assert args["replace"] == ""


def test_a_block_of_only_blank_lines_is_empty_not_whitespace():
    text = 'ACTION: edit_file\nPATH: a.py\nFIND:\n```\nx = 1\n```\nREPLACE:\n```\n\n\n```'
    assert parse_response(text)["args"]["replace"] == ""


# --- every block layout phi4-mini actually writes -----------------------------
#
# The corpus. Each entry is a real layout captured from live runs, and the
# point of collecting them in one place is the invariant below: they are all
# the SAME call, so they must all parse to the same arguments. Four of these
# were found one at a time, each after a task had already been lost to it, and
# each was fixed in isolation before anyone thought to ask how many more there
# were. This is the "how many more" test.

LAYOUTS = {
    "documented":       "AFTER:\n```\ndef total_count(self):\n```\n"
                        "CONTENT:\n```\nx = 1\n```",
    "language tag":     "AFTER:\n```python\ndef total_count(self):\n```\n"
                        "CONTENT:\n```python\nx = 1\n```",
    "fence on keyword": "AFTER: ```\ndef total_count(self):\n```\n"
                        "CONTENT: ```\nx = 1\n```",
    "blank after fence": "AFTER: ```\n\ndef total_count(self):\n```\n"
                         "CONTENT: ```\n\nx = 1\n```",
    "no fence at all":  "AFTER: def total_count(self):\nCONTENT: x = 1",
    "inline backticks": "AFTER: `def total_count(self):`\nCONTENT: `x = 1`",
    "inline triple":    "AFTER: ```def total_count(self):```\n"
                        "CONTENT: ```x = 1```",
    "unbalanced close": "AFTER: ```def total_count(self):``\n"
                        "CONTENT: ```x = 1``",
    "inline below":     "AFTER:\n```def total_count(self):```\n"
                        "CONTENT:\n```x = 1```",
}


@case("every captured block layout is the same call")
def test_every_captured_layout_is_the_same_call():
    for name, layout in LAYOUTS.items():
        text = f"ACTION: insert_after\nPATH: inventory.py\n{layout}"
        args = parse_response(text)["args"]
        assert args is not None, name
        assert args.get("path") == "inventory.py", name
        assert args.get("after") == "def total_count(self):", name
        assert args.get("content") == "x = 1", name


def test_no_layout_swallows_the_following_keyword():
    """
    The specific catastrophe. `AFTER: ```def f():``` ` used to parse the anchor
    as the literal string "CONTENT:", because the opening-fence pattern wanted
    a newline, so matching resumed on the next line and closed on the fence
    that OPENED the following block. An anchor of "CONTENT:" matches nothing,
    forever, and every check upstream passed it.
    """
    for name, layout in LAYOUTS.items():
        text = f"ACTION: insert_after\nPATH: inventory.py\n{layout}"
        args = parse_response(text)["args"] or {}
        for value in args.values():
            assert "CONTENT:" not in str(value), (name, value)
            assert "AFTER:" not in str(value), (name, value)


def test_no_layout_leaks_markup_into_the_value():
    """Backticks written to a file are a syntax error, not a docstring."""
    for name, layout in LAYOUTS.items():
        text = f"ACTION: insert_after\nPATH: inventory.py\n{layout}"
        args = parse_response(text)["args"] or {}
        for value in args.values():
            assert "`" not in str(value), (name, value)


@case("inline-code shortcut never truncates a real multi-line block")
def test_inline_shortcut_does_not_truncate_a_multiline_block():
    """
    The shortcut takes a first line that looks like inline code as the whole
    value. In a Markdown file a line of `code` is ordinary content, so a real
    multi-line block can begin that way -- and truncating it would silently
    write a fraction of what the model sent.
    """
    text = ('ACTION: append_file\nPATH: notes.md\nCONTENT:\n```\n'
            '`arthur doctor`\nchecks the GPU and the daemon.\n```')
    content = parse_response(text)["args"]["content"]
    assert "checks the GPU and the daemon." in content
    assert content.startswith("`arthur doctor`")
