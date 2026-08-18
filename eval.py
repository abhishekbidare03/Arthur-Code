"""
The acceptance harness: 3 small, realistic tasks against a scratch repo.

This exists because "the agent works" is not a claim you can defend without a
number. It resets the fixtures, runs each task end to end, and grades the
result by inspecting the file afterwards -- not by believing the model's
summary, which is exactly the thing that cannot be trusted.

    python eval.py --backend ollama
    python eval.py --backend gemini
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time

INVENTORY = '''"""A small stock-tracking module."""


class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, quantity):
        self.items[name] = self.items.get(name, 0) + quantity

    def remove_item(self, name, quantity):
        self.items[name] = self.items[name] - quantity

    def total_count(self):
        return sum(self.items.values())


def last_n_items(names, n):
    return names[-(n + 1):]


def find_item(items, name):
    for item in items:
        if item.name == name:
            return item
    return None
'''

UTILS = '''def divide(a, b):
    return a / b


def safe_get(d, key):
    return d[key].strip()
'''


def symbols(source: str) -> set[str]:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(f"{node.name}.{sub.name}")
    return names


# Each check gets the file's text and returns (passed, why).
# They deliberately test BEHAVIOUR, not the presence of particular words.

def check_docstring(src: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "total_count":
            doc = ast.get_docstring(node)
            if doc:
                return True, f"docstring: {doc.splitlines()[0][:50]!r}"
            return False, "total_count still has no docstring"
    return False, "total_count is gone"


def check_null_guard(src: str):
    ns = {}
    exec(compile(src, "utils.py", "exec"), ns)
    try:
        result = ns["safe_get"]({}, "missing")
    except Exception as e:
        return False, f"still raises {type(e).__name__}"
    return result is None, f"returns {result!r} for a missing key"


def check_off_by_one(src: str):
    ns = {}
    exec(compile(src, "inventory.py", "exec"), ns)
    got = ns["last_n_items"]([1, 2, 3, 4, 5], 2)
    return list(got) == [4, 5], f"last_n_items([1..5], 2) -> {got}"


def check_added_function(src: str):
    """
    Adding to a file, which is the case that used to fail hardest: with only
    edit_file available the model put the OLD function in FIND and the NEW one
    in REPLACE, deleting the original. So this grades both halves -- the new
    function must work AND everything that was there must still be there.
    """
    before = symbols(UTILS)
    after = symbols(src)
    lost = before - after
    if lost:
        return False, f"deleted {', '.join(sorted(lost))} while adding"

    ns = {}
    exec(compile(src, "utils.py", "exec"), ns)
    if "multiply" not in ns:
        return False, "multiply was never added"
    got = ns["multiply"](3, 4)
    return got == 12, f"multiply(3, 4) -> {got}"


TASKS = [
    ("docstring", "inventory.py", INVENTORY,
     "Add a docstring to the total_count method in inventory.py explaining what it returns",
     check_docstring),
    ("null-check", "utils.py", UTILS,
     "In utils.py, make safe_get return None when the key is missing instead of raising KeyError",
     check_null_guard),
    ("off-by-one", "inventory.py", INVENTORY,
     "Fix the off-by-one bug in last_n_items in inventory.py so it returns exactly n items",
     check_off_by_one),
    ("add-function", "utils.py", UTILS,
     "Add a new function multiply(a, b) to utils.py that returns a times b",
     check_added_function),
]


def run_one(name, filename, seed, prompt, check, backend, model):
    workdir = tempfile.mkdtemp(prefix=f"arthur-eval-{name}-")
    path = os.path.join(workdir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(seed)

    cmd = [sys.executable, "-m", "arthur.cli", "-p", prompt, "--backend", backend, "-y"]
    if model:
        cmd += ["--model", model]

    started = time.time()
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - started

    with open(path, "r", encoding="utf-8") as fh:
        after = fh.read()

    # Grade in order of severity: a file you can't parse is the worst outcome,
    # then losing code, then failing to do the job.
    try:
        ast.parse(after)
    except SyntaxError as e:
        return name, False, f"file no longer parses: {e}", elapsed
    lost = symbols(seed) - symbols(after)
    if lost:
        return name, False, f"deleted {', '.join(sorted(lost))}", elapsed
    try:
        ok, why = check(after)
    except Exception as e:
        return name, False, f"check raised {type(e).__name__}: {e}", elapsed

    shutil.rmtree(workdir, ignore_errors=True)
    return name, ok, why, elapsed


def screen(backend: str, model: str | None) -> bool:
    """
    A 30-second sanity check before committing to the full eval.

    Learned the hard way: qwen3:4b spent 25 minutes to score 1/3, and a
    /no_think variant was on course for over an hour, because it answered every
    turn with a page of prose instead of the protocol. One probe would have
    said so immediately. Checks the two things that make a run pointless --
    the model ignores the format, or it is too slow to finish.
    """
    from arthur import config
    from arthur.agent import parse_response
    from arthur.llm_backend import get_backend

    if model:
        config.OLLAMA_MODEL = model
    probe = [
        {"role": "system", "content": (
            "You are a coding agent. Respond using EXACTLY this protocol and nothing else:\n"
            "THOUGHT: <reasoning>\nACTION: <tool name>\nARGS: <one-line JSON>")},
        {"role": "user", "content": "Read calculator.py. The only tool is read_file(path)."},
    ]

    started = time.time()
    try:
        response = get_backend(backend).chat(probe)
    except Exception as e:
        print(f"  screen: FAILED -- {e}\n")
        return False
    elapsed = time.time() - started

    parsed = parse_response(response.text)
    ok_format = parsed["action"] == "read_file"
    ok_speed = elapsed < 30

    print(f"  screen: {elapsed:5.1f}s/turn, {len(response.text):4d} chars, "
          f"action={parsed['action']!r}")
    if not ok_format:
        print("          -> ignores the protocol; the full eval would just "
              "burn steps re-prompting it")
    if not ok_speed:
        print(f"          -> ~{elapsed * 12 * 3 / 60:.0f} min for the full eval "
              "at this rate")
    print()
    return ok_format and ok_speed


def main():
    ap = argparse.ArgumentParser(description="Arthur acceptance eval")
    ap.add_argument("--backend", default="ollama", choices=["mock", "ollama", "gemini"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--force", action="store_true",
                    help="run the tasks even if the screen says the model is unusable")
    args = ap.parse_args()

    print(f"backend={args.backend} model={args.model or 'default'}\n")

    if not screen(args.backend, args.model) and not args.force:
        print("Stopping. Re-run with --force to grade it anyway.")
        return 2

    results = []
    for task in TASKS:
        name, ok, why, elapsed = run_one(*task, args.backend, args.model)
        results.append(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {name:<12} {elapsed:6.1f}s  {why}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
