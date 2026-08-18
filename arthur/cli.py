"""
The `arthur` command.

Installed as a console script, so after `pip install -e .` this is what runs
when you type `arthur` in any terminal, in any repository:

    arthur                          interactive session in the current directory
    arthur -p "add a docstring"     one-shot: run a single task and exit
    arthur doctor                   pre-flight check on the configured backend

The default target repo is the directory you are standing in. That is the whole
ergonomic point -- `cd` into a project, type `arthur`, and it is already looking
at the right code, the same way `git` is.
"""

import argparse
import os
import sys

from . import __version__, config


SUBCOMMANDS = {"doctor", "runs"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arthur",
        description="A hand-built coding agent that gates its own file writes.",
        epilog=(
            "examples:\n"
            "  arthur                                  start an interactive session here\n"
            "  arthur -p \"fix the divide-by-zero bug\"   run one task and exit\n"
            "  arthur -p \"...\" --backend ollama         use the local model\n"
            "  arthur doctor                           check the backend is usable\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--print", dest="task", metavar="TASK",
        help="run a single task non-interactively and exit",
    )
    parser.add_argument(
        "-C", "--repo", default=None, metavar="PATH",
        help="repository to work in (default: the current directory)",
    )
    parser.add_argument(
        "--backend", choices=["mock", "ollama", "gemini"], default=None,
        help=f"which model backend to use (default: {config.BACKEND})",
    )
    parser.add_argument(
        "--model", default=None, metavar="NAME",
        help="override the model id for the chosen backend "
             f"(ollama: {config.OLLAMA_MODEL}, gemini: {config.GEMINI_MODEL})",
    )
    parser.add_argument(
        "--replay", metavar="FILE",
        help="pretty-print a saved run without calling any model "
             "(use `arthur runs` to list them)",
    )
    parser.add_argument(
        "--no-transcript", dest="save_transcript", action="store_false",
        help="don't record this run to ~/.arthur/runs/",
    )
    parser.add_argument(
        "-y", "--yes", dest="auto_approve", action="store_true",
        help="apply patches without asking, even below the confidence "
             "threshold (the gate still runs and is still reported)",
    )
    parser.add_argument(
        "--version", action="version", version=f"arthur {__version__}",
    )
    return parser


def _resolve_repo(path: str | None) -> str:
    repo = os.path.abspath(path or os.getcwd())
    if not os.path.isdir(repo):
        sys.exit(f"arthur: not a directory: {repo}")
    return repo


def _apply_overrides(backend: str, model: str | None) -> None:
    """
    Runtime overrides land in the config module, which every other module reads
    through at call time. Only the model is per-backend; everything else is
    already resolved by the time we get here.
    """
    config.BACKEND = backend
    if model:
        if backend == "ollama":
            config.OLLAMA_MODEL = model
        elif backend == "gemini":
            config.GEMINI_MODEL = model


def _preflight(backend: str) -> None:
    """Fail fast and legibly on the two mistakes people actually make."""
    if backend == "gemini" and not config.GEMINI_API_KEY:
        sys.exit(
            "arthur: no Gemini API key found.\n"
            "  Put GEMINI_API_KEY=your-key-here in a .env file, either in this\n"
            "  project's root or in ~/.arthur/.env, or export it in your shell.\n"
            "  Get a key at https://aistudio.google.com/apikey"
        )
    if backend == "ollama":
        try:
            import requests  # noqa: F401
        except ImportError:
            sys.exit("arthur: the ollama backend needs `requests`. "
                     "Install it with: pip install requests")


def _run_once(task: str, repo: str, backend: str, auto_approve: bool,
              save_transcript: bool = True) -> int:
    from . import intent
    from .render import run_to_terminal

    # Same guard as the interactive path: `arthur -p "hi"` should not index a
    # repository and wake a model up to be told there is nothing to do.
    kind = intent.classify(task)
    canned = intent.reply(kind)
    if canned:
        print(canned)
        return 0

    return run_to_terminal(task=task, repo_root=repo, backend=backend,
                           auto_approve=auto_approve,
                           save_transcript=save_transcript,
                           mode="answer" if kind == "question" else "edit")


def _replay(path: str) -> int:
    from . import transcript
    from .render import TerminalRenderer

    if not os.path.exists(path):
        sys.exit(f"arthur: no such transcript: {path}")
    try:
        return transcript.replay(path, TerminalRenderer())
    except ValueError as e:
        sys.exit(f"arthur: {e}")


def _list_runs() -> int:
    from . import transcript

    rows = transcript.list_runs()
    if not rows:
        print(f"No saved runs in {transcript.default_dir()}")
        return 0

    print(f"{len(rows)} most recent run(s) in {transcript.default_dir()}:\n")
    for path, when, task in rows:
        print(f"  {when}  {task[:60]}")
        print(f"              arthur --replay {path}")
    return 0


def _run_interactive(repo: str, backend: str, auto_approve: bool,
                     save_transcript: bool) -> int:
    from .session import run_session
    return run_session(repo=repo, backend=backend, auto_approve=auto_approve,
                       save_transcript=save_transcript)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Subcommands are matched before argparse sees them, so that the bare
    # `arthur` invocation stays a first-class thing rather than being forced
    # into an awkward `arthur repl`.
    if argv and argv[0] in SUBCOMMANDS:
        name, rest = argv[0], argv[1:]
        if name == "runs":
            return _list_runs()
        if name == "doctor":
            sub = argparse.ArgumentParser(prog="arthur doctor")
            sub.add_argument("--backend", choices=["mock", "ollama", "gemini"],
                             default=config.BACKEND)
            sub.add_argument("--model", default=None)
            sub_args = sub.parse_args(rest)
            _apply_overrides(sub_args.backend, sub_args.model)
            from . import doctor
            return doctor.run(sub_args.backend)

    args = _build_parser().parse_args(argv)

    # Replay reads a file and calls nothing, so it skips every backend check.
    if args.replay:
        return _replay(args.replay)

    repo = _resolve_repo(args.repo)
    backend = (args.backend or config.BACKEND).lower()
    _apply_overrides(backend, args.model)
    _preflight(backend)

    try:
        if args.task:
            return _run_once(args.task, repo, backend, args.auto_approve,
                             args.save_transcript)
        return _run_interactive(repo, backend, args.auto_approve,
                                args.save_transcript)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
