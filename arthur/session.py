"""
The interactive session -- what you get when you type `arthur` in a repo.

The difference from `arthur -p "task"` is not the prompt loop, it's the
**persistence**. One `Session` owns the conversation, so the second task knows
what the first one did:

    arthur > fix the divide-by-zero bug in calculator.py
      ...applies the patch...
    arthur > now add a test for it
      ...knows what "it" is...

A one-shot script structurally cannot do that, and it is the single biggest
difference in how the thing feels to use.

Input handling uses prompt_toolkit when it's installed (history, arrow keys,
sane Ctrl+C) and falls back to plain input() when it isn't, so the session
still works on a bare install.
"""

import os
import sys
import textwrap

from . import __version__, config, events as ev, intent, transcript
from .render import TerminalRenderer, drive, prompt_for_approval


BANNER = r"""
                _   _
     __ _  _ __| |_| |__  _   _ _ __
    / _` || '__| __| '_ \| | | | '__|
   | (_| || |  | |_| | | | |_| | |
    \__,_||_|   \__|_| |_|\__,_|_|
"""


class Session:
    """
    One conversation with one repository.

    `messages` is the whole point: it is handed to `agent.run()` on every task
    and mutated in place, so context accumulates across tasks the way it does
    in a real conversation.
    """

    def __init__(self, repo: str, backend: str, auto_approve: bool = False,
                 save_transcript: bool = True):
        self.repo = repo
        self.backend = backend
        self.auto_approve = auto_approve
        self.save_transcript = save_transcript

        self.messages: list[dict] = []
        self.events: list[ev.Event] = []       # everything, for /save
        self.tasks_run = 0
        self.files_changed: list[str] = []
        self.last_diff: str | None = None

    # -- state --------------------------------------------------------------

    @property
    def model(self) -> str:
        return {"ollama": config.OLLAMA_MODEL,
                "gemini": config.GEMINI_MODEL}.get(self.backend, "scripted")

    def reset(self) -> None:
        """Forget the conversation, keep the repo and settings."""
        self.messages.clear()
        self.tasks_run = 0

    def context_tokens(self) -> int:
        from . import context as ctx
        return ctx.estimate_messages_tokens(self.messages)


# --- slash commands ----------------------------------------------------------
#
# Handled locally and never sent to the model. Each returns True to keep the
# session going, False to end it.

COMMANDS: dict[str, str] = {
    "/help": "show this list",
    "/model": "show or change the model  (/model qwen2.5-coder:7b)",
    "/backend": "switch backend  (/backend ollama|gemini|mock)",
    "/repo": "show or change the working repository",
    "/context": "how much of the context window is in use",
    "/diff": "show the last proposed diff again",
    "/auto": "toggle auto-approve for patches that clear the gate",
    "/clear": "forget the conversation, keep the repo",
    "/save": "write this session's transcript now",
    "/runs": "list saved runs you can --replay",
    "/exit": "leave (Ctrl+D also works)",
}


def _handle_command(line: str, session: Session, renderer: TerminalRenderer) -> bool:
    c = renderer.c
    parts = line.split(maxsplit=1)
    cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

    def say(text=""):
        print(text)

    if cmd in ("/exit", "/quit", "/q"):
        return False

    if cmd == "/help":
        say()
        for name, description in COMMANDS.items():
            say(f"  {c('cyan', name.ljust(10))} {description}")
        say()
        say(c("dim", "  Describe a change and I'll propose a patch. Ask a "
                     "question and I'll answer it"))
        say(c("dim", "  without touching anything. Naming the file makes me "
                     "much more accurate."))
        say(c("dim", "  Prefix a path with @ to force it into context: @utils.py"))
        say()

    elif cmd == "/model":
        if not arg:
            say(f"  model: {session.model}  (backend {session.backend})")
        elif session.backend == "ollama":
            config.OLLAMA_MODEL = arg
            say(c("green", f"  model -> {arg}"))
        elif session.backend == "gemini":
            config.GEMINI_MODEL = arg
            say(c("green", f"  model -> {arg}"))
        else:
            say(c("yellow", "  the mock backend has no model to change"))

    elif cmd == "/backend":
        if arg not in ("ollama", "gemini", "mock"):
            say(f"  backend: {session.backend}  (choose ollama, gemini or mock)")
        else:
            session.backend = arg
            config.BACKEND = arg
            say(c("green", f"  backend -> {arg} ({session.model})"))

    elif cmd == "/repo":
        if not arg:
            say(f"  repo: {session.repo}")
        elif os.path.isdir(arg):
            session.repo = os.path.abspath(arg)
            session.reset()   # a new repo makes the old conversation nonsense
            say(c("green", f"  repo -> {session.repo}  (conversation cleared)"))
        else:
            say(c("red", f"  not a directory: {arg}"))

    elif cmd == "/context":
        used = session.context_tokens()
        limit = config.OLLAMA_NUM_CTX
        pct = 100 * used / limit if limit else 0
        say(f"  ~{used} of {limit} tokens ({pct:.0f}%), {len(session.messages)} messages, "
            f"{session.tasks_run} task(s) this session")
        if pct > 80:
            say(c("yellow", "  older steps will start being elided; /clear to reset"))

    elif cmd == "/diff":
        if session.last_diff:
            renderer.render(ev.DiffReady(path="(last proposed)", diff=session.last_diff))
        else:
            say(c("dim", "  no diff proposed yet this session"))

    elif cmd == "/auto":
        session.auto_approve = not session.auto_approve
        state = "on" if session.auto_approve else "off"
        say(c("green", f"  auto-approve {state}"))
        if session.auto_approve:
            say(c("dim", "  (patches that delete code are still blocked)"))

    elif cmd == "/clear":
        session.reset()
        say(c("green", "  conversation cleared"))

    elif cmd == "/save":
        if not session.events:
            say(c("dim", "  nothing to save yet"))
        else:
            path = transcript.new_path(f"session-{session.tasks_run}-tasks")
            transcript.save(path, session.events,
                            meta={"task": f"interactive session, "
                                          f"{session.tasks_run} task(s)",
                                  "repo": session.repo, "backend": session.backend})
            say(c("green", f"  saved {path}"))

    elif cmd == "/runs":
        rows = transcript.list_runs(limit=10)
        if not rows:
            say(c("dim", "  no saved runs yet"))
        for path, when, task in rows:
            say(f"  {when}  {task[:55]}")

    else:
        say(c("yellow", f"  unknown command {cmd} -- try /help"))

    return True


# --- @file mentions ----------------------------------------------------------

def expand_mentions(text: str, repo: str) -> str:
    """
    Inline files the user named with @path.

    Retrieval scores on keyword overlap, which misses a file whose name the
    user typed but whose contents share no vocabulary with the request. When
    someone points at a file explicitly, that is a stronger signal than any
    ranking, so it bypasses scoring entirely.
    """
    out = []
    for word in text.split():
        if not word.startswith("@") or len(word) < 2:
            continue
        candidate = os.path.join(repo, word[1:])
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="ignore") as fh:
                    body = fh.read()[:config.MAX_FILE_CHARS]
            except OSError:
                continue
            out.append(f"\n\nContents of {word[1:]}:\n```\n{body}\n```")

    return text + "".join(out)


# --- the loop ----------------------------------------------------------------

def _plain_reader():
    """Last-resort reader. Works anywhere, including piped stdin."""
    def read():
        line = input("arthur > ")
        # When input is piped, nothing echoes -- so echo it ourselves, or the
        # transcript of the session reads as answers with no questions.
        if not sys.stdin.isatty():
            print(line)
        return line
    return read


def _make_reader(session: Session):
    """
    prompt_toolkit when it can actually run, plain input() otherwise.

    Two separate reasons it can't: it may not be installed, and on Windows it
    refuses to start unless stdout is a real console -- which it isn't under
    Git Bash, or whenever input is piped. Both fall back rather than crash,
    because a session that won't start is worse than one without history.
    """
    if not sys.stdin.isatty():
        return _plain_reader()

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        history_dir = os.path.join(os.path.expanduser("~"), ".arthur")
        os.makedirs(history_dir, exist_ok=True)
        pt = PromptSession(
            history=FileHistory(os.path.join(history_dir, "history")),
            completer=WordCompleter(list(COMMANDS), sentence=True),
        )
        return lambda: pt.prompt("arthur > ")
    except Exception:
        # ImportError (not installed) or NoConsoleScreenBufferError (Git Bash,
        # mintty, piped output). Either way, degrade instead of dying.
        return _plain_reader()


def run_session(repo: str, backend: str, auto_approve: bool = False,
                save_transcript: bool = True) -> int:
    from . import agent

    session = Session(repo, backend, auto_approve, save_transcript)
    renderer = TerminalRenderer()
    c = renderer.c
    read = _make_reader(session)

    print(c("cyan", BANNER))
    print(f"  {c('bold', 'arthur ' + __version__)}   {session.backend} ({session.model})")
    print(c("dim", f"  {repo}"))
    print()
    if session.backend == "mock":
        # The mock replays one fixed script regardless of input. Without a loud
        # warning it looks like the agent is ignoring you and doing something
        # random -- which is exactly what it is doing.
        print(c("yellow", "  MOCK BACKEND -- replays a canned demo and ignores "
                          "what you type."))
        print(c("yellow", "  Use /backend ollama for the real thing."))
        print()

    print(c("dim", "  Describe a change and press enter. /help for commands, "
                   "Ctrl+D to leave."))
    print()

    while True:
        try:
            line = read().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            if not _handle_command(line, session, renderer):
                break
            continue

        # "hey" is not a task. Answering it here costs nothing and keeps the
        # model from inventing work to justify the run it was handed.
        kind = intent.classify(line)
        canned = intent.reply(kind)
        if canned:
            print()
            print(textwrap.indent(canned, "  "))
            print()
            continue

        # A question runs the agent too -- but read-only, so it answers rather
        # than hunting for something to rewrite.
        mode = "answer" if kind == "question" else "edit"

        task = expand_mentions(line, session.repo)
        collected: list[ev.Event] = []
        on_token = renderer.on_token if session.backend == "ollama" else None

        generator = agent.run(
            task=task, repo_root=session.repo, backend=session.backend,
            auto_approve=session.auto_approve, messages=session.messages,
            on_token=on_token, mode=mode,
        )

        try:
            drive(generator, renderer, sink=collected)
        except KeyboardInterrupt:
            # Cancel this task, keep the session and its history alive.
            generator.close()
            print()
            print(c("yellow", "  cancelled -- the conversation is still here"))
            continue
        except Exception as e:
            # The session is the valuable thing: the user has built up context
            # over several tasks, and throwing that away because one turn hit
            # an unhandled edge is the worst possible outcome. Report the bug,
            # keep the conversation, let them carry on.
            generator.close()
            print()
            print(c("red", f"  this task hit a bug: {type(e).__name__}: {e}"))
            print(c("dim", "  the session and its history are intact -- "
                           "try rephrasing, or /save to keep the transcript"))
            if os.environ.get("ARTHUR_DEBUG"):
                import traceback
                traceback.print_exc()
            else:
                print(c("dim", "  set ARTHUR_DEBUG=1 for the traceback"))
            continue

        session.tasks_run += 1
        session.events.extend(collected)
        for event in collected:
            if isinstance(event, ev.DiffReady):
                session.last_diff = event.diff
            elif isinstance(event, ev.Final):
                session.files_changed.extend(event.files_changed)

    # Leaving: save the whole session, so a good demo isn't lost by habit.
    if save_transcript and session.events:
        try:
            path = transcript.new_path(f"session-{session.tasks_run}-tasks")
            transcript.save(path, session.events,
                            meta={"task": f"interactive session, "
                                          f"{session.tasks_run} task(s)",
                                  "repo": session.repo, "backend": session.backend})
            print(c("dim", f"  transcript: {path}"))
        except OSError:
            pass

    changed = list(dict.fromkeys(session.files_changed))
    if changed:
        print(c("dim", f"  files changed this session: {', '.join(changed)}"))
    print(c("dim", "  bye."))
    return 0
