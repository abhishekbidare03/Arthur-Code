"""
Turning the event stream into something a human reads.

This is the only place in the project that prints, and `drive()` is the only
place that answers an ApprovalNeeded. Keeping both here is what lets the same
agent loop serve a live terminal, a JSON transcript, and a test without
knowing which one it is talking to.

Colour is ANSI written by hand rather than a dependency. `rich` would be
nicer, and the interactive session will pull it in -- but a one-shot run
should not need it, and diffs only really want three colours anyway.
"""

import os
import sys

from . import config, events as ev


# --- colour ------------------------------------------------------------------

def _supports_colour(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class Palette:
    """No-ops itself when the output isn't a terminal, so piping to a file or
    capturing in a test gives clean text."""

    CODES = {
        "dim": "\033[2m", "bold": "\033[1m", "red": "\033[31m",
        "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m",
        "cyan": "\033[36m", "reset": "\033[0m",
    }

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, name: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"{self.CODES[name]}{text}{self.CODES['reset']}"


# --- rendering ---------------------------------------------------------------

class TerminalRenderer:
    # Where live echoing stops. Everything up to one of these is the model's
    # reasoning, which is worth watching arrive on a slow local model.
    # Everything after is protocol payload -- possibly an entire rewritten file
    # as JSON -- which the structured renderer displays far better as a diff.
    _CUT_AT = ("ACTION:", "FINAL:")

    def __init__(self, out=None, verbose: bool = True):
        self.out = out or sys.stdout
        self.c = Palette(_supports_colour(self.out))
        self.verbose = verbose
        self._streaming = False
        self._buf = ""
        self._echoed = 0        # chars of _buf already written out
        self._cut = False       # hit a protocol keyword; stop echoing
        self._mode = "edit"     # set from RunStarted; see _on_final

    def _p(self, text: str = "") -> None:
        print(text, file=self.out)

    # A model streaming tokens is mid-line; anything else we print must start
    # on a fresh one or it lands in the middle of the model's sentence.
    def _endstream(self) -> None:
        if self._streaming:
            self._p()
            self._streaming = False

    def _reset_stream(self) -> None:
        self._buf, self._echoed, self._cut = "", 0, False

    def on_token(self, chunk: str) -> None:
        """
        Echo the model's reasoning as it arrives, then go quiet.

        A 3B on a 4GB card takes ~10s a turn; without this the terminal looks
        hung. But echoing the whole turn means printing the JSON payload twice
        -- once raw, once as a rendered diff -- so we stop at the first
        protocol keyword and let the structured events take over.
        """
        if not self.verbose or self._cut:
            return

        self._buf += chunk
        cut_at = min((self._buf.find(k) for k in self._CUT_AT if k in self._buf),
                     default=-1)

        visible = self._buf if cut_at < 0 else self._buf[:cut_at]
        new = visible[self._echoed:]
        if new:
            self._streaming = True
            self.out.write(self.c("dim", new))
            self.out.flush()
            self._echoed = len(visible)

        if cut_at >= 0:
            self._cut = True
            self._endstream()

    def streamed_thought(self) -> bool:
        """True when the thought for this step already appeared live, so the
        structured renderer shouldn't print it a second time."""
        return self._echoed > 0

    def _already_streamed(self, text: str) -> bool:
        """
        Has the user just watched this exact text arrive token by token?

        Compared on collapsed whitespace: the buffer holds raw stream chunks
        and the summary has been stripped, so they differ by leading and
        trailing space that nobody can see.
        """
        if not self._echoed or not text:
            return False
        return " ".join(text.split()) in " ".join(self._buf.split())

    def render(self, event: ev.Event) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler:
            self._endstream()
            handler(event)

    # -- lifecycle --

    def _on_run_started(self, e):
        self._mode = getattr(e, "mode", "edit")
        self._p()
        self._p(self.c("bold", f"arthur  {e.backend} ({e.model})"))
        self._p(self.c("dim", f"repo: {e.repo_root}"))
        label = "question" if self._mode == "answer" else "task"
        self._p(f"{label}: {e.task}")
        if self._mode == "answer":
            self._p(self.c("dim", "read-only -- nothing will be written"))
        self._p()

    def _on_index_built(self, e):
        self._p(self.c("dim", f"indexed {e.file_count} file(s)"))

    def _on_context_retrieved(self, e):
        if not e.files:
            self._p(self.c("yellow", "no relevant files found; the agent will explore"))
            return
        listed = ", ".join(f"{f['path']} ({f['score']})" for f in e.files)
        self._p(self.c("dim", f"context: {listed}"))
        if e.dropped:
            self._p(self.c("dim", f"         {e.dropped} more file(s) dropped to fit the window"))

    # -- per step --

    def _on_step_started(self, e):
        self._p()
        self._p(self.c("dim", f"--- step {e.step}/{e.max_steps} ---"))
        self._reset_stream()

    def _on_thought(self, e):
        if self.streamed_thought():
            return  # the user already watched it arrive token by token
        self._p(f"{self.c('cyan', 'THOUGHT')} {e.text}")

    def _on_action_proposed(self, e):
        conf = ""
        if e.confidence is not None:
            conf = self.c("dim", f"  (confidence {e.confidence:.2f})")
        self._p(f"{self.c('blue', 'ACTION')}  {e.action}({_summarize(e.args)}){conf}")

    def _on_diff_ready(self, e):
        self._p()
        self._p(self.c("bold", f"proposed change to {e.path}:"))
        for line in e.diff.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                self._p(self.c("dim", line))
            elif line.startswith("+"):
                self._p(self.c("green", line))
            elif line.startswith("-"):
                self._p(self.c("red", line))
            elif line.startswith("@@"):
                self._p(self.c("cyan", line))
            else:
                self._p(line)
        self._p()

    def _on_gate_decision(self, e):
        verdict = self.c("green" if e.critic_verdict == "APPROVE" else "red", e.critic_verdict)
        colour = "green" if not e.needs_human else "yellow"
        self._p(f"{self.c('bold', 'GATE')}    self={e.self_confidence:.2f}  "
                f"critic={verdict}  -> {self.c(colour, f'{e.score:.2f}')} "
                f"{self.c('dim', f'(threshold {e.threshold})')}")
        self._p(self.c("dim", f"        critic: {e.critic_reason}"))

        for warning in e.structural_warnings:
            self._p(self.c("red", f"        ! {warning}"))
        if e.blocked_structurally:
            self._p(self.c("red", "        BLOCKED by structural check "
                                  "-- this needs a human regardless of score"))

    def _on_approval_resolved(self, e):
        if e.decision == ev.Decision.REJECT.value:
            self._p(self.c("red", f"REJECTED  {e.reason}"))
        elif e.automatic:
            self._p(self.c("dim", f"applied automatically ({e.reason})"))
        else:
            self._p(self.c("green", "APPLIED"))

    def _on_observation(self, e):
        text = e.text if len(e.text) <= 800 else e.text[:800] + "\n[...truncated for display]"
        indented = "\n".join(f"        {ln}" for ln in text.splitlines())
        self._p(f"{self.c('dim', 'RESULT')}")
        self._p(self.c("dim", indented))

    def _on_protocol_violation(self, e):
        self._p(self.c("yellow", "the model broke protocol; re-prompting it"))
        if self.verbose:
            self._p(self.c("dim", f"        got: {e.raw[:200].strip()!r}"))

    # -- outcome --

    def _on_final(self, e):
        self._p()
        self._p(self.c("green", self.c("bold", "DONE")) +
                self.c("dim", f"  {e.steps_used} step(s), {e.elapsed:.1f}s"))
        # An answer given as prose was streamed in full as it arrived, so
        # printing the summary would show the same three paragraphs twice.
        # A normal FINAL is a one-line report of work done and is worth
        # repeating; this is the answer itself.
        if not self._already_streamed(e.summary):
            self._p(e.summary)
        self._p()
        if e.files_changed:
            self._p(self.c("dim", "files changed: " + ", ".join(dict.fromkeys(e.files_changed))))
        elif getattr(self, "_mode", "edit") == "answer":
            # Changing nothing is the whole point here, so the "no files were
            # changed" warning would report success as if it were a caveat.
            # The hint covers the one way this routing can be wrong: a change
            # request phrased as a question ("why not return [] instead?").
            # Answering it is the safe error, and this is how the user
            # un-makes it without having to know a mode exists.
            self._p(self.c("dim", "answered without changing anything -- "
                                  "say it as an instruction if you want it applied"))
        else:
            # Small models will happily report a job well done having written
            # nothing at all -- sometimes on the very first step. The summary
            # is the model's claim; this line is the fact.
            self._p(self.c("yellow", "no files were changed -- the summary above "
                                     "is the model's claim, not a result"))
        self._p()

    def _on_run_failed(self, e):
        self._p()
        self._p(self.c("red", f"FAILED at step {e.step}: {e.error}"))
        self._p()

    def _on_step_limit_reached(self, e):
        self._p()
        self._p(self.c("yellow",
                       f"stopped: hit the {e.max_steps}-step limit without a FINAL. "
                       "The task may be too big, or the model may be looping."))
        self._p()


def _summarize(args: dict) -> str:
    """Print args without dumping an entire rewritten file into the terminal."""
    shown = {
        k: (v if not isinstance(v, str) or len(v) <= 60 else f"{v[:60]}... [{len(v)} chars]")
        for k, v in args.items()
    }
    return ", ".join(f"{k}={v!r}" for k, v in shown.items())


# --- asking the human --------------------------------------------------------

def prompt_for_approval(event: ev.ApprovalNeeded, renderer: TerminalRenderer):
    """
    Three answers, not two.

    'reject with a reason' is the one that matters: the reason is fed back to
    the model as its next observation, so a refused patch becomes a correction
    rather than a dead end.
    """
    c = renderer.c
    renderer._p()
    headline = "this patch was BLOCKED" if event.blocked_structurally else "this patch needs your call"
    renderer._p(c("yellow", f"{event.path}: {headline}"))
    for reason in event.reasons or ["below the confidence threshold"]:
        renderer._p(c("yellow", f"  - {reason}"))
    renderer._p(c("dim", "  [y] apply   [n] reject (you'll be asked why)   "
                         "[a] apply and stop asking this session"))

    try:
        answer = input("  apply this patch? [y/N/a] ").strip().lower()
    except EOFError:
        # Non-interactive stdin: fail closed rather than writing to disk.
        renderer._p(c("dim", "  (no tty; treating as reject)"))
        return ev.Decision.REJECT, "not running interactively"

    if answer == "a":
        return ev.Decision.ALWAYS, "user approved for the session"
    if answer == "y":
        return ev.Decision.APPLY, "user approved"

    try:
        why = input("  what's wrong with it? (enter to skip) ").strip()
    except EOFError:
        why = ""
    return ev.Decision.REJECT, why


# --- the driver --------------------------------------------------------------

def drive(generator, renderer: TerminalRenderer, on_approval=None, sink=None):
    """
    Pump the agent generator, rendering events and answering approvals.

    The send()-based protocol is a little fiddly, so it lives here once instead
    of at every call site: everything yielded gets rendered, and an
    ApprovalNeeded gets a Decision sent back in.
    """
    on_approval = on_approval or (lambda e: prompt_for_approval(e, renderer))
    reply = None

    while True:
        try:
            event = generator.send(reply)
        except StopIteration:
            return
        reply = None

        if sink is not None and event.kind not in ev.TRANSIENT:
            sink.append(event)

        if isinstance(event, ev.ApprovalNeeded):
            renderer.render(event)
            reply = on_approval(event)
            continue

        renderer.render(event)


def run_to_terminal(task: str, repo_root: str, backend: str | None = None,
                    auto_approve: bool = False, stream: bool = True,
                    out=None, save_transcript: bool = True,
                    transcript_path: str | None = None,
                    mode: str = "edit") -> int:
    """One-shot convenience wrapper: run a task, print it, return an exit code."""
    from . import agent, transcript

    renderer = TerminalRenderer(out=out)
    collected: list[ev.Event] = []

    # Streaming is only worth it on a slow local model; the mock is instant and
    # the cloud path returns one blob, so tokens would just duplicate output.
    on_token = renderer.on_token if (stream and backend == "ollama") else None

    generator = agent.run(task=task, repo_root=repo_root, backend=backend,
                          auto_approve=auto_approve, on_token=on_token,
                          mode=mode)
    drive(generator, renderer, sink=collected)

    if save_transcript and collected:
        path = transcript_path or transcript.new_path(task)
        try:
            transcript.save(path, collected,
                            meta={"task": task, "repo": repo_root, "backend": backend})
            renderer._p(renderer.c("dim", f"transcript: {path}"))
        except OSError as e:
            # Losing the recording must never fail the run that produced it.
            renderer._p(renderer.c("dim", f"(could not write transcript: {e})"))

    failed = any(e.kind in (ev.RunFailed.kind, ev.StepLimitReached.kind) for e in collected)
    return 1 if failed else 0
