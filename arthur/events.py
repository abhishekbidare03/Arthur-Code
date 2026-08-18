"""
The vocabulary the agent loop speaks in.

`run_agent` used to drive the loop *and* print it, which meant nothing else
could consume a run. Now it yields these instead, and three consumers share one
implementation:

  - the terminal renderer, which prints them as they arrive
  - the transcript, which serializes them to JSON
  - the tests, which assert on the sequence instead of scraping stdout

Replay then costs nothing extra: a saved run is just a list of these events fed
back through the same renderer.

One event needs a reply rather than just being displayed -- ApprovalNeeded.
The loop yields it and waits for the consumer to `.send()` back a Decision.
That is what keeps human approval out of the agent and in the UI, so the same
loop can run interactively, headless, or under a test.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import time


@dataclass
class Event:
    """Base class. `kind` is what the JSON transcript keys off."""
    kind: str = field(init=False, default="event")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind
        return d


# --- run lifecycle -----------------------------------------------------------

@dataclass
class RunStarted(Event):
    kind = "run_started"
    task: str = ""
    repo_root: str = ""
    backend: str = ""
    model: str = ""
    started_at: float = field(default_factory=time.time)
    # "edit" or "answer". Carried on the event rather than passed to the
    # renderer separately so that a replayed transcript renders identically --
    # the mode changes what the closing lines are allowed to say.
    mode: str = "edit"


@dataclass
class IndexBuilt(Event):
    kind = "index_built"
    file_count: int = 0
    tree: str = ""


@dataclass
class ContextRetrieved(Event):
    kind = "context_retrieved"
    files: list[dict] = field(default_factory=list)   # [{path, score, chars}]
    dropped: int = 0                                  # trimmed away by the budget


# --- per-step ----------------------------------------------------------------

@dataclass
class StepStarted(Event):
    kind = "step_started"
    step: int = 0
    max_steps: int = 0


@dataclass
class TokenChunk(Event):
    """Streamed model output. Not written to the transcript -- the assembled
    text arrives in Thought/ActionProposed anyway, and replaying a run
    token-by-token is theatre, not information."""
    kind = "token"
    text: str = ""


@dataclass
class Thought(Event):
    kind = "thought"
    text: str = ""


@dataclass
class ActionProposed(Event):
    kind = "action_proposed"
    action: str = ""
    args: dict = field(default_factory=dict)
    confidence: float | None = None
    # The unparsed turn, kept ONLY when the arguments came back empty.
    #
    # Every parser bug in Phase 9 had to be reproduced live before it could be
    # fixed, because a transcript records what survived parsing and these bugs
    # are defined by what did not. `insert_after` with `args: {}` appears in the
    # transcript identically whether the model sent nothing or sent a perfect
    # call the parser could not read -- and it was the second one, four times.
    # Empty args is exactly the case where the raw text is the only evidence,
    # and it is rare enough that keeping it costs nothing.
    raw: str = ""


@dataclass
class DiffReady(Event):
    kind = "diff_ready"
    path: str = ""
    diff: str = ""


@dataclass
class GateDecision(Event):
    kind = "gate_decision"
    self_confidence: float = 0.0
    critic_verdict: str = ""
    critic_reason: str = ""
    score: float = 0.0
    threshold: float = 0.0
    needs_human: bool = False
    # Deterministic checks (symbols removed, file truncated, syntax broken).
    # A structural block outranks the score: two models agreeing that deleting
    # half a file is fine is still two models being wrong.
    structural_warnings: list[str] = field(default_factory=list)
    blocked_structurally: bool = False


class Decision(str, Enum):
    APPLY = "apply"
    REJECT = "reject"
    ALWAYS = "always"    # apply, and stop asking for the rest of the session


@dataclass
class ApprovalNeeded(Event):
    """The one event that expects a reply: send() back a Decision, and
    optionally a reason when rejecting."""
    kind = "approval_needed"
    path: str = ""
    diff: str = ""
    score: float = 0.0
    threshold: float = 0.0
    # Why we're asking. A patch can be stopped by a low score OR by the
    # structural check while scoring well -- telling the user "0.75 is below
    # 0.65" in the second case is just wrong, so carry the real reasons.
    reasons: list[str] = field(default_factory=list)
    blocked_structurally: bool = False


@dataclass
class ApprovalResolved(Event):
    kind = "approval_resolved"
    decision: str = ""
    reason: str = ""
    automatic: bool = False   # True when auto-approved by the gate or /auto


@dataclass
class Observation(Event):
    kind = "observation"
    text: str = ""
    action: str = ""


@dataclass
class ProtocolViolation(Event):
    """The model broke format. Worth its own event: the rate of these is the
    single most useful number for judging whether a local model is usable."""
    kind = "protocol_violation"
    step: int = 0
    raw: str = ""
    detail: str = ""


# --- run outcome -------------------------------------------------------------

@dataclass
class Final(Event):
    kind = "final"
    summary: str = ""
    steps_used: int = 0
    files_changed: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class RunFailed(Event):
    kind = "run_failed"
    error: str = ""
    step: int = 0


@dataclass
class StepLimitReached(Event):
    kind = "step_limit_reached"
    max_steps: int = 0


# --- serialization -----------------------------------------------------------

_REGISTRY: dict[str, Any] = {
    cls.kind: cls for cls in (
        RunStarted, IndexBuilt, ContextRetrieved, StepStarted, Thought,
        ActionProposed, DiffReady, GateDecision, ApprovalNeeded,
        ApprovalResolved, Observation, ProtocolViolation, Final, RunFailed,
        StepLimitReached,
    )
}

# Live-only events: meaningful while a run is happening, noise in a transcript.
TRANSIENT = {TokenChunk.kind, ApprovalNeeded.kind}


def from_dict(d: dict) -> Event:
    """Rebuild an event from its JSON form, for --replay."""
    kind = d.get("kind")
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown event kind: {kind!r}")
    payload = {k: v for k, v in d.items() if k != "kind"}
    return cls(**payload)
