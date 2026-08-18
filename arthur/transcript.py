"""
Recording a run, and playing it back.

The payoff for making the loop an event generator. A transcript is just the
event list serialized; `--replay` feeds it back through the same renderer that
drew it live. Nothing is re-run, no model is called, and the output is
identical to the original -- which is what makes it worth showing someone.

Deliberately not a log file. A log is prose about what happened; this is the
actual structured record, so it can be diffed, graded, or replayed.
"""

import json
import os
import time
from datetime import datetime

from . import config, events as ev


def default_dir() -> str:
    return os.environ.get("ARTHUR_TRANSCRIPT_DIR",
                          os.path.join(os.path.expanduser("~"), ".arthur", "runs"))


def new_path(task: str, directory: str | None = None) -> str:
    """A timestamped, human-scannable filename with a slug of the task in it."""
    directory = directory or default_dir()
    slug = "".join(c if c.isalnum() else "-" for c in task.lower())[:40].strip("-")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(directory, f"{stamp}-{slug or 'run'}.json")


def save(path: str, collected: list[ev.Event], meta: dict | None = None) -> str:
    """Write a run to disk. Returns the path actually written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "arthur_version": __import__("arthur").__version__,
        "saved_at": time.time(),
        "meta": meta or {},
        "events": [e.to_dict() for e in collected],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load(path: str) -> tuple[list[ev.Event], dict]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict) or "events" not in payload:
        raise ValueError(f"{path} is not an arthur transcript")

    restored, skipped = [], 0
    for raw in payload["events"]:
        try:
            restored.append(ev.from_dict(raw))
        except (ValueError, TypeError):
            # A transcript written by a newer version may contain event kinds
            # or fields this build doesn't know. Replay the rest rather than
            # refusing the whole file -- a partial replay beats no replay.
            skipped += 1

    meta = dict(payload.get("meta") or {})
    meta["_skipped_events"] = skipped
    meta["_arthur_version"] = payload.get("arthur_version", "unknown")
    return restored, meta


def replay(path: str, renderer, delay: float = 0.0) -> int:
    """
    Render a saved run. No model, no network, no filesystem writes.

    `delay` paces the output for a live demo; the default replays instantly.
    """
    restored, meta = load(path)

    c = renderer.c
    print(c("dim", f"replaying {os.path.basename(path)} "
                   f"({len(restored)} events, arthur {meta['_arthur_version']})"))
    if meta.get("_skipped_events"):
        print(c("yellow", f"  {meta['_skipped_events']} event(s) from a newer "
                          "version were skipped"))

    for event in restored:
        renderer.render(event)
        if delay:
            time.sleep(delay)

    failed = any(e.kind in (ev.RunFailed.kind, ev.StepLimitReached.kind)
                 for e in restored)
    return 1 if failed else 0


def list_runs(directory: str | None = None, limit: int = 20) -> list[tuple[str, str, str]]:
    """(path, when, task) for the most recent runs, newest first."""
    directory = directory or default_dir()
    if not os.path.isdir(directory):
        return []

    rows = []
    for name in sorted(os.listdir(directory), reverse=True)[:limit]:
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        task = "(unreadable)"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            task = (payload.get("meta") or {}).get("task", "(no task recorded)")
        except (OSError, json.JSONDecodeError):
            pass
        when = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        rows.append((path, when, task))
    return rows
