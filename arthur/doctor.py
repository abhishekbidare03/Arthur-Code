"""
`arthur doctor` -- the pre-flight check.

Two seconds here saves you from a run that dies halfway through a task because
of a wrong model name, a stopped Ollama daemon, or a stale API key. It answers
three questions in order:

  1. Is the backend reachable at all?
  2. Is the model I asked for actually installed?
  3. Does it speak the protocol, or does it answer in prose?

(3) matters as much as (1). A model that replies "Sure! I'd be happy to read
that file for you." is reachable and useless -- it will burn every step of the
loop being re-prompted. Better to find that out now.
"""

import shutil
import subprocess
import time

from . import config


# --- environment probes ------------------------------------------------------

def _gpu_info() -> str | None:
    """Free/total VRAM via nvidia-smi, since that number decides the model."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return line
    name, total, used = parts[0], int(parts[1]), int(parts[2])
    return f"{name} -- {total - used} MiB free of {total} MiB"


def _ollama_models() -> tuple[bool, list[str], str | None]:
    """(daemon_up, installed_model_names, error)."""
    try:
        from .llm_backend import OllamaBackend
        return True, OllamaBackend().installed_models(), None
    except ImportError:
        return False, [], "the `requests` package is not installed"
    except Exception as e:
        return False, [], str(e)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _offer_pull(missing: list[str]) -> bool:
    """
    Ask before downloading gigabytes, then show progress while doing it.

    A missing model is the single most common reason a first local run fails,
    and telling someone to go run a different command in another terminal is a
    worse answer than just offering to do it.
    """
    print(f"\nMissing:  {', '.join(missing)}")
    try:
        answer = input(f"Pull {len(missing)} model(s) now? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("", "y", "yes"):
        print("Skipped. Pull them yourself with:")
        for m in missing:
            print(f"    ollama pull {m}")
        return False

    from .llm_backend import OllamaBackend
    backend = OllamaBackend()
    for model in missing:
        print(f"\npulling {model}")
        last = ""
        try:
            for status, done, total in backend.pull(model):
                if total:
                    pct = 100 * done / total
                    bar = "#" * int(pct / 4)
                    line = f"  [{bar:<25}] {pct:5.1f}%  {_fmt_bytes(done)}/{_fmt_bytes(total)}"
                elif status != last:
                    line = f"  {status}"
                else:
                    continue
                last = status
                print(line.ljust(70), end="\r", flush=True)
        except Exception as e:
            print(f"\n  FAILED: {e}")
            return False
        print(f"\n  done: {model}".ljust(70))
    return True


# --- protocol probe ----------------------------------------------------------

PROBE = [
    {"role": "system", "content": (
        "You are a coding agent. Respond using EXACTLY this protocol and nothing else:\n"
        "THOUGHT: <reasoning>\nACTION: <tool name>\nARGS: <one-line JSON>"
    )},
    {"role": "user", "content": "Read the file calculator.py. The only tool is read_file(path)."},
]


def _probe_protocol(backend_name: str) -> int:
    from .agent import parse_response
    from .llm_backend import get_backend

    try:
        backend = get_backend(backend_name)
        start = time.time()
        response = backend.chat(PROBE)
        elapsed = time.time() - start
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Rough throughput, which is the number that decides whether a 12-step
    # loop is bearable on this machine.
    tokens = max(1, int(len(response.text) / 3.5))
    rate = f"  (~{tokens / elapsed:.0f} tok/s)" if elapsed > 0.05 else ""
    print(f"  round-trip: {elapsed:.2f}s{rate}")
    print("  --- raw response ---")
    for line in response.text.strip().splitlines():
        print(f"  | {line}")
    print("  --- end ---")

    parsed = parse_response(response.text)
    print(f"  parsed action: {parsed['action']!r}  args: {parsed['args']!r}")
    if parsed["action"] == "read_file":
        print("  OK: reachable and protocol-compliant.")
        return 0
    print("  WARN: reachable, but it did not follow the protocol cleanly.")
    print("        Expect wasted steps. Try a coder-tuned model.")
    return 0


# --- entry point -------------------------------------------------------------

def run(backend_name: str | None = None) -> int:
    backend_name = (backend_name or config.BACKEND).lower()
    model = {
        "gemini": config.GEMINI_MODEL,
        "ollama": config.OLLAMA_MODEL,
    }.get(backend_name, "scripted")

    print("arthur doctor\n")

    gpu = _gpu_info()
    print(f"GPU:      {gpu or 'no NVIDIA GPU detected (CPU inference will be slow)'}")
    print(f"Backend:  {backend_name}")
    print(f"Model:    {model}\n")

    if backend_name == "ollama":
        up, installed, err = _ollama_models()
        if not up:
            print(f"Ollama:   UNREACHABLE at {config.OLLAMA_HOST}\n          {err}")
            print("\nFAIL: start the daemon with `ollama serve`, then re-run.")
            return 1
        print(f"Ollama:   up at {config.OLLAMA_HOST}")
        print(f"Installed: {', '.join(installed) if installed else '(none)'}")

        print(f"Context:  {config.OLLAMA_NUM_CTX} tokens "
              f"(num_ctx is set explicitly -- unset, Ollama truncates the "
              f"system prompt away without telling you)")

        # Ollama stores an untagged name as "<name>:latest", so a bare
        # `phi4-mini` never matches the tag list literally and a perfectly
        # installed model is reported missing.
        def tagged(name: str) -> str:
            return name if ":" in name else f"{name}:latest"

        have = {tagged(m) for m in installed}
        wanted = dict.fromkeys([config.OLLAMA_MODEL, config.OLLAMA_CRITIC_MODEL])
        missing = [m for m in wanted if tagged(m) not in have]
        if missing:
            if not _offer_pull(missing):
                return 1

    if backend_name == "gemini":
        key = config.GEMINI_API_KEY
        print(f"API key:  {'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")
        if not key:
            print("\nFAIL: put GEMINI_API_KEY=... in a .env file "
                  "(project root, or ~/.arthur/.env)")
            return 1

    print("\nProtocol probe:")
    return _probe_protocol(backend_name)
