"""
Central configuration for the mini code agent.

Backends:
  "mock"   - scripted, offline, zero deps. Proves the loop works.
  "ollama" - local model via `ollama serve`. The offline-first path.
  "gemini" - Google Gemini API. The cloud path, for a reliable demo without
             fighting a 3B model on limited VRAM.

The API key is read from the environment (GEMINI_API_KEY, or GOOGLE_API_KEY
as a fallback) -- never hardcode it here, this file is committed.
A `.env` file next to this one is loaded automatically if present.
"""

import os

# --- .env loader (tiny, no python-dotenv dependency) -------------------------


def _dotenv_candidates() -> list[str]:
    """
    Where to look for a .env, nearest-wins-last so later files override earlier.

    `arthur` is installed on PATH and run from arbitrary directories, so unlike
    the original script there is no single "next to this file" answer:
      1. ~/.arthur/.env      -- your machine-wide key, set once
      2. <project root>/.env -- this checkout, one level above the package
      3. ./.env              -- the repo you're currently standing in
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(package_dir)
    return [
        os.path.join(os.path.expanduser("~"), ".arthur", ".env"),
        os.path.join(project_root, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]


def _load_dotenv() -> None:
    seen: set[str] = set()
    for path in _dotenv_candidates():
        real = os.path.normcase(os.path.abspath(path))
        if real in seen or not os.path.exists(path):
            continue
        seen.add(real)
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                # .env WINS over an existing environment variable. Deliberate:
                # a stale GEMINI_API_KEY left in the Windows user environment
                # otherwise silently shadows the project's key, and you get an
                # "API key not valid" 400 from a key you never intended to send.
                os.environ[key] = value


_load_dotenv()


# --- Backend selection -------------------------------------------------------

# "mock" | "ollama" | "gemini"   (override at runtime with `--backend`)
#
# Defaults to the local model. "mock" replays a fixed script about the bundled
# demo repo no matter what you type, which is invaluable for testing the loop
# and thoroughly baffling as a default -- you ask for one thing and watch it
# confidently do another.
BACKEND = os.environ.get("AGENT_BACKEND", "ollama")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Chosen by measurement, not reputation -- see `python eval.py`. On the 3-task
# acceptance set, on a 4GB card:
#
#   phi4-mini           3/3   4.9s/turn   2.5GB   <- default
#   qwen2.5-coder:3b    1/3  13.1s/turn   1.9GB
#   qwen3:4b            1/3  ~130s/turn   2.5GB   (thinking; unusable here)
#
# The surprise is that the *code-tuned* model loses. These tasks need trivial
# code and rigorous format-following, and phi4-mini is much better at the
# second -- which is the actual bottleneck for an agent loop.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi4-mini")

# The critic only reads a task + a diff and votes APPROVE/REJECT with one line
# of reasoning. That is classification, not generation, and a 1.5B does it
# competently -- which keeps the gate cheap enough to run on every patch and
# makes the second opinion genuinely independent rather than the same weights
# re-scoring their own work.
OLLAMA_CRITIC_MODEL = os.environ.get("OLLAMA_CRITIC_MODEL", "qwen2.5-coder:1.5b")

# --- Ollama generation options ----------------------------------------------
#
# These are not tuning knobs, they are the difference between "the 3B follows
# the protocol" and "the 3B is useless". Read the comments before changing any
# of them.

# THE important one. Ollama's default context is small (2048 on many builds)
# and when you exceed it, it truncates *from the front* -- silently. The front
# of our conversation is the system prompt, which is where the entire protocol
# definition and the repo context live. Blow past the default and the model
# stops being an agent and starts free-associating, with no error anywhere.
# 8192 fits comfortably next to a 3B on a 4GB card. Raise it and you start
# trading VRAM that the weights need.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

# Cap on a single turn's output. One protocol turn is a thought plus a JSON
# blob that may contain a whole rewritten file; 1536 is generous for that and
# stops a looping model from generating until the timeout.
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1536"))

# Small chat-tuned models love to keep the conversation going by themselves:
# they emit an action and then cheerfully write the tool's OBSERVATION: and a
# follow-up turn, all hallucinated. Cutting generation at those tokens keeps
# one turn to one action, which is what the loop assumes.
#
# "\nTHOUGHT:" catches the commonest version: the model emits its action, then
# immediately writes a *second* THOUGHT and a FINAL announcing the success of
# a tool call that has not run yet. The leading newline means the turn's own
# opening THOUGHT: (at position 0) is unaffected.
OLLAMA_STOP = ["OBSERVATION:", "\nUSER:", "\nHUMAN:", "\nTHOUGHT:"]

# We want protocol compliance, not creativity.
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))

# How long Ollama keeps the model resident after a request. The default is 5
# minutes, but every step of the loop is a separate request, and an eviction
# mid-run means paying a multi-second reload on the next step.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# First token on a cold model can take a while (weights load from disk); after
# that a 3B is fast. Generous enough for a cold start, short enough to fail.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

# Reasoning models (qwen3, deepseek-r1, ...) think before answering. On a rigid
# THOUGHT/ACTION/ARGS protocol that buys almost nothing -- the format IS the
# reasoning scaffold -- while costing a large multiple in latency, and risking
# the scratchpad eating the whole num_predict budget before any protocol output
# appears. Qwen3 honours a literal "/no_think" marker in the prompt; Ollama's
# newer `think` request field covers the rest.
OLLAMA_DISABLE_THINKING = os.environ.get("OLLAMA_NO_THINK", "1") not in ("0", "false", "")

# Models that need the marker rather than (or as well as) the request field.
THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "magistral")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TEMPERATURE = 0.2          # low: we want protocol compliance, not creativity
GEMINI_MAX_OUTPUT_TOKENS = 2048
# 2.5-series models think before answering. Thinking costs latency and tokens
# and buys little on a rigid THOUGHT/ACTION/ARGS protocol, so it's off by
# default. Set to -1 for dynamic thinking, or a token budget like 1024.
GEMINI_THINKING_BUDGET = 0
GEMINI_TIMEOUT = 120
GEMINI_MAX_RETRIES = 3            # free-tier 429s and transient 503s are common


# --- Agent loop --------------------------------------------------------------

# Hard cap on agent loop iterations. Twelve was picked when a step was cheap;
# on a 4GB card at ~11 tok/s it is nearly two minutes of watching a model that
# decided at step 2 what it was going to do. Every task this agent completes
# takes one to four steps, and runs that reach double figures have never once
# recovered -- they were failing, slowly. Cap where the useful work stops.
MAX_STEPS = 8

# How many identical replies in a row count as "stuck". Local models run near
# deterministic, so a turn they got wrong they will get wrong again verbatim;
# without this the loop cheerfully burns every remaining step proving it.
REPEAT_LIMIT = 3

# When an edit_file FIND block fails to match, echo the file back to the model
# up to this size so it can copy the real text. Bounded because the observation
# goes into the context window, and a large file would crowd out the protocol.
EDIT_ECHO_MAX_CHARS = 3000

CONFIDENCE_THRESHOLD = 0.65      # below this -> pause and ask the human before writing to disk

# Fraction of a file a single patch may delete before the gate blocks it
# outright, regardless of what the coder and critic think. Small models asked
# for a whole rewritten file sometimes just stop writing partway through, and
# the result is a valid-looking patch that silently truncates the file.
MAX_SHRINK_RATIO = 0.35
MAX_CONTEXT_FILES = 6            # how many files the retriever injects as context
MAX_FILE_CHARS = 4000            # truncate large files before injecting into the prompt

# --- Context budgeting -------------------------------------------------------
#
# MAX_CONTEXT_FILES * MAX_FILE_CHARS is 24000 chars ~= 6000 tokens, which on
# its own nearly fills an 8k window before the model has said anything. Rather
# than discover that as mysterious truncation, we measure and trim explicitly.

# Chars per token. Code is denser than prose (punctuation and short
# identifiers tokenize badly), so 3.5 is a deliberately pessimistic estimate --
# overestimating the token count is safe, underestimating is what silently
# truncates the system prompt.
CHARS_PER_TOKEN = 3.5

# Fraction of the window the system prompt (repo tree + retrieved files) may
# occupy. The rest is left for the conversation: observations, file contents
# read back by the model, and its own replies.
CONTEXT_PROMPT_BUDGET = 0.45

# Once the conversation exceeds this fraction of the window, the oldest
# observations are elided from the middle of the history. The system prompt and
# the most recent turns are always preserved -- those are what the model
# actually needs to act correctly.
CONTEXT_HISTORY_BUDGET = 0.80

IGNORE_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", ".mypy_cache"}
IGNORE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".lock", ".ico", ".pyc"}

# Tools that mutate state / the filesystem. These are the ones gated by the
# confidence + critic check before they're allowed to execute.
DESTRUCTIVE_TOOLS = {"edit_file", "append_file", "apply_patch", "run_command"}
