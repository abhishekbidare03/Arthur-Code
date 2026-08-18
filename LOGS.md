# Build log — Arthur CLI

A running record of what was built, why, and what was learned. Written to give
a future session (or a future me) enough context to pick this up cold.

**Last updated:** 2026-08-17, end of Phase 9.
**State:** All planned phases complete, plus four rounds of fixes driven by
real use. **431 tests passing.** Verified live on mock, ollama (`phi4-mini`,
`qwen2.5-coder:3b`), and gemini (`gemini-2.5-flash`). `arthur` runs an
interactive session, records every run, and replays them.

Phases 7–9 came from actually using it rather than from the plan, and they are
where the interesting failures are. If you are picking this up cold, read
**Phase 8** first — it is the difference between a demo and something that can
take a follow-up instruction — then **Phase 9**, which is about the agent
having only one gear.

---

## Hardware this is built for

Everything about the model choices below follows from this:

| | |
|---|---|
| GPU | NVIDIA GTX 1650 Ti, **4096 MiB VRAM** |
| CPU | Intel i5-10300H @ 2.50GHz |
| RAM | 16 GB |
| OS | Windows 11, PowerShell + Git Bash |
| Python | 3.12.3 (Anaconda, `D:\Anaconda`) |

4GB is the binding constraint. It is why the default coder is a 3B and not a 7B.

---

## Where things stand

```
E:\mini-code-agent\Arthur-CLI\
├── pyproject.toml          console script: arthur = "arthur.cli:main"
├── arthur/
│   ├── cli.py              the `arthur` command + subcommands
│   ├── intent.py           is this a task, a question, or just "hey"?
│   ├── session.py          the interactive session, slash commands, @mentions
│   ├── doctor.py           pre-flight + auto-pull
│   ├── agent.py            the loop (a generator), protocol parser, prompt
│   ├── events.py           typed event vocabulary
│   ├── render.py           the ONLY module that prints
│   ├── transcript.py       save / load / replay a run
│   ├── context.py          token budgeting
│   ├── safety_gate.py      self-confidence + critic + structural veto
│   ├── patcher.py          unified diff + ast structural analysis
│   ├── tools.py            the 8 tools, all path-contained
│   ├── llm_backend.py      Mock / Ollama / Gemini
│   ├── indexer.py          file tree + Python symbols
│   ├── retriever.py        filename mentions first, keyword overlap second
│   └── config.py           all knobs + .env loader
├── tests/                  431 tests, 19 files
├── eval.py                 the 4-task acceptance harness (grades by executing)
├── demo_repo/              10-line toy fixture
├── main.py                 back-compat shim for `python main.py`
├── PLAN.md                 the original plan (unchanged)
├── README.md               user-facing docs + findings
└── LOGS.md                 this file
```

Runs are written to `~/.arthur/runs/`; prompt history to `~/.arthur/history`.

Install: `pip install -e .` (already done; `arthur` is at `/d/Anaconda/Scripts/arthur`).

---

## Phase 1 — package + entry point

**Goal:** turn a pile of flat scripts into an installable `arthur` command.

### What was done

- Created the `arthur/` package; moved `config, llm_backend, indexer,
  retriever, tools, patcher, safety_gate, agent` into it and converted every
  import to relative (`from . import config`).
- `check_backend.py` → `arthur/doctor.py`, rewritten as a callable `run()`.
- `test_parser.py` → `tests/`, renamed functions to `test_*` so pytest
  collects them (the `@case` decorator kept the standalone runner working).
- `pyproject.toml` with `[project.scripts] arthur = "arthur.cli:main"`.
- `main.py` reduced to a shim mapping the old `--repo/--task` onto the new flags.

### Key design decision

**The target repo defaults to the current directory.** `cd` into a project and
`arthur` is already pointed at the right code, the way `git` is. This is the
whole ergonomic point of making it a real command.

### Three real bugs found

1. **`--model` was a silent no-op.** `OllamaBackend.__init__(self, model=config.OLLAMA_MODEL)`
   — a default argument is evaluated *once at import*, so mutating
   `config.OLLAMA_MODEL` at runtime never reached the backend. Now resolved in
   the body. Regression test: `test_model_override_reaches_ollama_backend`.
2. **`.env` would have gone missing.** The loader looked next to `config.py`,
   which after the move is `arthur/.env`. Since `arthur` runs from arbitrary
   directories there is no single "next to this file" answer — it now checks
   `~/.arthur/.env` → project root → cwd, nearest wins.
3. **The default model didn't fit the card.** `config.py` shipped
   `qwen2.5-coder:7b` (~4.7GB weights vs 4096 MiB VRAM).

---

## Phase 2 — hardening the Ollama backend

**Goal:** make a local model actually usable, not just wired up.

### Model tiers (`config.py`)

| role | model | ~size Q4 | rationale |
|---|---|---|---|
| coder (default) | `qwen2.5-coder:3b` | 1.9 GB | fits fully in 4GB with an 8k KV cache; ~20 tok/s |
| coder (opt-in) | `qwen2.5-coder:7b` | 4.7 GB | better, but spills to CPU here (~5 tok/s). **Not pulled.** |
| critic | `qwen2.5-coder:1.5b` | 1.0 GB | judging a diff is classification, not generation |

Using a *different, smaller* model for the critic is deliberate: it keeps the
gate cheap and makes the second opinion genuinely independent weights rather
than the same model re-scoring its own work. `get_critic_backend()` routes to it.

Also installed but unused: `llama3.2:3b`, `qwen3:4b`, `qwen2.5:1.5b`.
`qwen3:4b` is avoided as a default — it emits `<think>` blocks that fight a
rigid text protocol.

### The settings that matter (all in `config.py`)

These have no visible effect until they're wrong, and then they fail as "the
model is stupid" rather than "the client is misconfigured":

- **`OLLAMA_NUM_CTX = 8192`** — *the* important one. Ollama's default is small
  and it truncates **from the front**, silently. The front is the system
  prompt: the protocol definition and repo context. Overflow it and the model
  stops being an agent, with no error anywhere.
- **`OLLAMA_STOP`** = `["OBSERVATION:", "\nUSER:", "\nHUMAN:", "\nTHOUGHT:"]` —
  small models write the tool's output themselves and keep going solo. The
  `\nTHOUGHT:` entry was added in Phase 3 (see finding #4).
- **`OLLAMA_TEMPERATURE = 0.1`**, `TOP_P = 0.9` — compliance, not creativity.
- **`OLLAMA_KEEP_ALIVE = "30m"`** — every loop step is a separate request.
  Measured: cold 8.55s → warm 2.99s.
- **`OLLAMA_NUM_PREDICT = 1536`**, `OLLAMA_TIMEOUT = 180`.

### Other Phase 2 work

- **Streaming** via NDJSON (`_stream`), surviving partial lines.
- **`strip_thinking()`** handles both closed and *unclosed* `<think>` blocks —
  the unclosed case matters because generation cut short by a stop token never
  emits the closing tag.
- **Actionable errors**: 404 → `ollama pull <model>`; ConnectionError →
  `ollama serve`; Timeout → raise `OLLAMA_TIMEOUT` or use a smaller model.
- **`arthur doctor`**: GPU + free VRAM via `nvidia-smi`, daemon reachability,
  installed vs wanted models, **offers to pull** with a progress bar
  (`OllamaBackend.pull()` streams `/api/pull`), then a protocol-compliance probe
  reporting round-trip and rough tok/s.
- **`context.py`** — deliberate token budgeting. Estimate is pessimistic on
  purpose (`CHARS_PER_TOKEN = 3.5`); overestimating wastes a little window,
  underestimating eats the system prompt. `trim_history()` preserves
  `messages[0]` (protocol), `messages[1]` (original task) and the most recent
  turns, eliding the middle with a visible marker.

---

## Phase 3 — the event refactor

**Goal:** stop the loop from printing, so other things can consume it.

`run_agent` used to drive the loop *and* write to stdout. Now `agent.run()` is
a **generator yielding typed events** and never prints:

```
RunStarted → IndexBuilt → ContextRetrieved
  → [StepStarted → Thought → ActionProposed
      → DiffReady → GateDecision → (ApprovalNeeded ⇄ Decision)
      → ApprovalResolved → Observation]*
  → Final | RunFailed | StepLimitReached
```

Three consumers share one implementation: the terminal renderer, the JSON
transcript (next phase), and the tests — which assert on the event sequence
instead of scraping stdout.

**Approval:** the loop *yields* `ApprovalNeeded` and the consumer `.send()`s a
`Decision` (APPLY / REJECT / ALWAYS). The loop never calls `input()`, so the
same code runs interactively, headless, and under test. **A consumer that
doesn't answer is read as rejection** — silence is not consent when the question
is "may I write to disk". `render.drive()` hides the send-protocol.

`render.py` owns all printing: hand-rolled ANSI (no `rich` dependency yet),
auto-disabled when not a tty. Streaming echoes the model's reasoning live then
**cuts at the first `ACTION:`/`FINAL:`** so the JSON payload isn't printed twice.

---

## What the live runs found

Five findings, none of which the mock backend could ever surface. All are
covered by tests now.

### 1. The model deleted code, and the critic approved it ← most important

Asked to add **one docstring** to a 25-line file, `qwen2.5-coder:3b` returned a
"patched" file with two top-level functions and the module docstring **missing**.
The 1.5B critic **approved** it — correctly observing the docstring had been
added, never noticing what was gone.

This is inherent to the whole-file-rewrite design: the choice that makes small
models usable (they can't emit valid unified diffs) is exactly what lets them
silently truncate. **An LLM reviewer cannot be relied on to catch it**, because
the patch does contain the thing it was asked to check for.

**Fix — the structural veto.** `patcher.analyze()` does an `ast`-level
comparison of top-level symbols (including methods, as `Class.method`) before
and after. `safety_gate` blocks outright — regardless of score — if the patch:
- removes existing definitions, or
- breaks Python parsing, or
- deletes more than `MAX_SHRINK_RATIO` (0.35) of the file.

Verified: self-confidence 0.99 + critic APPROVE (score 1.0) → **still blocked**.
`-y` / auto-approve **cannot** override it: "stop asking me about judgement
calls" is not "turn the safety system off". `test_auto_approve_cannot_bypass_a_structural_block`.

Only meaningful if the *original* parsed — we can't claim a symbol vanished
from a file we never understood. Non-Python falls back to line counts.

### 2. Path containment (security)

The model echoed back an **absolute path**. `os.path.join(root, path)` honours
those by discarding the root entirely; `../..` walks straight out. Added
`tools.resolve()` which refuses anything outside the repo, plus a `_safe`
decorator turning an escape into an observation (the model gets told off and
retries) rather than a crash. Guards against the shared-prefix case too
(`/repo-backup` must not pass a naive `startswith` against `/repo`).

### 3. JSON-embedded source doesn't work on a 3B

Across live runs the same model failed **differently every time**:
- Python triple quotes: `{"new_content": """<file>"""}`
- single-quoted value wrapping a bogus nested object
- unescaped literal newlines

Each fix is another regex and there is always another malformation. So
`apply_patch` now accepts a second, escape-free spelling:

```
ACTION: apply_patch
PATH: calculator.py
CONTENT:
```python
<the whole file, verbatim>
```
```

Small models produce fenced code blocks reliably. JSON args still work and
**take precedence when they parse**; this is an additional form, not a
replacement. Other tools keep JSON (their args are short and fine).

**Trap worth remembering:** the surviving triple-quote repair must match
**greedily**. The file being written usually contains triple quotes of its own
(any module with a docstring). A non-greedy match stops at that inner docstring
and truncates everything after — producing a plausible patch that deletes most
of the file. This was a bug *I introduced* and then caught live.

The repair also only runs **after** plain JSON parsing fails, so valid input is
never mangled.

### 4. A model will report success having done nothing

Two variants, both observed:
- `FINAL: ...fixed it` on step 1 with no tool call at all.
- **Worse:** emits a *correct patch*, then without waiting invents its own
  observation and a triumphant `FINAL` in the same turn. The parser honoured
  `FINAL` first → **silently discarded the patch and reported success**.

Fixes: whichever of `ACTION`/`FINAL` appears **first in the text** wins;
`\nTHOUGHT:` added to the stop list so it can't start a second turn; and the
run summary states plainly *"no files were changed — the summary above is the
model's claim, not a result"*.

### 5. Near-deterministic models repeat mistakes verbatim

At temperature 0.1 a wrong turn recurs **identically** until the step cap.
`REPEAT_LIMIT = 3` aborts at step 4 instead of burning 12.

---

## Measured results

### Phase 3 acceptance bar: 2/3 tasks unaided. **The 3B scores 0/3.**

Tasks (on `evalrepo/`, 25 lines, 6 functions): add a docstring, add a null
check, fix an off-by-one.

| backend | result |
|---|---|
| `qwen2.5-coder:3b` | **0/3** — every patch drops existing functions; gate blocks all |
| `gemini-2.5-flash` | correct, 4 steps, 10.0s, structural check clean |

On `demo_repo/` (10 lines, single concern) the 3B *has* succeeded — 2 steps,
13.2s, correct patch — but not reproducibly.

### Latency

| | |
|---|---|
| 3B cold start | 8.55s |
| 3B warm (keep_alive working) | 2.99s, ~10 tok/s |
| full 3B task, demo_repo | 13.2s / 2 steps |
| gemini-2.5-flash task | 10.0s / 4 steps |

### The honest conclusion

With everything above fixed, the 3B parses cleanly, reaches the gate with real
content, and receives **precise machine-generated feedback naming exactly what
it dropped**. It visibly *understands* it — *"I need to rewrite the patch...
without modifying the existing functions"* — and emits the same truncated file
anyway. It cannot reproduce a 10-line file verbatim.

The gate caught every attempt; **no file was ever corrupted**. That is the
system working as designed. But **whole-file rewriting is the wrong primitive
for a 3B**, and no amount of prompt engineering fixed it.

---

## Phase 4 — search/replace editing (the fix for 0/3)

**The diagnosis.** The model was doing two jobs: decide what to change (good at
it) and faithfully retype the lines it *wasn't* changing (terrible at it, and
it's most of the output tokens). Every failure was in job two. So job two was
deleted.

**`edit_file`** takes a FIND/REPLACE pair in the same fenced form as
`apply_patch`:

```
ACTION: edit_file
PATH: calculator.py
FIND:
```
def divide(a, b):
    return a / b
```
REPLACE:
```
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```
```

Design decisions worth keeping:

- **Exact match only.** The one tolerance is trailing whitespace (models get
  invisible characters wrong while getting the code right), and even that
  re-maps onto the real text so the file keeps its own spacing. No fuzzy
  matching beyond that — a paraphrased snippet is *refused*, which is what
  makes silent corruption structurally impossible rather than something the
  gate catches afterwards.
- **Ambiguity is an error, not a guess.** If FIND appears twice, refuse with
  the count and ask for more context. Picking one would be a coin flip on the
  user's code.
- **Empty REPLACE is a deletion**, distinguished from a missing block.
- **A failed match hands the file back** (`_edit_error`, bounded by
  `EDIT_ECHO_MAX_CHARS = 3000`). The commonest miss is the model writing what
  it wants the code to *become* rather than copying what's there, and it
  cannot fix that from a scolding alone. This was the single change that took
  the docstring task from looping to passing.
- Chose this over **line-range edits** (`replace lines 9-12`) deliberately:
  line numbers require counting, which small models are bad at, and they
  break the moment an earlier edit shifts them.
- `apply_patch` survives for **creating new files**, where there's nothing to
  preserve. Both tools converge on `(old_content, new_content)` inside
  `_handle_patch`, so the critic and structural veto behave identically.

### Measured effect

| backend | before | after |
|---|---|---|
| `qwen2.5-coder:3b` | 0/3 | **1/3** |
| `gemini-2.5-flash` | (3/3) | **3/3** |

The number matters less than the change in *failure mode*: since this landed
there have been **no deletions and no syntax errors in any run**. The 3B now
makes wrong edits rather than destructive ones. It's non-deterministic — the
docstring task passed one run and failed the next.

**`eval.py`** is the harness. It seeds a scratch repo in a tempdir, runs each
task end to end, and grades by **executing the result** (`safe_get({}, 'x')`
must return None; `last_n_items([1..5], 2)` must be `[4, 5]`) — never by
believing the model's summary. Grades in severity order: unparseable file →
deleted symbols → task not done.

---

## Phase 5 — transcripts, replay, and the interactive session

### `transcript.py`

Runs are recorded to `~/.arthur/runs/<timestamp>-<task-slug>.json` (override
with `ARTHUR_TRANSCRIPT_DIR`). Replay feeds the saved events back through the
*same* renderer that drew them live, so output is identical — measured 0.186s
for a 26-event two-task session, zero model calls.

Unknown event kinds from a newer build are **skipped, not fatal** — a partial
replay beats refusing the file. `arthur runs` lists recent ones with the exact
replay command.

### `session.py` — the headline feature

`arthur` with no args. The point is not the prompt loop, it's that `Session`
owns `messages` and hands the same list to `agent.run()` every task, mutated in
place. **Verified live:**

```
arthur > Add a zero check to divide
arthur > now add a docstring to the same function you just changed
   → it knew which function
```

- Slash commands handled locally, never sent to the model: `/help /model
  /backend /repo /context /diff /auto /clear /save /runs /exit`.
- `/repo` **clears the conversation** — old context describes code that is no
  longer in scope, so keeping it would mislead.
- `@path` mentions inline a file, bypassing keyword retrieval (which misses
  files whose contents share no vocabulary with the request).
- Ctrl+C cancels the *current task* and keeps the session; Ctrl+D exits.
- The session transcript is saved on exit automatically.

**Gotcha fixed here:** `prompt_toolkit` raises `NoConsoleScreenBufferError` on
Windows whenever stdout isn't a real console — which includes Git Bash, mintty,
and any piped input. `_make_reader` now checks `isatty()` first and catches
*all* exceptions, falling back to plain `input()`. A session without history is
much better than one that won't start.

---

## Test suite (431 tests)

| file | covers |
|---|---|
| `test_parser.py` | protocol parsing, incl. cases captured verbatim from live 3B output. Also runs standalone: `python tests/test_parser.py` |
| `test_events.py` | loop behaviour asserted on the event stream: rejection feedback, fail-closed on silence, auto-approve vs structural block, session continuity, repeat detection |
| `test_structural_gate.py` | the deletion veto and repo containment |
| `test_ollama_backend.py` | request shape (num_ctx, stop, keep_alive), streaming, `<think>`, error messages. No live model needed |
| `test_context.py` | what survives a trim — the system prompt, always |
| `test_cli.py` | entry-point wiring: repo resolution, `--model` override, preflight |
| `test_edit_file.py` | search/replace matching rules, refusals, FIND/REPLACE parsing |
| `test_session.py` | slash commands, `@` mentions, transcript round-trip and replay |
| `test_tool_robustness.py` | every tool failure becomes an observation, never a crash |
| `test_targeting.py` | which file a task is about: filename mentions, the per-task briefing, the refusal to rewrite an existing file, what a trim keeps |
| `test_edit_recovery.py` | recovering the two edit mistakes phi4-mini actually makes — the unclosed FIND fence and the stray `}` |
| `test_indentation.py` | line-anchored matching: every spelling of an anchor the model uses lands at the right depth |
| `test_append_file.py` | adding a function without deleting one; no-op edits; the misleading "file does not exist" error |
| `test_insert_after.py` | inserting below an anchor, the insertion depth rule, and splitting a merged AFTER block |
| `test_intent.py` | task vs question vs "hey" — and the lopsided bar that keeps real requests out of the chat path |
| `test_answer_mode.py` | a question runs read-only: every write tool refused at the dispatcher, prose taken as the answer |
| `test_stalling.py` | observations that hand back a line to copy, and aborting on a repeated call rather than a repeated sentence |
| `test_append_placement.py` | a method with `self` lands inside the class the file ends in — and a plain function never does |
| `test_render_dedup.py` | a streamed answer is not printed a second time as the summary |

Run: `pytest` (or `python tests/test_parser.py` without pytest).
Acceptance eval: `python eval.py --backend ollama|gemini`.

---

## Gotchas for next session

- **`arthur` is already installed editable.** Code edits take effect
  immediately; no reinstall unless `pyproject.toml` changes.
- **`demo_repo/calculator.py` and `evalrepo/*` are fixtures** that agent runs
  mutate. Reset them before demos — `demo_repo` should have a bare
  `def divide(a, b): return a / b`.
- Both `.env` (has a real Gemini key) and `~/.arthur/.env` are read; **`.env`
  deliberately overrides the shell environment** (a stale `GEMINI_API_KEY` in
  the Windows user env otherwise shadows it and you get a confusing 400).
- The `qwen2.5-coder:7b` tier is documented but **never pulled or tested** —
  4.7GB, would spill to CPU. Worth trying if the roadmap item #1 stalls.
- `evalrepo/` was created by me as a test fixture, not part of the original
  plan. It exists to be *harder* than `demo_repo`.

---

## Phase 6 — model selection, measured

Ran `eval.py` across every model that fits 4GB. Result overturned the
assumption the whole project was built on.

| model | score | per-turn | verdict |
|---|---|---|---|
| **`phi4-mini`** | **3/3** | **4.9s** | **new default** |
| `qwen2.5-coder:3b` | 1/3 | 13.1s | previous default |
| `qwen3:4b` | 1/3 | ~130s | unusable, see below |
| `gemini-2.5-flash` | 3/3 | ~3s | cloud control |

**The code-tuned model lost to the general-purpose one.** These tasks need
trivial code (`return d.get(key)`, `names[-n:]`) and rigorous
format-following. The bottleneck was always the second, and phi4-mini is far
better at it. A local model now matches the cloud on this set.

### Two bugs this uncovered

**1. My parser was case-sensitive.** phi4-mini follows the protocol perfectly
but writes it in Title Case (`Thought:` / `Action:` / `Args:`). The parser read
that as prose and scored `action=None`, so the model looked broken while doing
exactly as instructed. `_DECORATED_KEYWORD` is now `IGNORECASE` and upper-cases
what it matches. **A model that looks like it's ignoring the protocol is worth
one look at the raw output before it's written off** — the first screen said
"ignores the protocol" and the model was fine.

**2. Normalization was corrupting fenced content.** Making keywords
case-insensitive meant an ordinary source line like `# Action: tidy this up`
inside a CONTENT block would be rewritten to `ACTION:` and written to the
user's file. `_iter_regions` now splits on ``` and normalizes only outside
fences. This bug existed before the case change (for exact-case decorated
lines); case-insensitivity just made it likely enough to notice.

### `qwen3:4b` — do not use on this hardware

Reasoning mode can't be turned off. Given both `/no_think` (Qwen3's documented
marker, verified present in the request) and Ollama's `think: false`, it moves
its reasoning *out* of `<think>` tags and emits it as untagged prose — so
`strip_thinking()` can't remove it either. 130s per turn, 2372 chars of
rambling, no protocol. The full eval with thinking took 25 minutes to score
1/3; the `/no_think` variant was on course for over an hour and was killed.

The `/no_think` machinery was kept anyway: it's correct, it's scoped to
`THINKING_MODEL_PREFIXES`, and it'll work on builds that honour the flag.

### `eval.py` now screens first

One probe (~30s) before committing to the full set, checking the only two
things that make a run pointless: the model won't follow the format, or it's
too slow to finish. Prints a projected total runtime. `--force` overrides. This
directly pays for the qwen3 detour.

### Not tested

`qwen2.5-coder:7b` — user explicitly ruled it out (4.7GB, would spill to CPU
on a 4GB card). Do not pull it without asking.

---

## Phase 7 — surviving a bad turn

Five failures found by running it rather than by reading it. All were reachable
from a single malformed tool call.

- **`path='.'` killed the session.** The model named a directory; `open()` on a
  directory raises `PermissionError` on Windows, and it escaped the generator
  and took the whole conversation with it. Fixed at three levels: `BadTarget` +
  `check_target()` refuse directories and empty paths with a message the model
  can act on; `_safe` now catches `OSError`; `run_session` has a last-resort
  catch that reports and keeps going (`ARTHUR_DEBUG=1` for the traceback).
- **`ast.parse` is too lenient to be a validity check.** It builds a tree; a
  whole class of errors is only raised at *compile*. A model wrote a file
  ending in a module-level `return None`, `ast.parse` accepted it, the
  structural check reported "valid Python", and an unimportable file was
  written. `_top_level_symbols` now calls `compile()` as well.
- **`-y` overrode a critic REJECT.** A REJECTed edit (score 0.25) was applied
  and corrupted a file the agent had just written correctly. `-y` now means
  "stop asking me about judgement calls", not "turn the safety system off": a
  structural block and an outright REJECT both still stop.
- **The default backend was `mock`.** A user ran `arthur`, asked for
  `twosum.py`, and watched a canned `calculator.py` demo. Default is now
  `ollama`; mock announces itself in yellow.
- **The critic was never actually stubbed in tests.** `safety_gate` does
  `from .llm_backend import get_critic_backend`, so it holds its own reference
  — patching the name on `llm_backend` alone left the real critic in place.
  Every critic assertion in `test_events.py` was silently vacuous. Found when a
  new REJECT test failed with `APPROVE`.

---

## Phase 8 — knowing which file the task is about

The complaint that started this: *"one task it did pretty well, and after that I
told corrections to make in that same file — it struggles to read a file and
make changes, it always tries to build new files."*

That is one symptom with three independent causes.

### 1. Follow-up tasks were briefed on a stale repo

The repo tree and file contents lived in the **system prompt**, which
`run()` writes once per session (`if not messages:`). Task two therefore ran
against a snapshot taken *before* task one had written anything — and if task
one created the file, task two could not see it at all. Retrieval still ran
each task; its result was emitted as an event and then **thrown away**.

The fix splits the prompt in two:

- `build_system_prompt(repo_root)` — the protocol. Static, written once. This
  is the one message trimming may never drop, so only things that stay true
  belong in it.
- `build_task_briefing(task, index, files)` — the repo tree and the relevant
  file contents, rebuilt **every task** and appended as the user turn.

`context.trim_history` had to learn about this too: it used to preserve
`messages[1]` as "the original task", which in a multi-task session is the
*first* task's briefing — the wrong goal and a stale view of the repo. It now
keeps the newest briefing (`BRIEFING_PREFIX`), and skips it if the recent tail
already carries it.

### 2. Retrieval was a length contest

Old scoring: `sum(1 for t in haystack_terms if t in query_terms)` over
tokenized file *contents*, counting repeats. A long file that happened to use
`nums` and `target` forty times outranked the file the user had named outright.

Two mechanisms now, and the first matters far more:

- **Mention.** `mentioned_paths()` matches the task against indexed files by
  full relative path, bare filename, and bare stem (`"the twosum file"`),
  tolerating surrounding punctuation. A named file is pinned at
  `MENTION_SCORE = 10_000` — not a ranking question at all.
- **Overlap**, for when nothing is named: set-based, weighted by *where* the
  term matched (path 8 / symbol 5 / content 1) rather than how often.

### 3. `apply_patch` would overwrite an existing file

The cheapest thing for a small model to reach for, and the easiest way for it
to lose code. Now refused outright, with the current file echoed back and a
pointer to `edit_file`. Refusing rather than warning is the point: the
structural gate already catches rewrites that *delete* something, but it
catches them by stopping to ask a human, which turns every correction into an
approval prompt. Closing the path means the model must name the lines it is
changing — and a change it can name is one it cannot accidentally make.
Whole-file replacement is still reachable: `edit_file` with the entire current
text in FIND.

Fallout: every gate fixture in `test_events.py` was patching an existing file
via `apply_patch`, so they all had to move to `edit_file` — which is what a
real run does anyway. The bundled mock demo script moved too, so the canned
demo now models the behaviour we want.

---

## Phase 8b — two checks that ask a different question

Everything in the structural gate asked *does this symbol still exist*. Two
live failures got through because the answer was yes.

**Gutting.** Asked to add a docstring to `two_sum`, phi4-mini replaced the
whole function with a signature and a docstring. The name survived, the file
compiled, and the file got **longer** — so symbol-removal, compile and shrink
checks all passed a function that now returns `None`. `_is_hollow()` asks
whether a function still *does* anything; `gutted_symbols` reports any that
went from substantive to hollow. The reverse direction (a stub gaining an
implementation) is exactly what we want and is not flagged.

**Truncation on creation.** The same failure from the other side: asked to
*write* `two_sum`, the model produced a signature and a twelve-line docstring
and stopped — no algorithm. Nothing was deleted, nothing broke, the file only
grew. `hollow_additions` reports functions that *arrive* with no body. Tracked
separately from the destructive signals via `PatchShape.unfinished`, because
nothing was lost here — it just isn't finished. Methods are exempt: `...` in a
Protocol or ABC is deliberate code.

---

## Phase 8c — the two mistakes phi4-mini actually makes

Captured verbatim from a live turn. Both are unrecoverable *by the model* —
"the FIND text does not appear in the file" does not say which character is
wrong — so it repeated the identical turn until the repeat detector stopped the
run. Both are trivially recoverable in the parser.

**The unclosed FIND fence.** The model closes `REPLACE`'s fence but not
`FIND`'s. The search for ``` ran on and closed on the fence that *opens* the
replacement: FIND came out ending in the literal text `REPLACE:`, and the
replacement was swallowed whole. `_fenced_after` now stops at
`_BLOCK_BOUNDARY` — deliberately narrower than `KEYWORDS`, because real source
can contain a line beginning `PATH:` or `THOUGHT:` (this project's own prompt
strings do). Only a block keyword *alone on its line* ends a block early.

**The stray `}`.** Models trained mostly on C-family code close a Python block
with a brace. One character, invisible in a diff summary, and it makes the FIND
text unmatchable forever. `tools.strip_stray_closers()` drops trailing
brace-only lines — but **only unbalanced ones**, so the last line of a
multi-line dict literal is left alone, and only after an exact match has
already failed. It is applied to REPLACE too, but only on the pass where FIND
needed it: repairing the search half alone would splice the brace into the file
and guarantee a syntax error.

Also: a block that opens with no fence and ends with a stray one
(`_strip_loose_fences`), and better rejection feedback — each structural signal
now gets its own concrete instruction, quoting the model's own FIND block back
at it rather than offering a principle to apply.

### Prompt lessons

- **Example filenames leak into the model's world model.** The `apply_patch`
  example used `twosum.py` — which was also the file the user was asking
  about. Later, an example using `math_helpers.py` had the model open a turn
  with *"the repository contains files related to math helpers"* — in an empty
  repo. Placeholders must look like placeholders, and a test now asserts no
  indexed filename appears in the system prompt.
- **A worked example about docstrings teaches the model to write docstrings.**
  After adding one, phi4-mini started producing twelve-line docstrings and no
  function body. The example now uses a divide-by-zero guard, and the prompt
  says plainly: write the working code first, keep docstrings to one line.
- Showing a **WRONG** version next to the right one lands better than a rule.

---

## Phase 8d — `append_file`, and why adding a function needed its own tool

Task three of the live session — *"now add a second function three_sum to
twosum.py"* — failed in a way none of the above fixed, and it is the most
interesting failure of the lot.

To append with `edit_file` the model has to FIND the last lines of the file and
repeat them in REPLACE before its new code. phi4-mini instead put the **old**
function in FIND and the **new** one in REPLACE — which deletes the original.
The gate blocked it three turns running and the file was never damaged, but the
model never recovered either, because every piece of feedback it got was about
REPLACE blocks when the real answer was *a different tool*.

Appending is not a matching problem, so it should not be expressed as one.
`append_file` takes `PATH:` and `CONTENT:` and adds to the end. What makes it
safe is structural rather than checked: **the old content is a strict prefix of
the new**, so nothing the model writes in CONTENT can alter a byte that is
already there. `compute_append` handles the PEP 8 blank lines so the model
doesn't have to.

The rejection path learned it too: a FIND that removes a definition while
REPLACE introduces a new one is an *add* expressed with the wrong tool, and
saying so is more use than another lecture about REPLACE blocks.

### The misleading error that cost five turns

After the hollow-body check rejected a truncated `apply_patch`, the model
switched to `edit_file` on a file that **did not exist yet**. `preview_edit`
read it as empty, handed the empty string to the matcher, and returned *"the
FIND text does not appear in the file"* — true, and useless. The model read it
as a matching problem, rewrote the FIND block, and got the identical answer
until the repeat detector fired. `_handle_patch` now checks existence first and
names the actual problem.

A pattern worth keeping: **for a small model, an error message is the entire
recovery mechanism.** Every unrecoverable loop in this project traces back to
an observation that was technically accurate and diagnostically useless.

### Measured effect (phi4-mini, live, same three-task session)

| | before | after |
|---|---|---|
| create `twosum.py` | truncated to a docstring, or 5 wasted turns | one turn, correct |
| follow-up edit to that file | built a new file instead | `edit_file`, one turn |
| add a second function | deleted the first one, 3 turns, then gave up | `append_file`, one turn |
| file ever corrupted | yes | no — bad patches never reached disk |
| unclosed fence / stray brace | fatal, repeated to the step cap | recovered in the parser |

Final state of the file, all three tasks in one session, ~50s per task:

```python
def two_sum(nums, target):        # task 1
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []                     # task 2: [] instead of None


def three_sum(nums, target):      # task 3, appended, two_sum untouched
    ...
```

All three verified by execution, not by eyeballing the diff.

---

## Phase 8e — `insert_after`, and the lesson from the whole phase

The docstring eval task — "add a docstring to the `total_count` method" — was
the last one still failing, and it failed differently every run. Chasing each
variant produced a string of genuine fixes (indentation-aware matching, no-op
detection, a precise dropped-anchor message) and it still failed, because none
of them addressed the actual problem.

**The tool could not say what the model meant.** `edit_file` expresses
*replace these lines with those lines*. The model means *insert this after that
line*. Bridging the two requires repeating the anchor inside REPLACE, and
models will not reliably do it. Told in as many words that REPLACE had to
contain the `def` line, phi4-mini went on sending REPLACE blocks holding a
docstring and nothing else — and once "fixed" the complaint by dropping the
docstring instead and reproducing the file unchanged, which the agent then
reported as success.

`insert_after` takes `AFTER:` (an existing line) and `CONTENT:` (the new lines)
and, like `append_file`, cannot delete anything by construction. The insertion
depth comes from the file rather than from the model: the deeper of the
anchor's own indent and the indent of the line below it. Two cases, one rule —
after `def f():` the line below is the body and is deeper, so the insertion
joins the body; after the last statement of a block the line below dedents out
of it, so the insertion stays put. No knowledge of Python required.

### The merged block

First live run with the new tool, the model switched to it immediately on being
shown the form — and then wrote the anchor and the new lines *together* under
`AFTER:`, omitting `CONTENT:` entirely. Because that is what the finished code
looks like. Five turns, each answered "insert_after is missing: content", which
it plainly could not act on.

That one is recoverable and not by guessing: the longest run of leading AFTER
lines that is actually **in the file** is the anchor, and what follows cannot
be, or it would have matched too. `REQUIRED_ARGS` no longer demands `content`,
and `compute_insert` splits the block.

### Measured

Five consecutive runs, same code, temperature 0.1. Per task, and per run:

| task | runs passed | what it needs |
|---|---|---|
| `docstring` | **5/5** | insert a line -> `insert_after` |
| `add-function` | **5/5** | add a function -> `append_file` |
| `null-check` | 4/5 | one small edit, correct logic |
| `off-by-one` | 1/5 | work out that `-(n+1)` should be `-n` |
| **run totals** | **4, 2, 3, 3, 3 of 4** | mean 3.0/4 |

Before this phase the two addition tasks passed **0/4 and 0/4**. They are now
the only two that never fail, and neither needed a smarter model -- each needed
a tool that could express an addition.

What is left is not mechanical. `off-by-one` fails because phi4-mini writes
`names[:n]` when it means `names[-n:]`: a clean edit, correctly applied to the
right file, that is simply wrong. The same thing happened in a live session --
asked to return `[]` instead of `None`, it produced valid, well-formed code that
returns `[]` on the first iteration every time. The 1.5B critic approved it.

That is the honest ceiling here: the harness now reliably does what the model
asks it to do, and the model is a 3.8B. The next real gain is not another
parser fix, it is verification -- `run_command` exists and the model never
chooses to use it (see the roadmap).

### The lesson worth keeping

Every unrecoverable loop in this project traces to one of two things:

1. **An observation that was accurate and diagnostically useless.** "The FIND
   text does not appear in the file" is true and tells the model nothing about
   *which character* is wrong. For a small model the error message is the
   entire recovery mechanism; if it does not name the fix, there is no fix.
2. **A tool that could not express what the model was trying to say.** No
   amount of prompt engineering closes that gap. Both times the fix was a new
   verb, and both times it worked on the first live run.

Before concluding a local model is too small for a task, check whether your
tools can say what it is trying to say.

---

## Phase 9 — not everything you type is a task

Reported from real use, and the transcript is the whole bug report:

```
arthur > hey
--- step 1/12 ---
THOUGHT: The task is to add a new function to an existing file. I need to find
a suitable file to add the function to.
ACTION  list_dir()
--- step 2/12 ---
ACTION  search_code()      ERROR: missing required argument 'query'
--- step 3/12 ---
ACTION  search_code()      ERROR: missing required argument 'query'
--- step 4/12 ---
ACTION  search_code()      ERROR: missing required argument 'query'
FAILED at step 5
```

"The task is to add a new function to an existing file" appears nowhere in the
input. It is the subject of the system prompt's own worked examples. Handed
`TASK: hey` by a briefing that assumes there is work to do, the model borrowed
the nearest plausible job and spent five steps failing at it.

Five separate causes, all of which had to be fixed.

### 1. A greeting is not a task — `intent.py`

Answered in Python, instantly, without a model call. The classifier is
deliberately lopsided: routing a task to chat is a real failure, routing chat
to the agent is only the status quo, so every rule proves something is *chat*
and anything unproven is a task. Code signals (a file extension, a path
separator, `snake_case`, a call, a backtick) and write verbs override
everything; past five words a line is making a request whatever its vocabulary.

The one genuinely ambiguous opener in English is `do`: "do you ever call
run_command" asks, "do that again for the class below" instructs. Settled by
what follows — a subject pronoun asks, anything else instructs, and `it` is
excluded because "do it again" is the commonest instruction there is.

### 2. A question is not a change request — answer mode

The deeper half. Asked "what does the retriever score files on?", the agent ran
four turns of `edit_file` with no FIND block. Its own thought gives it away:

> "The user has not provided the specific lines to find and replace."

It had worked out there was nothing to replace and called the replacing tool
anyway — because the briefing's closing paragraph is entirely about which file
to **change**, and the protocol's only non-editing exit is `FINAL`, twenty
lines below eight editing examples.

A question now runs the agent with the writing tools **refused at the
dispatcher**. Not advice — telling a 3B not to edit gets it to agree and then
edit anyway. Reading tools are untouched; the briefing asks for an answer.

Two follow-on fixes, both found live:

- **Prose is an answer.** phi4-mini answered the question correctly and in
  full, in plain prose, and got `PROTOCOL_REMINDER` for it — then produced "No
  action taken" on the retry. A correct answer discarded over a missing
  five-character prefix. In answer mode a turn with no ACTION is now taken as
  the answer. Safe only because nothing can be written; edit mode still treats
  it as a violation, since prose there means no patch was produced.
- **Don't print it twice.** Streaming echoes tokens until a protocol keyword
  appears. Prose has no keyword, so the whole answer streamed and then arrived
  again as the summary.

### 3. The block layout the parser insisted on — worth two eval tasks

`docstring` regressed to 0/3 and the cause was none of the above. The model
writes all three of these, and only the first used to parse:

```
AFTER:                      AFTER: ```                  AFTER: total_count:
```                         def total_count(self):
def total_count(self):      ```
```
```

`_fenced_after` required `^KEYWORD:[ \t]*\n`. The other two returned `None` —
and `parse_block_form` drops the **whole call** when a required block is
missing, so a turn carrying a perfectly good `PATH:` arrived as
`insert_after()` with no arguments at all, and the observation said "missing:
path, after" about a turn that had supplied the path. The model apologised and
re-sent it identically until the stall detector fired.

Everything after the keyword is now optional: the newline, the fence, both.

This is the older lesson from the other side. The observation was not merely
useless, it was **false** — and it was false because the parser threw the
answer away before anything looked at it. A model cannot recover from a
correction about a mistake it did not make.

### 3b. `AFTER: total_count:` — an abbreviation, not a mismatch

The third layout above leaves a different problem: the anchor is the method's
*name*, where the file says `    def total_count(self):`. "The AFTER text does
not appear in the file" is true and unactionable.

A one-line anchor that matches nothing now falls back to identifier
containment: if every identifier in it appears on exactly one line of the file,
that line is the anchor. Two candidates is a guess rather than a near-miss, and
both get named back instead.

Only safe because of what `insert_after` is — every existing line survives
whatever this resolves to, so the worst case is a new line in the wrong place,
never lost code. The same fallback in `edit_file` would be indefensible.

### 4. "missing required argument 'query'" taught nothing

Naming the missing key tells the model what it already knew. The read tools now
hand back a line to copy — `ACTION: search_code` / `ARGS: {"query": "def
retrieve"}` — and a test asserts every hint actually parses to the call it
claims to be, so a hint can never teach a broken form.

### 5. Twelve steps of watching a model fail

`MAX_STEPS` 12 → **8**. At ~11 tok/s twelve steps is two minutes of a model
that decided at step 2 what it was going to do. Every task this agent completes
takes one to four steps; runs that reach double figures have never recovered.

The repeat detector also compared response *text*, so rewording the THOUGHT
while re-sending the same broken call underneath bought three more steps. It
now compares **calls**.

### And one correctness bug found by testing this

`now add a count method to cart.py` produced:

```python
+def count(self):
+    return len(self.items)
```

At column zero, after a file that ended inside `class Cart`. A module-level
function taking a parameter called `self`. It compiles, no symbol was removed
or hollowed, every existing line survived — so the structural veto had nothing
to say, and the 1.5B critic approved it, reporting that "a `count` method has
been added to the `Cart` class". It read the diff the way a person skims one:
by the words in it rather than the columns.

This is the shape of every failure left in this project. It was worth taking as
an exception because the model's intent was not ambiguous at all — it wrote
`self`, `self` means method, and *where* the method goes is a question about
the file, which the file can be asked. `compute_append` now indents into a
class the file ends inside, when every definition being added takes `self` or
`cls`. A plain function is never dragged in; an unparseable file falls back to
a plain append.

**Verified live**: `count()` lands inside `Cart` and returns 3 for three items.

### Measured, five consecutive runs

| task | before Phase 9 | after |
|---|---|---|
| `add-function` | 5/5 | **5/5** |
| `null-check` | 4/5 | **5/5** |
| `off-by-one` | 1/5 | **5/5** |
| `docstring` | 5/5 | 3/5 |
| **totals** | 4,2,3,3,3 | **4,3,4,3,4** — mean 3.0 → **3.6** |

**`off-by-one` is the row that matters, and it is an embarrassment.** Phase 8e
wrote it up as the clearest example of a residual *reasoning* failure — "the
model writes `names[:n]` when it means `names[-n:]`, a clean edit that is
simply wrong". Phase 9 changed nothing about slicing, prompts about slicing, or
the gate. It fixed three parser and placement bugs, and the task went 1/5 to
5/5. The model had been solving it all along and the harness was throwing the
answer away.

That conclusion was wrong in a specific and instructive way: a mechanical
failure was diagnosed as a cognitive one because the *output* was inspected and
the *transport* was assumed correct. Anything the harness discards looks
exactly like something the model never produced.

`docstring` regressing 5/5 → 3/5 turned out to be a fourth variant of the same
class — a blank line after the opening fence, making the anchor two lines
instead of one — caught by capturing raw turns. Fixed after this table was
measured, and then measured on its own: **6/6 live**, against 3/5 before. The
table above is left as it was recorded rather than re-run, because a number
attached to a build that no longer exists is worse than no number.

### What this phase adds to the running lesson

Phase 8e concluded that every unrecoverable loop traced to a useless
observation or a tool that could not say what the model meant. Phase 9 adds a
third:

3. **A mode the agent did not have.** "Answer this" was not expressible. The
   agent had one gear — find a file and change it — so every input was bent
   into that shape, including `hey`. The greeting was the visible symptom; the
   missing gear was the bug.
4. **A correct answer the harness discarded.** The worst of the four, because
   it is invisible from the outside and it corrupts the diagnosis. Four
   separate variants of "the block did not parse" produced observations that
   were not merely unhelpful but **false** — "missing: path, after" to a turn
   that supplied both, "the AFTER text does not appear in the file" about a
   line sitting in the file. A model cannot recover from a correction about a
   mistake it did not make, and a task lost this way is indistinguishable from
   a task the model could not do. That is how `off-by-one` spent a phase
   labelled a reasoning limit.

The operational form of all four: **when a small model fails, capture the raw
turn before theorising.** Every fix in this phase came from `repr(response.text)`
and none from reading the rendered transcript, which by construction shows only
what survived parsing.

That is now built in rather than remembered: `ActionProposed` carries the
unparsed turn whenever the arguments came back **empty**. That is exactly the
case where the transcript is otherwise evidence-free — `insert_after` with
`args: {}` reads identically whether the model sent nothing or sent a perfect
call the parser could not read — and it is rare enough to cost nothing. Four
reproductions in this phase would have been four `grep`s.

---

## Next up (roadmap, highest value first)

0. **Fuzz the parser against real turns.** Promoted above verification by
   Phase 9, which found four separate layouts phi4-mini writes and the parser
   rejected -- each one costing a whole task, and one of them costing a phase
   of believing the model could not reason. There is no reason to think the
   fourth was the last. Collect raw turns across the eval set (the machinery
   now exists: `ActionProposed.raw`), assert every one either parses or is
   deliberately refused, and keep them as fixtures. This is cheap, and it is
   the only item here with a demonstrated 1/5 → 5/5 behind it.
1. **Verify by running the code.** Still the top *design* change. Some
   remaining failures are correct-looking wrong answers, and no structural
   check catches one -- `run_command` exists and the model never chooses it. A
   hard-coded post-patch step (run the file or the tests, feed a failure back
   as an observation) would turn "wrong logic" into another correctable
   mistake,
   exactly as echoing the file turned a failed FIND into one. Also try
   `phi4-mini` as the critic: the 1.5B approves logic errors.
2. **Calibration numbers.** Self-confidence is pinned at ~1.0 on Gemini
   (carrying no information at all) and shows a real spread on the 3B
   (0.5–0.9 observed). Measuring this decides whether the 50/50 self/critic
   weighting is defensible or should become critic-only on cloud backends. It
   is currently a guess, and it's the weakest claim in the writeup.
2. **Try `qwen2.5-coder:7b`.** Never pulled or tested — 4.7GB, spills to CPU
   on the 4GB card, expect ~5 tok/s. The open question is whether the residual
   3B failures are a *size* problem or a *fundamentally too small* problem.
   One `python eval.py` run answers it. This is the cheapest remaining
   experiment with a real chance of moving 1/3 → 3/3 locally.
3. **`/undo`.** Listed in no menu because `patcher.py` keeps no backups. Add
   a backup on write and the slash command is trivial.
4. **Multi-file changes** — a list of edits per turn applied as one
   transaction with rollback. The gate would need to evaluate the set, not
   each file.
5. Embedding retrieval (ChromaDB) vs keyword, compared on the `eval.py` task
   set — the comparison table is itself a good writeup.
6. `rich` rendering for the session. Isolated to `TerminalRenderer`; nothing
   else would change. Declared in the `[repl]` extra;
   `prompt_toolkit` **is** installed, `rich` is not.

### Ideas that came up but weren't done

- Auto-repairing a truncated patch by re-appending the dropped functions —
  rejected as too clever; it hides a model failure instead of surfacing it.
- A `verify` step that runs the tests after each patch. The tools support it
  (`run_command`) but the 3B never chooses to. Would need prompting or a
  hard-coded post-step.
