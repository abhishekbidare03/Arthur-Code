# Arthur

**A coding agent built from scratch — no LangChain, no framework — that runs
against a 3B local model on a 4GB laptop GPU.**

`cd` into a repo, type `arthur`, and describe a change in plain English. It
finds the right file, proposes a patch, shows you the diff, and applies it only
after passing a safety gate. Every mechanism — the ReAct protocol, the parser,
retrieval, the confidence gate, the event loop — is hand-written and visible.

| | |
|---|---|
| **Language** | Python 3.12, stdlib + `requests` |
| **Tests** | 438, across 19 files |
| **Backends** | Ollama (local) · Gemini · scripted mock (no network) |
| **Target hardware** | GTX 1650 Ti, **4 GB VRAM** — the binding constraint on every design decision |
| **Evaluation** | 4-task harness that grades by **executing** the result, not by reading the diff |

```bash
pip install -e .
cd demo_repo
arthur -p "Add error handling for division by zero in calculator.py"
```

The mock backend is the default: no key, no network, no model. You'll see the
whole loop run end to end.

---

## Contents

[What it looks like](#what-it-looks-like) · [Results](#results) ·
[How it works](#how-it-works) · [Design decisions](#design-decisions-that-mattered) ·
[What broke](#what-broke-and-what-it-taught) · [Usage](#usage) ·
[Limitations](#limitations) · [Roadmap](#roadmap)

---

## What it looks like

```
arthur > add a count method to cart.py that returns how many items there are

indexed 1 file(s)
context: cart.py (10000)

--- step 1/8 ---
THOUGHT: The count method should return the number of items in the cart.
CONFIDENCE: 0.9
ACTION  append_file(path='cart.py', content='def count(self):...')

proposed change to cart.py:
@@ -5,3 +5,6 @@
     def total(self):
         return sum(item["price"] for item in self.items)
+
+    def count(self):
+        return len(self.items)

GATE    self=0.90  critic=APPROVE  -> 0.95 (threshold 0.65)
        critic: adds a `count` method to the `Cart` class...
applied automatically (cleared the gate)

DONE  2 step(s), 54.9s
files changed: cart.py
```

The session **keeps its conversation across tasks**, which is the difference
between a script and a tool:

```
arthur > add a zero check to divide
arthur > now add a docstring to the same function you just changed
  ...knows which function...
```

Three kinds of input are routed *before* the agent starts:

| you type | what happens |
|---|---|
| `hey`, `thanks`, `what can you do` | answered in Python, instantly — no model call |
| `what does the retriever score on?` | agent runs **read-only** — writing tools refused |
| `add a docstring to total_count` | agent runs normally, proposes a patch |

Every run is recorded to `~/.arthur/runs/` and replays through the same
renderer that drew it live — same output, no model, ~0.2s.

---

## Results

Five consecutive `python eval.py --backend ollama` runs on `phi4-mini`,
temperature 0.1, each task graded by **executing** the patched file:

| task | before | after | what it needs |
|---|---|---|---|
| `add-function` | 5/5 | **5/5** | add a function → `append_file` |
| `null-check` | 4/5 | **5/5** | one small edit, correct logic |
| `off-by-one` | 1/5 | **5/5** | work out that `-(n+1)` should be `-n` |
| `docstring` | 5/5 | 3/5 → **6/6** | insert a line → `insert_after` |
| **run totals** | 4,2,3,3,3 | **4,3,4,3,4** | mean 3.0 → **3.6** of 4 |

The range is reported rather than the best run. A single 4/4 happened, and
quoting it would be picking a number instead of measuring one.

### The `off-by-one` row is the interesting one

It had been written up as a *reasoning* failure — "the model writes `names[:n]`
when it means `names[-n:]`, a clean edit that is simply wrong." Then a round of
parser fixes, touching nothing about slicing or prompts, took it from 1/5 to
5/5.

The model had been solving it all along. The harness was discarding the answer,
and a discarded answer is indistinguishable from a wrong one.

> **Before concluding a small model cannot reason, check that you are not
> throwing away its output on the way out.**

### The gate is not a rubber stamp

Fed a deliberately wrong patch — one that silently returns `0` instead of
raising on division by zero — the critic rejected it: *"handles division by
zero but returns 0, which is an incorrect result."* With the coder
self-reporting 0.95, the combined score fell to 0.47 and the gate paused for a
human.

The counterweight, stated honestly: Gemini rates its own correct patches
`CONFIDENCE: 1.00` every time. A signal that is always ~1.0 carries no
information, so on cloud backends the critic does nearly all the real work in
the 50/50 blend.

---

## How it works

```
             ┌──────────┐
input ──────▶│  intent  │──── greeting ──▶ answered locally, no model call
             └────┬─────┘──── question ──▶ agent, read-only
                  │ task
                  ▼
        index ──▶ retrieve ──▶ briefing ──▶ ┌─────────────┐
                                            │  agent.run  │  generator,
                                            │   (loop)    │  never prints
                                            └──────┬──────┘
                                                   │ yields events
                    ┌──────────────────────────────┼──────────────┐
                    ▼                              ▼              ▼
              renderer (TUI)               JSON transcript      tests
```

Each loop step: the model emits `THOUGHT / ACTION / ARGS` as plain text, the
parser recovers a tool call, the tool runs, and the observation goes back in.
Write tools detour through the safety gate first.

### The safety gate — three independent signals

| signal | asks | who answers |
|---|---|---|
| self-confidence | "how sure are you?" | the coder, in its own turn |
| critic vote | "is this diff correct?" | a **separate** 1.5B model that never saw the coder's reasoning |
| structural veto | "did this damage the file?" | deterministic `ast` analysis — no model involved |

The first two are blended against a threshold; the third is an absolute veto
that `-y` cannot override. **The gate fails closed** — if the critic call
errors, that is recorded as REJECT. A reviewer that never answered is not a
reviewer that said yes.

### The eight tools

`list_dir` · `read_file` · `search_code` · `run_command` ·
`edit_file` · `insert_after` · `append_file` · `apply_patch`

Three of the four write tools are **safe by construction** — `insert_after` and
`append_file` keep every existing line, so no argument they can be given
removes a character of what is there; `apply_patch` refuses to overwrite an
existing file. Only `edit_file` can delete, which makes it the only one the
structural veto has to watch closely.

---

## Design decisions that mattered

**Text protocol, not native function-calling.** Small local models are
unreliable at structured tool-calling. `THOUGHT: / ACTION: / ARGS:` is plain
text continuation, so it degrades gracefully — and hand-parsing it was the
point.

**The loop is a generator that never prints.** `agent.run()` yields typed
events (`Thought`, `DiffReady`, `GateDecision`, `Final`). Three consumers share
one implementation: the terminal renderer, the JSON transcript, and the tests —
which assert on the event sequence instead of scraping stdout. Human approval
works the same way: the loop *yields* `ApprovalNeeded` and receives a decision
back, so it never calls `input()` and runs identically interactive, headless,
and under test.

**Search/replace, not whole-file rewrites.** Asking a 3B to reproduce a whole
file verbatim fails — it truncates, and the truncation looks like a valid
patch. Asking for just the lines that change is the primitive that fits the
model.

**A separate small model as critic.** Judging a finished diff is
classification, not generation, so a 1.5B does it — which keeps the gate cheap
enough to run on every patch and makes the second opinion genuinely independent
weights rather than the same model re-scoring its own work.

**A forgiving parser, because real models are not the mock.** Chat-tuned models
bold the keywords, wrap args in JSON fences, append "hope that helps!", and put
literal newlines inside JSON strings. Keywords are only recognised at the start
of a line — otherwise a patch containing the text `FINAL:` would end the run
mid-edit.

**Context budgeted deliberately.** Ollama silently truncates from the *front*,
which is where the protocol definition lives. `context.py` decides what to drop
itself, and the system prompt is never a candidate.

---

## What broke, and what it taught

The interesting engineering is in the failures. Full detail — nine phases of it
— is in **[`LOGS.md`](LOGS.md)**. The short version:

- **An accurate error message can be useless.** "The FIND text does not appear
  in the file" is true and says nothing about *which character* is wrong. For a
  small model the error message is the entire recovery mechanism.
- **An error message can be worse than useless — it can be false.** Four
  different block layouts the model writes were rejected by the parser, so a
  turn that supplied a path was told "missing: path". A model cannot recover
  from a correction about a mistake it did not make.
- **A tool that cannot express what the model means is unfixable by
  prompting.** Asked to *add* a function, the model put the old one in FIND and
  the new one in REPLACE — deleting it. No prompt fixed that; a new verb
  (`append_file`) fixed it on the first live run.
- **Structural checks catch damage, not wrongness.** A method appended at
  column zero — a module-level function taking `self` — compiled, deleted
  nothing, and was approved by the critic as "a method added to the class".
- **Worked examples become the model's idea of what tasks are.** Given
  `TASK: hey`, the model announced "the task is to add a new function to an
  existing file" — a sentence found nowhere in the input and everywhere in the
  prompt.

Every fix came from capturing the raw model turn, so that is now built in: the
transcript keeps the unparsed text whenever a call comes back with a hole in
it.

---

## Usage

```bash
pip install -e .          # puts `arthur` on your PATH
```

The target repo defaults to the directory you're standing in, the way `git`
works.

```bash
arthur                    # interactive session in this repo
arthur -p "TASK"          # run one task and exit
arthur doctor             # GPU, backend, models, protocol compliance
arthur runs               # list saved runs
arthur --replay <file>    # re-show a past run, no model called
```

Slash commands are handled locally and never reach the model: `/help`,
`/model`, `/backend`, `/repo`, `/context`, `/diff`, `/auto`, `/clear`,
`/save`, `/runs`, `/exit`. Prefix a path with `@` to force it into context
(`@utils.py`) when keyword retrieval would miss it.

### Local models (Ollama)

Model choice is driven by VRAM. These are sized for a 4 GB card:

| role | model | size | why |
|---|---|---|---|
| coder | `phi4-mini` | 2.5 GB | measured best on the acceptance eval |
| critic | `qwen2.5-coder:1.5b` | 1.0 GB | classification, not generation |

```bash
ollama pull phi4-mini
ollama pull qwen2.5-coder:1.5b
arthur doctor --backend ollama
```

`arthur doctor` reports free VRAM, whether the daemon is up, which models are
installed, and — most usefully — whether the model actually *follows the
protocol* rather than answering in prose.

**Which model, measured** (the original 3-task set):

| model | score | per-turn |
|---|---|---|
| **`phi4-mini`** | **3/3** | 4.9 s |
| `qwen2.5-coder:3b` | 1/3 | 13.1 s |
| `qwen3:4b` | 1/3 | ~130 s |

The result worth internalising: **the code-tuned model lost to the general
one.** These tasks need trivial code and rigorous format-following, and the
bottleneck is the second.

> ⚠️ These numbers predate the parser fixes described in [Results](#results),
> and those fixes moved one task from 1/5 to 5/5 with no model change. The
> comparison is due a re-run.

`qwen3:4b` is a trap on this hardware — it reasons before every answer, and
`/no_think` doesn't disable it, it just moves the reasoning out of the tags
where it can't be stripped.

### Cloud (Gemini)

```bash
cp .env.example .env      # then paste your key into .env
arthur -p "TASK" --backend gemini
```

`.env` is searched nearest-wins: `~/.arthur/.env` → project root → current
repo, and it deliberately *overrides* a `GEMINI_API_KEY` already in your shell,
because a stale key in the Windows user environment otherwise silently shadows
the project's key.

Raw HTTP rather than the SDK, so the Gemini backend adds **zero new
dependencies** — the project already needs `requests` for Ollama.

---

## Testing

```bash
pytest                          # 438 tests
python tests/test_parser.py     # the parser suite, no pytest needed
python eval.py --backend ollama # the acceptance eval, grades by executing
```

The eval screens a model with one probe before running the full set, so an
unusable model costs 30 seconds instead of an hour.

---

## Limitations

Stated plainly, because the honest version is more useful than the flattering
one.

- **The gate checks structure, not correctness.** Every deterministic check
  answers "did this patch damage the file", and they work. None answers "is
  this code right", and the 1.5B critic reliably approves logic errors. A
  clean, well-formed, confidently-applied wrong answer passes everything.
- **Results vary run to run** at temperature 0.1 — 2/4 to 4/4 on the same code.
  Any single-number claim about a small local model is noise.
- **`phi4-mini` often omits `CONFIDENCE`**, so self-confidence falls back to
  0.5 and the gate is effectively critic-only for that model — weaker than the
  two-signal design claims.
- **The 50/50 self/critic weighting is a guess**, not a tuned value.
- **Single-file patches only.** No multi-file transactions, no rollback, no
  `/undo`.
- **Retrieval is keyword-based**, so it misses semantically related files that
  don't share vocabulary with the task.
- **Intent routing is rules, not a model** — it has to be, since asking the LLM
  whether your input is a task costs a turn on the thing being avoided. The
  known miss is a change request phrased as a pure question, which gets
  explained rather than applied. That is the safe direction.
- **`run_command` has no real sandboxing** beyond a timeout. Fine for your own
  repo, not for untrusted code.
- **Only tested on small repos.** Nothing here has met a codebase where
  retrieval has to genuinely discriminate.

---

## Roadmap

1. **Fuzz the parser against captured real turns.** Four layouts the model
   writes were being silently rejected, each costing a whole task. Nothing
   suggests the fourth was the last. The only item here with a measured
   1/5 → 5/5 behind it.
2. **Verify by running the code.** `run_command` exists and the model never
   chooses it. A hard-coded post-patch step — run the tests, feed a failure
   back as an observation — would turn "wrong logic" into another correctable
   mistake.
3. **Calibrate the confidence weighting** with real numbers instead of a guess.
4. **`/undo`** — needs the patcher to keep backups.
5. **Embedding retrieval** (ChromaDB) compared against keyword scoring on the
   same fixed task set.

---

## Repo map

```
arthur/
  cli.py          the `arthur` command: one-shot, doctor, interactive
  intent.py       task, question, or just "hey" — decided before the loop
  session.py      interactive session: history, slash commands, @mentions
  agent.py        the loop as a generator, the protocol parser, the prompt
  events.py       the typed vocabulary the loop speaks in
  render.py       the only module that prints
  safety_gate.py  self-confidence + critic vote + structural veto
  patcher.py      unified diff, plus ast-level structural analysis
  tools.py        the eight tools, all path-contained
  retriever.py    files the task NAMES first, keyword overlap second
  indexer.py      file tree + Python symbols (ast)
  context.py      token budgeting, so the window never silently truncates
  transcript.py   save / load / replay a run as JSON
  llm_backend.py  Mock / Ollama / Gemini
  config.py       every knob, plus a tiny .env loader
tests/            438 tests across 19 files
eval.py           the 4-task acceptance harness
LOGS.md           the build log — nine phases, and where the real story is
```
