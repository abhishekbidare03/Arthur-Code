# Arthur

A small, from-scratch coding agent you run from any terminal — the "read the
repo, plan, make changes, report back" loop that tools like Claude Code
implement, built by hand so every mechanism is visible instead of hidden
behind a framework.

**Status:** installable as an `arthur` command, with a persistent interactive
session, transcripts and replay. Verified end-to-end on the scripted mock
backend (zero network, zero deps), on **Ollama** (`phi4-mini` on a 4GB GTX
1650 Ti — the target hardware, and where every interesting failure came from),
and on the **Gemini API** (`gemini-2.5-flash`). 431 tests; the acceptance eval
grades by executing the result rather than by reading the diff.

## Install

```bash
pip install -e .
```

That puts `arthur` on your PATH. From then on, `cd` into any repository and
the agent is already pointed at it — the target repo defaults to the directory
you're standing in, the way `git` works.

```bash
arthur                           # interactive session in this repo
arthur -p "TASK"                 # run one task here and exit
arthur doctor                    # GPU, backend, models, protocol compliance
arthur runs                      # list saved runs
arthur --replay <file>           # re-show a past run, no model called
```

### The interactive session

`arthur` with no arguments opens a session that **keeps its conversation across
tasks**, which is the difference between a script and a tool:

```
arthur > add a zero check to divide
  ...applies the edit...
arthur > now add a docstring to the same function you just changed
  ...knows which function...
```

Three kinds of input, routed before the agent starts:

| you type | what happens |
|---|---|
| `hey`, `thanks`, `what can you do` | answered in Python, instantly, no model call |
| `what does the retriever score on?` | agent runs **read-only** — writing tools refused, it answers |
| `add a docstring to total_count` | agent runs normally, proposes a patch |

That split exists because it was missing. Typing `hey` used to start a real
run: the model, handed `TASK: hey` and a prompt full of editing examples,
announced "the task is to add a new function to an existing file" — a sentence
that appears nowhere in the input and everywhere in the prompt — and spent five
steps failing at it. See Phase 9 in `LOGS.md`.

Slash commands are handled locally and never reach the model: `/help`,
`/model`, `/backend`, `/repo`, `/context`, `/diff`, `/auto`, `/clear`,
`/save`, `/runs`, `/exit`. Prefix a path with `@` to force it into context
(`@utils.py`) when keyword retrieval would miss it.

Every run is recorded to `~/.arthur/runs/` and replays through the same
renderer that drew it live — same output, no model, ~0.2s. That's the thing
worth demoing.

## Run it right now

```bash
cd demo_repo
arthur -p "Add error handling for division by zero in the divide function in calculator.py"
```

The mock backend is the default and needs no key, no network and no model.
You'll see the full loop: indexing → retrieval → THOUGHT/ACTION → diff preview
→ the confidence gate + critic vote → the file actually getting patched → a
final report.

## Run it against a real model (Gemini)

```bash
cp .env.example .env                # then paste your key into .env
arthur doctor --backend gemini      # pre-flight: reachable + protocol-compliant?
arthur -p "Add error handling for division by zero" --backend gemini -C demo_repo
```

Get a key at <https://aistudio.google.com/apikey>. `.env` is gitignored.

Because `arthur` runs from arbitrary directories, `.env` is looked for in
three places, nearest-wins: `~/.arthur/.env` (your machine-wide key, set once)
→ the project root → the repo you're currently in.

**Gotcha worth knowing:** `.env` deliberately *overrides* an existing
`GEMINI_API_KEY` in your shell environment. A stale key left in the Windows
user environment otherwise silently shadows the project's key and you get an
`API key not valid` 400 from a key you never meant to send.

Backend selection, in precedence order: `--backend` flag → `AGENT_BACKEND`
env var → the `BACKEND` default in `config.py`. `--model` overrides the model
id for one run.

## Running local (Ollama)

Model choice is driven by VRAM, and the defaults here are sized for a 4GB
card:

| role | model | ~size (Q4) | why |
|---|---|---|---|
| coder (default) | `phi4-mini` | 2.5 GB | measured best on the acceptance eval — see below |
| critic | `qwen2.5-coder:1.5b` | 1.0 GB | judging a finished diff is classification, not generation — a small model does it, and different weights make the second opinion genuinely independent |

```bash
ollama pull phi4-mini
ollama pull qwen2.5-coder:1.5b
arthur doctor --backend ollama
```

### Which local model, measured

`python eval.py --backend ollama --model <name>` on a GTX 1650 Ti (4 GB):

| model | score | per-turn | note |
|---|---|---|---|
| **`phi4-mini`** | **3/3** | **4.9s** | matches `gemini-2.5-flash` on this set |
| `qwen2.5-coder:3b` | 1/3 | 13.1s | edits safely, but often edits the wrong thing |
| `qwen3:4b` | 1/3 | ~130s | thinking mode; `/no_think` is ignored, see below |

That table is the original three tasks. The eval has since grown a fourth
(adding a function to an existing file) and the editing tools have changed
underneath it, so for current numbers — including how much they vary run to
run — see [the four-task results](#current-results) below.

The result worth internalising: **the code-tuned model lost to the general
one.** These tasks need trivial code and rigorous format-following, and the
bottleneck is the second. Picking a model by "is it a coder model" would have
got this wrong; running the eval got it right.

`qwen3:4b` is a trap on this hardware. It reasons before every answer, and
disabling that doesn't work — given `/no_think` and `think: false` it simply
moves the reasoning out of `<think>` tags and emits it as untagged prose, so it
can't even be stripped. 130s per turn, and the output isn't protocol.

`eval.py` screens a model with one probe before running the full set, so an
unusable model costs 30 seconds instead of an hour.

`arthur doctor` reports free VRAM, whether the daemon is up, which models are
installed, and — most usefully — whether the model actually *follows the
protocol* rather than answering in prose. A reachable model that replies
"Sure! I'd be happy to help" is useless here and burns every step of the loop.

### Why raw HTTP instead of the `google-genai` SDK

The project already needs `requests` for Ollama, so the Gemini backend adds
**zero new dependencies** — and one visible HTTP call is easier to reason
about (and to explain) than an SDK abstraction. Three things about Gemini's
wire format differ from the OpenAI/Anthropic shape this project uses
internally, all handled in `GeminiBackend`:

1. the system prompt is a top-level `system_instruction`, not a message with
   role `system`;
2. the assistant role is called `model`, not `assistant`;
3. 2.5-series models emit internal *thinking* parts that must be filtered
   out of the response before parsing. Thinking is disabled by default
   (`GEMINI_THINKING_BUDGET = 0`) — it costs latency and tokens and buys
   little on a rigid THOUGHT/ACTION/ARGS protocol.

429s and 5xx are retried with exponential backoff; 400/403/404 fail loudly,
because those mean a bad key or a bad model name and no amount of retrying
fixes them.

## Architecture

```
pyproject.toml      declares the `arthur` console script
arthur/
  cli.py            the `arthur` command: one-shot, doctor, interactive
  intent.py         task, question, or just "hey" -- decided before the loop starts
  session.py        the interactive session: history, slash commands, @mentions
  doctor.py         pre-flight: GPU, daemon, installed models, auto-pull, protocol probe
  agent.py          the loop, as a generator yielding events -- never prints
  events.py         the typed vocabulary the loop speaks in
  render.py         the only module that prints; drives the loop, asks the human
  context.py        token budgeting, so the window never silently truncates
  indexer.py        walks the repo, extracts file tree + Python symbols (ast)
  retriever.py      files the task NAMES first, keyword overlap second
  tools.py          list_dir / read_file / search_code / edit_file / insert_after /
                    append_file / apply_patch / run_command, all resolved against
                    the repo root and refused if they escape
  transcript.py     save / load / replay a run as JSON
  patcher.py        unified diff for display, plus ast-level structural analysis
  safety_gate.py    self-confidence + independent Critic vote + a deterministic
                    structural veto -> auto-apply, or stop and ask
  llm_backend.py    pluggable: MockBackend (scripted) / OllamaBackend / GeminiBackend
  config.py         all knobs + a tiny .env loader (no python-dotenv dependency)
tests/              431 tests across 19 files
  test_parser.py    protocol parsing, incl. cases captured from live 3B output
  test_events.py    the loop's behaviour, asserted on the event stream
  test_structural_gate.py   the deletion veto and repo containment
  test_intent.py    task vs question vs "hey", and the bar for each
  test_answer_mode.py       a question runs read-only; prose counts as the answer
  test_append_placement.py  a method with `self` lands inside the class
  test_ollama_backend.py    request shape, streaming, error messages
  test_context.py   what survives a trim (the system prompt, always)
  test_cli.py       entry-point wiring: repo resolution, overrides, preflight
  ...               see LOGS.md for the full table
demo_repo/          10-line toy repo; the 3B handles this reliably
evalrepo/           25-line multi-function repo; the 3B does not
main.py             backwards-compatible shim for the old `python main.py` form
```

Run the tests with `pytest`, or `python tests/test_parser.py` if you'd rather
not install one.

### Why the loop yields events instead of printing

`run_agent` used to drive the loop *and* write to stdout, which meant nothing
else could consume a run. `agent.run()` is now a generator emitting
`Thought` / `DiffReady` / `GateDecision` / `Observation` / `Final`, and three
consumers share one implementation: the terminal renderer, the JSON transcript
(next phase), and the tests — which assert on the event sequence instead of
scraping stdout.

Human approval works the same way: the loop *yields* `ApprovalNeeded` and the
consumer sends a `Decision` back in. The loop never calls `input()`, so the
same code runs interactively, headless, and under a test. A consumer that
doesn't answer is read as a rejection — silence is not consent when the
question is "may I write to disk".

## Key design decisions (and why)

- **Text-based ReAct protocol, not native function-calling.** Small local
  models are unreliable at structured tool-calling. `THOUGHT: / ACTION: /
  ARGS:` is just text continuation, so it degrades gracefully on weak models
  — and hand-parsing it is the actual learning goal here.
- **Full-file rewrite instead of diff parsing.** Asking a small model to
  emit a syntactically valid unified diff is asking it to fail. Asking for
  the whole corrected file and computing the diff yourself is far more
  robust. The diff is still shown to the user/gate for review.
- **Two-agent safety gate.** The Coder proposes a change and self-rates its
  confidence. An independent Critic agent — that never saw the Coder's
  reasoning — reviews the diff and votes APPROVE/REJECT. The two signals are
  combined; only changes that clear the threshold auto-apply, everything
  else pauses for a human `y/N`. This is the same entropy/confidence-gating
  pattern already used elsewhere, generalized into an actor+critic pair.
- **Keyword retrieval, not embeddings — for now.** Simplest thing that
  works. The natural upgrade is exactly the ChromaDB embedding pipeline
  already built for the offline Phi-3 RAG project — a clean "what I'd do
  next" answer in an interview.
- **A forgiving parser, because real models are not the mock.** Chat-tuned
  models bold the keywords, wrap ARGS in ```json fences, append "hope that
  helps!" after the JSON, and put *literal* newlines inside JSON strings
  (which is invalid JSON). The parser normalizes all of that, extracts the
  first *balanced* `{...}` while tracking string state, and re-escapes stray
  control characters before a second parse attempt. Protocol keywords are
  only recognized at the start of a line — otherwise a patch whose contents
  contain the text `FINAL:` would end the run mid-edit. All 12 cases are in
  `test_parser.py`.
- **The gate fails closed.** If the critic call itself errors (rate limit,
  timeout), that is recorded as REJECT, not as an approval. A reviewer that
  never answered is not a reviewer that said yes.

## What the live local run actually showed

The first real run against `qwen2.5-coder:3b` on a 4GB card produced four
findings, none of which the mock backend could ever have surfaced. All four
are now covered by tests.

**1. The model deleted code, and the critic approved it.** Asked to add one
docstring to a 25-line file, the 3B returned a "patched" file with two
top-level functions and the module docstring missing. The 1.5B critic
approved it — correctly noting that the requested docstring had indeed been
added, and never noticing what was gone.

This is the failure mode inherent in the whole-file-rewrite design: the choice
that makes small models usable (they cannot emit valid unified diffs) is
exactly what lets them silently truncate a file. **An LLM reviewer cannot be
relied on to catch it**, because the patch does contain the thing it was asked
to check for.

The fix is the third signal in the gate, and the only one that isn't an
opinion: a deterministic `ast`-level check comparing top-level symbols before
and after. If a patch removes definitions, breaks parsing, or deletes more
than `MAX_SHRINK_RATIO` of the file, it is **blocked outright regardless of
score** — two models agreeing that deleting half a file is fine is still two
models being wrong. `-y` does not override it, because "stop asking me about
judgement calls" is not the same as "turn the safety system off".

**2. Whole-file rewrite doesn't scale down to 3B.** On a single-concern file
(`demo_repo/calculator.py`, 10 lines) the 3B succeeds reliably — 2 steps, 13s,
correct patch. On a 25-line file with six functions it fails every time,
because reproducing the untouched code verbatim as escaped JSON is most of the
work and it gets it wrong. Measured on three tasks (add a docstring, add a
null check, fix an off-by-one): **0/3 for `qwen2.5-coder:3b`.** The same tasks
and the same harness on `gemini-2.5-flash`: correct, 4 steps, 10s, structural
check clean. The pipeline is sound; 3B is simply under the bar for multi-function
files. Multi-file patches with line-range edits — not whole-file rewrites —
are the real fix, and that is what the next phase should be.

**3. Asking a 3B for JSON-embedded source code doesn't work, and patching the
parser is a losing game.** Across live runs the same model failed *differently
every time*: Python triple quotes one run (`{"new_content": """<file>"""}`),
single-quoted values wrapping a bogus nested object the next, unescaped literal
newlines the run after. Each fix is another regex and there is always another
malformation.

So `apply_patch` now accepts a second spelling with no escaping at all:

```
ACTION: apply_patch
PATH: calculator.py
CONTENT:
```python
<the whole file, verbatim>
```
```

Small models are heavily trained on fenced code blocks and produce them
reliably. JSON args still work, still take precedence when they parse, and are
still what the short-argument tools use — this is an additional accepted form,
not a replacement.

(The triple-quote repair survives as a fallback, with a trap worth recording:
it must match **greedily**, because the file being written usually contains
triple quotes of its own — any module with a docstring does. A non-greedy match
stops at that inner docstring and truncates everything after it, producing a
plausible-looking patch that deletes most of the file.)

**4. A model will report success having done nothing.** Two variants, both
observed. Sometimes the 3B emits `FINAL: ...fixed it` on step 1 without ever
calling a tool. More insidiously, it emits a *correct patch* and then, without
waiting, invents its own observation and a triumphant `FINAL` in the same turn
— and a parser that honours `FINAL` first silently discards the patch and
reports a success that changed nothing. Whichever of `ACTION` and `FINAL`
appears first in the text now wins, `\nTHOUGHT:` is a stop sequence so the
model can't start a second turn, and the run summary states plainly when no
files were changed. The model's claim and the actual result are never confused.

Also worth noting: near-deterministic local models repeat their mistakes
*verbatim*, so a malformed turn recurs identically until the step cap. The loop
detects three identical replies and stops at step 4 instead of burning 12.

**5. Whole-file rewriting was the wrong primitive — and fixing that fixed the
agent.** With everything above in place the 3B still could not reproduce a
10-line file verbatim. It visibly *understood* the feedback ("I need to rewrite
the patch... without modifying the existing functions") and emitted the same
truncated file anyway. The gate caught every attempt and no file was ever
corrupted, but the task was impossible as posed.

The diagnosis: the model was being asked to do two jobs — decide what to
change (it's good at this) and faithfully retype the 20 lines it *wasn't*
changing (it's terrible at this, and it's most of the output). So the second
job was deleted. `edit_file` takes a search/replace pair:

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

The model never mentions the other functions, so it cannot delete them. The
match is **exact** — no fuzzy fallback beyond trailing whitespace — so a
paraphrased snippet is refused rather than approximated, and silent corruption
becomes structurally impossible instead of something the gate detects
afterwards. Ambiguous matches (the snippet appears twice) are refused with a
count, and a failed match hands the file back so the model can copy the real
text instead of guessing again.

**Measured effect on the same three tasks: 0/3 → 1/3 on `qwen2.5-coder:3b`,
3/3 on `gemini-2.5-flash`.** More important than the number: across every run
since, there have been **no deletions and no syntax errors**. The 3B's failures
are now wrong edits rather than destructive ones — it makes a change that
doesn't fix the bug, instead of one that eats the file. That is a categorically
safer failure, and it's the difference between a toy and something you'd let
near real code.

Run it yourself: `python eval.py --backend ollama`. The harness seeds a scratch
repo, runs each task end to end, and grades by **executing the result** — not
by believing the model's summary, which is exactly the thing that can't be
trusted.

## What the live Gemini run actually showed

Two findings from running this for real rather than against the mock — both
are the kind of thing worth saying out loud in an interview:

- **The critic works, and it is not a rubber stamp.** Fed a deliberately
  wrong patch for the division-by-zero task — one that silently `return 0`
  instead of raising — the critic rejected it: *"handles division by zero but
  returns 0, which is an incorrect result."* With the coder self-reporting
  0.95, the combined score fell to 0.47 and the gate correctly paused for a
  human. This is the adversarial case Phase 4 of the plan asks for.
- **Self-reported confidence is badly calibrated on a strong cloud model.**
  Gemini rated its own correct patch `CONFIDENCE: 1.00`. A signal that is
  always ~1.0 carries no information, which means on cloud backends the
  critic is doing nearly all the real work in the 50/50 blend. The
  self-confidence signal is likely to matter far more on a small local
  model, where the model is actually uncertain sometimes — worth measuring
  on Ollama before defending the 50/50 weighting.

## Picking the file, and the right way to change it

The first real user session found the same complaint three times over: *"one
task it did pretty well, and after that I told corrections to make in that same
file — it struggles to read a file and make changes, it always tries to build
new files."* One symptom, four causes, all now fixed.

**The follow-up was briefed on a stale repo.** The repo tree and file contents
lived in the system prompt, which is written **once per session**. Task two ran
against a snapshot taken before task one had written anything — so if task one
created the file, task two could not see it and rebuilt it from scratch. The
prompt is now split: `build_system_prompt()` holds the protocol and nothing
else, and `build_task_briefing()` rebuilds the repo state for **every** task.
Trimming had to learn this too — it used to preserve `messages[1]` as "the
original task", which in a multi-task session is the *first* task's briefing.

**Retrieval was a length contest.** Scoring summed keyword hits over file
contents *counting repeats*, so a long file that happened to use the same words
outranked the file the user had named outright. Files the task names are now
found by path, filename or bare stem and pinned above everything; keyword
overlap is the fallback, weighted by *where* a term matched rather than how
often.

**`apply_patch` would overwrite an existing file** — the cheapest thing for a
small model to reach for and the easiest way for it to lose code. Now refused
outright, with the current file handed back and a pointer to `edit_file`.
Refusing rather than warning is the point: closing the path means the model has
to *name* the lines it is changing, and a change it can name is one it cannot
accidentally make.

**Adding had no good spelling at all.** This turned out to be the big one, and
it took two new tools to fix.

`edit_file` can only say *replace these lines with those lines*. But most small
tasks are additions, and the model thinks about them as *insert this after
that* — which `edit_file` can only express by repeating the anchor inside
REPLACE. Models will not reliably do that. Asked to add a second function,
phi4-mini put the *old* function in FIND and the *new* one in REPLACE, deleting
the original. Asked to add a docstring, it sent a REPLACE block containing the
docstring and nothing else, run after run, failing a different way each time —
and once "fixed" the complaint by dropping the docstring instead and
reproducing the file unchanged.

None of that is a capability problem. It is a mismatch between what the model
means and what the tool can say. So:

| the model means | tool | can it delete anything? |
|---|---|---|
| insert this after that line | `insert_after` | **no** |
| add a whole new function | `append_file` | **no** |
| change these lines | `edit_file` | yes |
| create this file | `apply_patch` | refused if it exists |

Both new tools are safe by construction rather than by inspection — every
existing line is kept, so there is no argument they can be given that removes a
character of what is there. `insert_after` works out the insertion depth from
the file itself (the deeper of the anchor's own indent and the line below it),
so the model never has to think about columns.

Both tools paid for themselves on the first live run — `add-function` and
`docstring` went from never passing to passing, and neither ever needed a
smarter model. They needed a tool that could express an addition. The pattern
generalises past this project: when a small model keeps failing one *shape* of
task, check whether your tools can say what it is trying to say before
concluding it is too small.

### Two checks that ask a different question

Everything in the structural gate asked *does this symbol still exist*. Two
live failures got through because the answer was yes:

- **Gutting.** Asked to add a docstring, the model replaced the whole function
  with a signature and a docstring. The name survived, the file compiled, and
  the file got *longer* — so the removal, compile and shrink checks all passed
  a function that now returns `None`.
- **Truncation on creation.** The same failure inverted: asked to *write*
  `two_sum`, it produced a signature and a twelve-line docstring and stopped.
  Nothing deleted, nothing broken, file only grew.

`_is_hollow()` asks whether a function still *does* anything. Going the other
way — a stub gaining an implementation — is exactly what we want, so only the
substantive→hollow direction counts, and methods are exempt (`...` in a
Protocol is deliberate code).

### Prompt lessons worth keeping

- **Example filenames leak into the model's world model.** The `apply_patch`
  example used `twosum.py`, which was also the file the user was asking about.
  A later example using `math_helpers.py` had the model open a turn with *"the
  repository contains files related to math helpers"* — in an empty repo. A
  test now asserts no indexed filename appears in the system prompt.
- **A worked example about docstrings teaches the model to write docstrings.**
  After adding one, phi4-mini started producing twelve-line docstrings and no
  function body.
- Showing a **WRONG** version beside the right one lands better than a rule.
- **The examples become the model's idea of what tasks are.** Given `TASK: hey`
  — nothing to do, and a prompt whose bulk is editing examples — phi4-mini
  announced "the task is to add a new function to an existing file". It had
  read the prompt correctly. A prompt that only ever demonstrates one kind of
  work implies that all work is that kind.

### Knowing when *not* to write

Three input kinds, decided in `intent.py` before the agent starts, because the
agent had exactly one gear and bent every input into it.

The classifier is deliberately lopsided: routing a real task to the chat reply
is a genuine failure, while routing chat to the agent is only the old
behaviour. So every rule *proves* something is chat, and anything unproven is a
task. File extensions, path separators, `snake_case` and write verbs override
everything — "hey can you fix the parser" is a task, not a hello.

A question runs the agent with the writing tools **refused at the dispatcher**,
not discouraged in the prompt. Asked what the retriever scores on, phi4-mini
had spent four turns calling `edit_file` with no FIND block, its own thought
reading *"the user has not provided the specific lines to find and replace"* —
it knew there was nothing to replace and called the replacing tool anyway,
because that was the only shape of action the briefing described. Telling a 3B
not to edit gets it to agree and then edit anyway.

Two smaller things fell out of testing it:

- **Prose is an answer.** It answered the question correctly, in full, in plain
  English — and got a protocol warning, then replied "No action taken" on the
  retry. A correct answer thrown away for want of a five-character prefix.
  Since nothing can be written in this mode, a turn with no ACTION is now
  simply the answer.
- **The block layout the parser insisted on.** `AFTER:` then a fence on the
  next line was the only accepted form; the model also writes `AFTER: ``` ` and
  a bare `AFTER: total_count:`. Both parsed as `None`, and a missing block
  drops the *whole* call — so a turn carrying a perfectly good `PATH:` arrived
  with no arguments, and the observation said "missing: path, after" about a
  turn that had supplied the path. **A model cannot recover from a correction
  about a mistake it did not make.** Two eval tasks were lost to this.

## Current results

Five consecutive `python eval.py --backend ollama` runs on `phi4-mini`, same
code, temperature 0.1, each graded by **executing** the result:

| task | before Phase 9 | after | what it needs |
|---|---|---|---|
| `add-function` | 5/5 | **5/5** | add a function → `append_file` |
| `null-check` | 4/5 | **5/5** | one small edit, correct logic |
| `off-by-one` | 1/5 | **5/5** | work out that `-(n+1)` should be `-n` |
| `docstring` | 5/5 | 3/5 | insert a line → `insert_after` |
| **run totals** | 4,2,3,3,3 | **4,3,4,3,4** | mean 3.0 → **3.6** of 4 |

Report the range, not the best run — quoting the 4/4 would be picking a number
rather than measuring one.

**Read the `off-by-one` row carefully, because it is the surprise.** It was
called a reasoning failure and written up as one: the model "writes `names[:n]`
when it means `names[-n:]`". Phase 9 changed no prompt about slicing and no
gate. It fixed three parser and placement bugs — and the task went 1/5 → 5/5.
The model had been getting it right and the harness had been dropping it.

That is worth stating plainly because the earlier writeup got it wrong. Before
concluding a small model cannot reason, check that you are not discarding the
answer on the way out.

The `docstring` regression traced to a fourth variant of the same class — a
blank line after the opening fence, making the anchor two lines instead of one
— found by capturing raw turns. Fixed after this measurement and then measured
on its own: **6/6 live**, against 3/5 before. The table is left as recorded
rather than re-run, because a number attached to a build that no longer exists
is worse than no number.

**What is left really is not mechanical**, though there is less of it than
claimed. In a live session, asked to return `[]` instead of `None`, phi4-mini
produced well-formed code returning `[]` on the first iteration every time, and
the 1.5B critic approved it. The next real gain is verification: `run_command`
exists, and the model never chooses to use it.

## Known limitations (own these in an interview, don't hide them)

- Single-file patches only; no multi-file transactions or rollback.
- phi4-mini frequently omits `CONFIDENCE`, so self-confidence falls back to
  0.5 and the gate is effectively critic-only for that model. A single
  APPROVE from a 1.5B then clears the threshold — weaker than the two-signal
  design claims.
- **The gate checks structure, not correctness.** Every deterministic check
  here answers "did this patch damage the file", and they work. None answers
  "is this code right", and the 1.5B critic reliably approves logic errors.
  A clean, well-formed, confidently-applied wrong answer passes everything.
- Results vary run to run at temperature 0.1 — 2/4 to 4/4 on the same code.
  Any single-number claim about a small local model is noise.
- `run_command` has no real sandboxing beyond a timeout — fine for your own
  repo, not fine for untrusted code.
- Retrieval is keyword-based, so it misses semantically related files that
  don't share vocabulary with the task description.
- The mock backend proves the *loop* works, not that any particular LLM is
  good at this. Real quality depends entirely on the model you plug in.
- Only tested on a one-file toy repo so far. Nothing here has met a real
  codebase where retrieval has to actually discriminate.
- The 50/50 self-confidence/critic weighting is a guess, not a tuned value —
  see the calibration note above.
- **Intent routing is rules, not a model.** It has to be — asking the LLM
  whether your input is a task costs a turn on the thing being avoided. The
  known miss is a change request phrased as a pure question ("why not return
  `[]` instead?"), which gets explained rather than applied. That is the safe
  direction, and the answer-mode footer says how to re-ask; the reverse
  mistake, ignoring a real request, would be much worse.
- **The abbreviated-anchor fallback is a heuristic.** `insert_after` accepts
  an anchor that merely names its target when exactly one line matches. Safe
  only because that tool cannot delete; it would be indefensible in
  `edit_file`, and it is not used there.

## Roadmap

Done: hardened Ollama backend (`num_ctx`, stop sequences, `keep_alive`,
streaming, `<think>` stripping, actionable errors); the loop as an event
generator; a deterministic structural veto in the gate; repo-contained tool
paths; search/replace editing; per-task briefings and filename-aware
retrieval; `append_file`; `insert_after`; the hollow-function checks; intent
routing and read-only answer mode; JSON transcripts with `--replay`; and the
interactive session.

1. **Fuzz the parser against real turns.** Promoted to the top by the
   `off-by-one` result above. Four separate layouts the model writes were
   being rejected, each costing a whole task, and one of them cost a phase of
   believing the model could not reason. Nothing suggests the fourth was the
   last. Collect raw turns across the eval set — the machinery now exists,
   since `ActionProposed` keeps the unparsed text whenever args come back
   empty — assert each one either parses or is deliberately refused, and keep
   them as fixtures. Cheap, and the only item here with a measured 1/5 → 5/5
   behind it.
2. **Verify by running the code.** Still the top *design* change. Some
   remaining failures are genuinely correct-looking wrong answers, and no
   structural check can catch one. `run_command` already exists and the model
   never chooses it. A hard-coded post-patch step — run the file, or the tests,
   and feed a failure back as an observation — would turn "wrong logic" into
   another correctable mistake, exactly as showing the file back turned a
   failed FIND into one.
3. **Calibration numbers.** Self-confidence is worth measuring now that both a
   local and a cloud model run: Gemini pins itself at ~1.0 (carrying no
   information at all), the 3B shows a real spread, and phi4-mini frequently
   omits the line altogether so the gate silently falls back to 0.5. That
   decides whether the 50/50 self/critic weighting is defensible, or whether
   a missing CONFIDENCE should be treated as a refusal to answer rather than
   as mediocre confidence. Related: try `phi4-mini` as the critic instead of
   `qwen2.5-coder:1.5b` — the 1.5B approved a module-level function as a
   method, and a better critic costs seconds per patch.
4. **Try `qwen2.5-coder:7b`.** Never tested — 4.7GB, spills to CPU on a 4GB
   card. Worth much less than it was: the failures that motivated it turned
   out to be parser bugs, not size. Run item 1 before spending an hour here.
5. **`/undo`.** The patcher doesn't keep backups yet, so the slash command is
   absent. Cheap and obviously useful.
6. **Multi-file changes** — a list of edits per turn, applied as one
   transaction with rollback.
7. Swap `retriever.py`'s keyword scoring for embedding + ChromaDB retrieval,
   and compare on the same fixed task set.
8. `rich`-based rendering for the session (the plumbing is already isolated in
   `render.py`; only `TerminalRenderer` would change).
