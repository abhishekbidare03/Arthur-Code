# Build Plan: Mini Coding Agent

A phased plan for building a proper, portfolio-grade version of a small
"read the repo → plan → make changes → report" agent, using Claude Code to
implement it. A rough working reference prototype is attached alongside
this plan (mock-backend tested, runs end-to-end) — use it as a baseline to
compare against and improve on, not as the finish line.

---

## 0. Scope decision (decide this before writing any code)

Pick ONE target explicitly and tell Claude Code which one — it changes a
lot of downstream decisions:

- **A. Standalone CLI tool** ("mini-code-agent"), demoable on its own,
  portable to any repo. Fastest to build, cleanest to demo in an interview.
- **B. A module inside Arthur**, reusing Arthur's existing Ollama
  integration, SQLite state, and 4GB VRAM budget constraints. Stronger
  narrative ("I extended my own offline assistant"), more integration work.

If unsure: build A first as a clean standalone project, then port the
core (agent.py + safety_gate.py) into Arthur as a second commit. That
gives you two resume lines instead of one.

---

## 1. Non-negotiable design goals

State these explicitly to Claude Code as constraints, not suggestions:

1. **No agent framework** (no LangChain/LangGraph/CrewAI for the core loop).
   The entire point is demonstrating you understand the mechanism. You've
   already used LangGraph in Autonomous Data Scientist — this project's
   differentiation is that it's hand-built.
2. **Works with a local model first**, cloud API as a fallback/comparison
   option — keeps it consistent with your offline-first identity.
3. **Every filesystem-mutating action is gated** by a confidence check
   before it executes — this is the feature that makes the project yours,
   not a generic agent-loop clone.
4. **Every run produces a reviewable artifact** (diff + transcript), not
   just terminal output that disappears — this is what you'll actually
   show in an interview or demo video.

---

## 2. Architecture (module responsibilities)

```
indexer      -> understands the repo structure
retriever    -> picks relevant context for a given task
llm_backend  -> talks to the model (local or cloud), backend-agnostic
agent        -> the loop: prompt, parse response, dispatch tool, observe, repeat
tools        -> the actual read/search/write/run operations
patcher      -> turns a proposed file into a reviewable diff
safety_gate  -> Coder confidence + independent Critic vote -> apply or ask human
transcript   -> logs every step to a JSON file for later review/demo
cli          -> entry point
```

Data flow for one task:

```
task text
  -> indexer.build_index(repo)              [file tree + symbols]
  -> retriever.retrieve(task, index)        [top-K relevant files]
  -> agent.run(task, index, context)
       loop:
         llm.chat(messages) -> THOUGHT / ACTION / ARGS  (or FINAL)
         if ACTION is destructive:
             patcher.diff(...) -> safety_gate.evaluate(...)
             if not approved: ask human / skip
         tools.dispatch(ACTION, ARGS) -> observation
         messages.append(observation)
       until FINAL or MAX_STEPS
  -> transcript.save(all steps)
  -> print final report
```

---

## 3. The agent protocol (be explicit about this with Claude Code)

Use a plain-text ReAct-style protocol, not native function-calling, so it
works on small local models too:

```
THOUGHT: <reasoning>
ACTION: <tool_name>
ARGS: <json on one line>
```

For any action in the destructive set (file writes, shell commands), the
model must also emit:

```
CONFIDENCE: <0.0-1.0>
```

Completion:

```
THOUGHT: <reasoning>
FINAL: <plain-English summary of every change made and why>
```

Decide up front how strict the parser should be (regex vs. asking the
model to output JSON-only) and have Claude Code write unit tests against
5-10 example model outputs, including malformed ones, before wiring it
into the live loop.

---

## 4. The safety gate (your differentiator — spend real design time here)

Two independent signals, combined:

- **Self-confidence**: the Coder agent's own `CONFIDENCE:` value.
- **Critic vote**: a second LLM call, different system prompt, that only
  sees the task + diff (never the Coder's reasoning) and votes
  APPROVE/REJECT with a one-line reason.

Combine (e.g. weighted average, or require critic APPROVE as a hard gate
with self-confidence only affecting a secondary threshold) and pick a
threshold below which execution pauses for human confirmation.

Things worth actually testing, not just building:
- Does the critic catch anything the coder missed on a deliberately
  wrong patch? (Write 2-3 adversarial test cases.)
- What's the false-positive rate (blocking correct patches) vs.
  false-negative rate (approving bad ones) on a small hand-built test set?
  Even 10 examples with a table of results is a strong thing to show.

This is the part that turns "I built an agent loop" into "I built a
trust mechanism for autonomous code changes" — the second is a much
better fellowship/interview story.

---

## 5. Build order (with acceptance criteria per phase)

**Phase 1 — Indexing + retrieval (no LLM yet)**
- Walk a repo, build file tree + Python symbol extraction.
- Keyword-based retrieval scored against a task string.
- Acceptance: given a real multi-file repo and 5 sample task strings,
  the top retrieved file is manually verified correct for at least 4/5.

**Phase 2 — Mock-backend agent loop**
- Scripted LLM responses driving the full loop on a toy repo (already
  proven in the attached prototype).
- Acceptance: a full run produces THOUGHT → diff preview → gate decision
  → applied patch → FINAL report, with zero crashes.

**Phase 3 — Real local backend**
- Wire up Ollama with a small coder model.
- Acceptance: on 3 real small tasks (add a docstring, fix an off-by-one,
  add a null check), the agent completes at least 2/3 without human
  intervention, and the parser doesn't choke on real model output
  quirks (extra prose, markdown fences, etc. — handle these explicitly).

**Phase 4 — Safety gate hardening**
- Add the critic call, tune the threshold using the adversarial test
  set from section 4.
- Acceptance: at least one deliberately-wrong patch is caught and
  blocked; document the case in the README.

**Phase 5 — Transcript + demo polish**
- JSON transcript of every run (thoughts, diffs, gate scores, decisions).
- A `--replay transcript.json` mode that pretty-prints a past run without
  re-calling the LLM — this is what you actually show people.
- Acceptance: you can demo a completed run without live-calling anything.

**Phase 6 (stretch) — Embedding retrieval**
- Swap keyword retrieval for the same ChromaDB embedding approach used in
  the offline Phi-3 RAG pipeline. Compare retrieval quality before/after
  on the same 5 task strings from Phase 1 — this comparison table is
  itself a good writeup.

**Phase 7 (stretch) — Multi-file patches + Arthur integration**
- Support a list of file changes per turn, not just one.
- If pursuing scope B, port `agent.py` + `safety_gate.py` into Arthur.

---

## 6. What to actually measure (for the resume line and the writeup)

Don't just build it — collect 3-5 numbers while you do:

- Task success rate on a fixed test set (N tasks, how many completed
  correctly without human override).
- Gate accuracy: of the patches you manually reviewed as objectively
  right or wrong, how often did the gate's decision match?
- Latency per step, local model vs. cloud API, on your actual hardware.
- Context window used vs. repo size (shows you understand the
  retrieval-scales-with-repo-size problem).

These numbers are what separate "I built a demo" from "I built and
evaluated a system" on a resume.

---

## 7. Resume / interview framing (once built)

> Built a coding agent from first principles (no LangChain) implementing
> a ReAct-style tool-use loop with repo indexing, keyword/embedding
> retrieval, and a dual-agent (Coder + independent Critic) confidence
> gate that blocks low-confidence file mutations pending human review —
> generalizing the entropy-based safety-gating pattern from [OSCC /
> Docket / Phi-3 RAG work] into autonomous code editing.

Be ready to answer, unprompted:
- "Why not just use an existing framework?" → understanding the
  mechanism was the point; also small local models need graceful
  degradation frameworks don't provide for free.
- "How do you know the gate actually works?" → point to the adversarial
  test set and the measured false-positive/negative numbers, not vibes.
- "What would you do with more time?" → embeddings retrieval, multi-file
  transactions with rollback, real sandboxing for run_command.

---

## Reference prototype (attached)

`config.py`, `llm_backend.py`, `indexer.py`, `retriever.py`, `tools.py`,
`patcher.py`, `safety_gate.py`, `agent.py`, `main.py`, `demo_repo/`,
`README.md` — a working, mock-backend-tested version of Phases 1-2 above.
Run it, read it, then have Claude Code rebuild it properly against this
plan rather than patching the prototype in place — the prototype cut
corners (no tests, no transcript, single-file patches only) that Phase
5-7 above are meant to fix.
