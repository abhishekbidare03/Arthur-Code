"""
Working out WHICH file a task is about, and changing that file rather than
writing a new one.

Every test here comes from the same live complaint: the agent would do a task
well, and then be unable to act on a correction to the file it had just
written. It would build a new file instead. Three separate causes, one
symptom:

  - the repo tree and file contents lived in the system prompt, which is
    written once per session, so a follow-up task was answered against a
    snapshot taken before the first task ran;
  - retrieval scored by counting repeated tokens in a file's body, which is a
    length contest -- a file named outright in the task could be outranked;
  - apply_patch would happily overwrite an existing file with one rewritten
    from memory, which is the cheapest thing for a small model to reach for
    and the easiest way for it to lose code.
"""

import pytest

from arthur import agent, context as ctx, indexer, retriever
from arthur.llm_backend import MockBackend


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "twosum.py").write_text(
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n",
        encoding="utf-8",
    )
    # Deliberately fat and full of common words: under the old scoring, a file
    # like this beat the one the user actually named.
    (tmp_path / "utils.py").write_text(
        "\n".join(f"def helper_{i}(nums, target, seen, result):\n    return nums"
                  for i in range(40)),
        encoding="utf-8",
    )
    return str(tmp_path)


def scripted(monkeypatch, coder, critic="VERDICT: APPROVE\nREASON: fine."):
    from arthur import llm_backend, safety_gate
    monkeypatch.setattr(agent, "get_backend", lambda name=None: MockBackend(coder))
    for module in (llm_backend, safety_gate):
        monkeypatch.setattr(module, "get_critic_backend",
                            lambda name=None: MockBackend([critic]))


def collect(gen, answer=None):
    out, reply = [], None
    while True:
        try:
            event = gen.send(reply)
        except StopIteration:
            return out
        reply = None
        out.append(event)
        if event.kind == "approval_needed":
            reply = answer


# --- naming a file -----------------------------------------------------------

def test_filename_in_the_task_is_found(repo):
    index = indexer.build_index(repo)
    assert retriever.mentioned_paths("fix the bug in twosum.py", index) == ["twosum.py"]


def test_filename_wrapped_in_punctuation_is_found(repo):
    index = indexer.build_index(repo)
    for phrasing in ("update `twosum.py` please", "(twosum.py) needs a docstring",
                     "look at twosum.py, then fix it"):
        assert retriever.mentioned_paths(phrasing, index) == ["twosum.py"], phrasing


def test_bare_stem_is_found(repo):
    """Users type "the twosum file", not "twosum.py"."""
    index = indexer.build_index(repo)
    assert retriever.mentioned_paths("the twosum file is wrong", index) == ["twosum.py"]


def test_unrelated_task_names_nothing(repo):
    index = indexer.build_index(repo)
    assert retriever.mentioned_paths("write a haiku about winter", index) == []


def test_a_named_file_outranks_a_keyword_heavy_one(repo):
    """
    The scoring bug in one test. utils.py repeats `nums`, `target` and `seen`
    forty times over; twosum.py mentions each once. Relevance is not a word
    count, and a file the user named is not a ranking question at all.
    """
    index = indexer.build_index(repo)
    hits = retriever.retrieve("fix the off-by-one in twosum.py for nums and target", index)

    assert hits[0].path == "twosum.py"
    assert hits[0].mentioned is True


def test_unnamed_tasks_still_fall_back_to_keywords(repo):
    index = indexer.build_index(repo)
    hits = retriever.retrieve("something about seen and enumerate", index)
    assert [h.path for h in hits]                      # found something
    assert all(h.mentioned is False for h in hits)


def test_mentioned_flag_survives_the_context_budget(repo):
    """
    fit_context_files copies each result. A copy that dropped `mentioned`
    would strip exactly the files the task named of the marking that says so,
    and the briefing would stop pointing at the target.
    """
    index = indexer.build_index(repo)
    fitted = ctx.fit_context_files(retriever.retrieve("fix twosum.py", index))
    assert fitted[0].mentioned is True


# --- the briefing ------------------------------------------------------------

def test_briefing_names_the_target_file(repo):
    index = indexer.build_index(repo)
    fitted = ctx.fit_context_files(retriever.retrieve("fix twosum.py", index))
    briefing = agent.build_task_briefing("fix twosum.py", index, fitted)

    assert "TASK: fix twosum.py" in briefing
    assert "def two_sum" in briefing                   # the contents are there
    assert "edit_file" in briefing                     # and what to do with them


def test_briefing_admits_when_it_found_nothing(repo):
    index = indexer.build_index(repo)
    briefing = agent.build_task_briefing("write a haiku", index, [])
    assert "search_code" in briefing                   # told to go look


def test_system_prompt_holds_no_repo_state(repo):
    """
    It is the one message trimming may never drop, so it may only contain
    things that stay true. A file listing does not.
    """
    prompt = agent.build_system_prompt(repo)
    assert "def two_sum" not in prompt                 # no file contents
    assert "helper_0" not in prompt                    # no symbol listing
    assert "THOUGHT:" in prompt                        # the protocol, though


def test_the_protocol_example_filename_is_not_a_real_one(repo):
    """
    The prompt's apply_patch example used to be named twosum.py -- which was
    also, in live use, the file the user was asking about. Handing a small
    model an example filename that collides with the task is asking it to
    copy the wrong one. Placeholders must look like placeholders.
    """
    index = indexer.build_index(repo)
    prompt = agent.build_system_prompt(repo)
    for entry in index.files:
        assert entry.path not in prompt

    # Every filename the prompt does mention must be the one obvious
    # placeholder. A second plausible name is a second thing to hallucinate:
    # an example using "math_helpers.py" had the model open a turn with "the
    # repository contains files related to math helpers", in an empty repo.
    import re
    mentioned = set(re.findall(r"\b[\w./-]+\.py\b", prompt))
    assert mentioned == {"example.py"}, mentioned


# --- the regression this whole file exists for -------------------------------

def test_a_follow_up_task_sees_a_file_the_first_task_created(repo, monkeypatch):
    """
    The headline failure, end to end.

    Task one creates a file. Task two asks for a correction to it. Before this
    fix, task two was handed the system prompt built for task one -- a repo
    tree from before the file existed, and no contents -- so the model could
    not see the thing it was being asked to change, and wrote a new one.
    """
    scripted(monkeypatch, [
        'THOUGHT: creating it.\nCONFIDENCE: 0.9\nACTION: apply_patch\n'
        'PATH: fizz.py\nCONTENT:\n```\ndef fizz(n):\n    return n\n```',
        "FINAL: created fizz.py.",
    ])
    history: list[dict] = []
    collect(agent.run("create fizz.py", repo, backend="mock", auto_approve=True,
                      messages=history))

    scripted(monkeypatch, ["FINAL: nothing to do."])
    collect(agent.run("add a docstring to fizz.py", repo, backend="mock",
                      messages=history))

    briefing = history[-2]["content"]
    assert "TASK: add a docstring to fizz.py" in briefing
    assert "fizz.py" in briefing
    assert "def fizz(n):" in briefing        # the CURRENT contents, freshly read
    assert "do not create a new file" in briefing


def test_every_task_gets_its_own_briefing(repo, monkeypatch):
    scripted(monkeypatch, ["FINAL: one."])
    history: list[dict] = []
    collect(agent.run("first", repo, backend="mock", messages=history))
    scripted(monkeypatch, ["FINAL: two."])
    collect(agent.run("second", repo, backend="mock", messages=history))

    briefings = [m for m in history
                 if m["content"].startswith(ctx.BRIEFING_PREFIX)]
    assert len(briefings) == 2
    assert sum(1 for m in history if m["role"] == "system") == 1


# --- refusing to rewrite what already exists ---------------------------------

def test_apply_patch_on_an_existing_file_is_refused(repo, monkeypatch):
    scripted(monkeypatch, [
        'THOUGHT: rewriting it.\nCONFIDENCE: 0.9\nACTION: apply_patch\n'
        'PATH: twosum.py\nCONTENT:\n```\ndef two_sum(nums, target):\n    return []\n```',
        "FINAL: gave up.",
    ])
    stream = collect(agent.run("fix twosum.py", repo, backend="mock", auto_approve=True))

    kinds = [e.kind for e in stream]
    assert "diff_ready" not in kinds          # never even reached the gate
    assert "gate_decision" not in kinds

    observation = next(e for e in stream if e.kind == "observation")
    assert "already exists" in observation.text
    assert "edit_file" in observation.text
    # and the refusal hands back the file, so the next turn can build a FIND
    assert "seen[n] = i" in observation.text

    assert "seen = {}" in open(f"{repo}/twosum.py").read()   # untouched


def test_apply_patch_still_creates_new_files(repo, monkeypatch):
    scripted(monkeypatch, [
        'THOUGHT: new file.\nCONFIDENCE: 0.9\nACTION: apply_patch\n'
        'PATH: brand_new.py\nCONTENT:\n```\ndef hello():\n    return "hi"\n```',
        "FINAL: created it.",
    ])
    stream = collect(agent.run("create brand_new.py", repo, backend="mock",
                               auto_approve=True))

    assert "gate_decision" in [e.kind for e in stream]
    assert "def hello" in open(f"{repo}/brand_new.py").read()


def test_an_empty_existing_file_is_not_treated_as_a_rewrite(repo, monkeypatch):
    """A file that exists but has no content in it is nothing to protect."""
    open(f"{repo}/blank.py", "w").close()
    scripted(monkeypatch, [
        'THOUGHT: filling it in.\nCONFIDENCE: 0.9\nACTION: apply_patch\n'
        'PATH: blank.py\nCONTENT:\n```\ndef x():\n    return 1\n```',
        "FINAL: done.",
    ])
    stream = collect(agent.run("fill in blank.py", repo, backend="mock",
                               auto_approve=True))

    assert "gate_decision" in [e.kind for e in stream]
    assert "def x" in open(f"{repo}/blank.py").read()


# --- trimming must not lose the target ---------------------------------------

def test_trimming_keeps_the_newest_briefing_not_the_first():
    """
    In a long session messages[1] is the FIRST task's briefing: the wrong goal,
    and a view of the repo taken before anything was changed. Keeping it while
    dropping the current one is worse than dropping both.
    """
    filler = [{"role": "user", "content": "x" * 4000} for _ in range(20)]
    messages = (
        [{"role": "system", "content": "protocol"}]
        + [{"role": "user", "content": f"{ctx.BRIEFING_PREFIX}\nTASK: old task"}]
        + filler
        + [{"role": "user", "content": f"{ctx.BRIEFING_PREFIX}\nTASK: current task"}]
        + [{"role": "assistant", "content": "working"}]
    )
    trimmed = ctx.trim_history(messages, num_ctx=2048)
    text = "\n".join(m["content"] for m in trimmed)

    assert trimmed[0]["content"] == "protocol"
    assert "TASK: current task" in text
    assert "TASK: old task" not in text


def test_the_briefing_is_not_duplicated_when_it_is_already_recent():
    messages = (
        [{"role": "system", "content": "protocol"}]
        + [{"role": "user", "content": "x" * 4000} for _ in range(20)]
        + [{"role": "user", "content": f"{ctx.BRIEFING_PREFIX}\nTASK: current"}]
    )
    trimmed = ctx.trim_history(messages, num_ctx=2048)
    assert sum(1 for m in trimmed if m["content"].startswith(ctx.BRIEFING_PREFIX)) == 1
