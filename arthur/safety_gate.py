"""
The signature piece of this project: no destructive tool call reaches disk
without passing through a confidence gate first. This is the same pattern
as the entropy-based safety gate used for the offline Phi-3 RAG pipeline
and the OSCC/Docket confidence-gating work -- generalized here into a
reusable two-agent check:

    Coder Agent  --proposes a change + self-reported confidence-->
    Critic Agent --independently reviews the diff, votes APPROVE/REJECT-->
    Safety Gate  --combines both into one score-->
        score >= THRESHOLD  -> auto-apply
        score <  THRESHOLD  -> pause and ask the human

Two independent, differently-biased signals (the actor's own self-estimate,
and a separate reviewer that never saw the actor's reasoning) catch more
than either alone -- the same reason ensembling beats a single model.
"""

import re
from dataclasses import dataclass, field

from . import config, patcher
from .llm_backend import LLMBackend, get_critic_backend


@dataclass
class GateResult:
    approved: bool
    score: float
    self_confidence: float
    critic_verdict: str
    critic_reason: str
    needs_human: bool
    structural_warnings: list[str] = field(default_factory=list)
    blocked_structurally: bool = False


def _parse_self_confidence(coder_response_text: str) -> float:
    m = re.search(r"CONFIDENCE:\s*([0-9.]+)", coder_response_text)
    return float(m.group(1)) if m else 0.5  # unknown -> assume mediocre, don't auto-trust


def _run_critic(task: str, diff_text: str, critic_backend: LLMBackend) -> tuple[str, str]:
    system = (
        "You are a careful, independent code reviewer. You did not write this "
        "change and have no stake in it being accepted. Given the original "
        "task and the proposed diff, decide if it is correct and safe to apply.\n"
        "Respond in exactly this format:\n"
        "VERDICT: APPROVE or REJECT\n"
        "REASON: <one sentence>"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK: {task}\n\nPROPOSED DIFF:\n{diff_text}"},
    ]
    resp = critic_backend.chat(messages)
    verdict_m = re.search(r"VERDICT:\s*(APPROVE|REJECT)", resp.text, re.IGNORECASE)
    reason_m = re.search(r"REASON:\s*(.+)", resp.text)
    verdict = verdict_m.group(1).upper() if verdict_m else "REJECT"
    reason = reason_m.group(1).strip() if reason_m else "(no reason parsed)"
    return verdict, reason


def _structural_check(old_content: str, new_content: str, path: str) -> tuple[list[str], bool]:
    """
    The signal that isn't an opinion. Returns (warnings, hard_block).

    Measured, not guessed: on the first live local eval, a 3B asked to add one
    docstring returned a file with two top-level functions deleted, and the LLM
    critic approved it -- correctly observing that the docstring had been added,
    and never noticing what was gone. Symbol removal and wholesale shrinkage
    are facts you can check with `ast`, so we check them.
    """
    shape = patcher.analyze(old_content, new_content, path)
    warnings, block = [], False

    if shape.parse_broken:
        warnings.append("the patched file is no longer valid Python")
        block = True

    if shape.removed_symbols:
        names = ", ".join(shape.removed_symbols)
        warnings.append(f"removes existing definitions: {names}")
        block = True

    if shape.gutted_symbols:
        names = ", ".join(shape.gutted_symbols)
        warnings.append(
            f"replaces the body of {names} with nothing but a docstring"
        )
        block = True

    if shape.hollow_additions:
        names = ", ".join(shape.hollow_additions)
        warnings.append(
            f"{names} has a signature and a docstring but no code in the body"
        )
        block = True

    if shape.shrink_ratio >= config.MAX_SHRINK_RATIO:
        warnings.append(
            f"deletes {shape.shrink_ratio:.0%} of the file "
            f"({shape.old_lines} lines -> {shape.new_lines})"
        )
        block = True

    return warnings, block


def evaluate(task: str, coder_response_text: str, diff_text: str,
             critic_backend: LLMBackend | None = None,
             backend_name: str | None = None,
             old_content: str | None = None,
             new_content: str | None = None,
             path: str = "") -> GateResult:
    self_conf = _parse_self_confidence(coder_response_text)

    if critic_backend is None:
        # A fresh backend instance per evaluation: the critic must not inherit
        # the coder's conversation history, or it stops being independent.
        critic_backend = get_critic_backend(backend_name)
    try:
        verdict, reason = _run_critic(task, diff_text, critic_backend)
    except Exception as e:
        # A critic that failed to answer is not an approval. Fail closed:
        # no verdict -> no auto-apply -> the human decides.
        verdict, reason = "REJECT", f"critic call failed: {e}"

    critic_score = 1.0 if verdict == "APPROVE" else 0.0
    combined = 0.5 * self_conf + 0.5 * critic_score

    warnings, blocked = [], False
    if old_content is not None and new_content is not None:
        warnings, blocked = _structural_check(old_content, new_content, path)

    # A structural block overrides the score outright rather than nudging it.
    # Two models agreeing that deleting half a file is fine is still two models
    # being wrong, and no weighting of their opinions should be able to clear
    # it. This is the one veto in the system that no amount of confidence wins.
    needs_human = blocked or combined < config.CONFIDENCE_THRESHOLD

    return GateResult(
        approved=not needs_human,
        score=combined,
        self_confidence=self_conf,
        critic_verdict=verdict,
        critic_reason=reason,
        needs_human=needs_human,
        structural_warnings=warnings,
        blocked_structurally=blocked,
    )
