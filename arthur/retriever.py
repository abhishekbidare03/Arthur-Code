"""
Decides which files the model gets to see for a given task.

Two mechanisms, and the first one matters far more than the second:

  1. MENTION. If the task names a file that exists, that file is the task.
     "fix the off-by-one in twosum.py" needs exactly one file, and no amount
     of keyword scoring should be able to outrank it. Named files are pinned
     to the front unconditionally.

  2. OVERLAP. When nothing is named -- "fix the divide-by-zero bug" -- fall
     back to keyword overlap between the task and each file's path, symbols
     and contents.

Mechanism 1 exists because of a real failure: asked to correct a file it had
just written, the agent rebuilt it from scratch instead. It had never been
shown the file. Ranking by keywords alone let a large, vaguely-related file
outscore the one the user had named outright, since the old scoring counted
every repeated token in a file's body -- which is a length contest, not a
relevance one.

The natural upgrade to mechanism 2 -- and a good "what would you improve"
answer -- is embedding each file/chunk and doing cosine-similarity retrieval,
exactly like the ChromaDB RAG pipeline in the offline Phi-3 project. Keyword
overlap is the right MVP baseline to compare that upgrade against. Mechanism 1
would stay regardless: an explicitly named file is not a similarity question.
"""

import os
import re
from dataclasses import dataclass

from . import config
from .indexer import RepoIndex


@dataclass
class RetrievedFile:
    path: str
    score: int
    snippet: str
    mentioned: bool = False       # the task named this file outright


# A named file outranks anything overlap scoring can produce.
MENTION_SCORE = 10_000

# Where a term matched matters more than how often. A term in the filename or
# in a def/class name is a strong signal; the same term in the body is weak,
# and counting body repeats just favours whichever file is longest.
PATH_WEIGHT = 8
SYMBOL_WEIGHT = 5
CONTENT_WEIGHT = 1


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def mentioned_paths(task: str, index: RepoIndex) -> list[str]:
    """
    Indexed files the task refers to by name, best match first.

    Three spellings are accepted, in descending order of confidence: the full
    relative path ("demo_repo/twosum.py"), the bare filename ("twosum.py"),
    and the stem alone ("twosum") when it is distinctive enough to be worth
    trusting. The stem rule is what catches "the two sum file needs a
    docstring" -- users rarely type the extension.
    """
    haystack = _norm(task)
    # Punctuation around a filename shouldn't hide it: "in `twosum.py`," and
    # "(twosum.py)" are both just twosum.py.
    padded = re.sub(r"[^a-z0-9_./-]+", " ", haystack)

    ranked: list[tuple[int, str]] = []
    for entry in index.files:
        rel = _norm(entry.path)
        base = os.path.basename(rel)
        stem = os.path.splitext(base)[0]

        if rel in padded:
            ranked.append((0, entry.path))
        elif re.search(rf"(^|[\s/]){re.escape(base)}($|[\s/])", padded):
            ranked.append((1, entry.path))
        elif len(stem) >= 4 and re.search(rf"(^|[\s/_-]){re.escape(stem)}($|[\s/_.-])", padded):
            ranked.append((2, entry.path))

    ranked.sort()
    return [path for _, path in ranked]


def _read(index: RepoIndex, rel_path: str) -> str | None:
    try:
        with open(os.path.join(index.root, rel_path), "r",
                  encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return None


def retrieve(task: str, index: RepoIndex,
             top_k: int = config.MAX_CONTEXT_FILES) -> list[RetrievedFile]:
    query_terms = set(_tokenize(task))
    named = mentioned_paths(task, index)
    named_set = set(named)

    scored: list[RetrievedFile] = []

    # Pinned first, in the order they were matched, so the most confident
    # spelling leads and gets the largest share of the context budget.
    for rank, rel in enumerate(named):
        content = _read(index, rel)
        if content is None:
            continue
        scored.append(RetrievedFile(
            path=rel,
            score=MENTION_SCORE - rank,
            snippet=content[: config.MAX_FILE_CHARS],
            mentioned=True,
        ))

    for entry in index.files:
        if entry.path in named_set:
            continue
        content = _read(index, entry.path)
        if content is None:
            continue

        path_terms = set(_tokenize(entry.path))
        symbol_terms = {s.lower() for s in entry.symbols}
        content_terms = set(_tokenize(content))

        score = (
            PATH_WEIGHT * len(path_terms & query_terms)
            + SYMBOL_WEIGHT * len(symbol_terms & query_terms)
            + CONTENT_WEIGHT * len(content_terms & query_terms)
        )

        if score > 0:
            scored.append(RetrievedFile(
                path=entry.path,
                score=score,
                snippet=content[: config.MAX_FILE_CHARS],
            ))

    # Stable within a score so results don't shuffle between identical runs.
    scored.sort(key=lambda r: (-r.score, r.path))
    return scored[:top_k]
