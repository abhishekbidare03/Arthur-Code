"""
Builds a lightweight index of a repository: a file tree plus, for Python
files, the top-level function/class names (via `ast`, not regex -- regex
on source code lies to you the moment someone nests a def inside a string).

This is the "read directory, understand the code" half of the brief. Real
tools (Claude Code, Cursor, etc.) do a much heavier version of this with
tree-sitter across many languages plus embeddings; this is the
learn-the-mechanism version.
"""

import ast
import os
from dataclasses import dataclass, field

from . import config


@dataclass
class FileEntry:
    path: str          # relative to repo root
    size: int
    symbols: list[str] = field(default_factory=list)  # top-level def/class names


@dataclass
class RepoIndex:
    root: str
    files: list[FileEntry]

    def tree_str(self) -> str:
        lines = []
        for f in sorted(self.files, key=lambda e: e.path):
            sym = f"  [{', '.join(f.symbols)}]" if f.symbols else ""
            lines.append(f"- {f.path} ({f.size}b){sym}")
        return "\n".join(lines)


def _python_symbols(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def build_index(root: str) -> RepoIndex:
    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in config.IGNORE_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext in config.IGNORE_EXT:
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            symbols = _python_symbols(full) if ext == ".py" else []
            entries.append(FileEntry(path=rel, size=size, symbols=symbols))
    return RepoIndex(root=root, files=entries)
