"""Read-only extraction of the local mathematical source library."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .types import IndexedDocument

SUPPORTED_SUFFIXES = {".md": "markdown", ".markdown": "markdown", ".tex": "latex", ".ipynb": "notebook", ".pdf": "pdf"}
IGNORED_DIRECTORIES = {".git", ".obsidian", ".mindduet", ".venv", "__pycache__", "node_modules"}
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
TEX_HEADING = re.compile(r"\\(?:part|chapter|section|subsection|subsubsection)\*?\{([^}]*)\}")
TEX_LINK = re.compile(r"\\(?:ref|cite|href)\{([^}]*)\}")


@dataclass(frozen=True, slots=True)
class IndexResult:
    indexed: int
    skipped: int
    errors: list[str]


class NotesIndexer:
    """Build an index without ever opening source files for writing."""

    def scan(self, notes_root: Path) -> tuple[list[IndexedDocument], IndexResult]:
        if not notes_root.exists():
            return [], IndexResult(0, 0, [f"Notes root does not exist: {notes_root}"])
        documents: list[IndexedDocument] = []
        skipped = 0
        errors: list[str] = []
        for path in notes_root.rglob("*"):
            if any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue
            if not path.is_file():
                continue
            file_type = SUPPORTED_SUFFIXES.get(path.suffix.lower())
            if not file_type:
                skipped += 1
                continue
            try:
                documents.append(self._extract(notes_root, path, file_type))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                errors.append(f"{path.relative_to(notes_root)}: {error}")
        return documents, IndexResult(len(documents), skipped, errors)

    def _extract(self, root: Path, path: Path, file_type: str) -> IndexedDocument:
        raw = path.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        text = ""
        headings: list[str] = []
        links: list[str] = []
        math_blocks = 0
        if file_type == "markdown":
            text = raw.decode("utf-8-sig")
            headings = MARKDOWN_HEADING.findall(text)
            links = MARKDOWN_LINK.findall(text)
            math_blocks = text.count("$$") // 2 + len(re.findall(r"(?<!\\)\$[^$\n]+(?<!\\)\$", text))
        elif file_type == "latex":
            text = raw.decode("utf-8-sig")
            headings = TEX_HEADING.findall(text)
            links = TEX_LINK.findall(text)
            math_blocks = text.count("\\[") + text.count("\\begin{equation") + text.count("$$") // 2
        elif file_type == "notebook":
            notebook = json.loads(raw.decode("utf-8-sig"))
            cell_text = []
            for cell in notebook.get("cells", []):
                source = cell.get("source", [])
                cell_text.append("".join(source) if isinstance(source, list) else str(source))
            text = "\n\n".join(cell_text)
            headings = MARKDOWN_HEADING.findall(text)
            links = MARKDOWN_LINK.findall(text)
            math_blocks = text.count("$$") // 2
        # PDFs are indexed by file metadata. Full text extraction can be added later
        # without changing the source library or the database contract.
        title = headings[0] if headings else path.stem.replace("_", " ")
        stat = path.stat()
        return IndexedDocument(
            relative_path=path.relative_to(root).as_posix(),
            file_type=file_type,
            title=title[:250],
            headings=headings[:100],
            links=links[:200],
            math_blocks=math_blocks,
            text=text[:250_000],
            content_hash=content_hash,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            byte_size=stat.st_size,
        )
