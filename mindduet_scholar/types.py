"""Small domain objects shared by the indexer, repository and web layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    relative_path: str
    file_type: str
    title: str
    headings: list[str]
    links: list[str]
    math_blocks: int
    text: str
    content_hash: str
    modified_at: datetime
    byte_size: int


@dataclass(frozen=True, slots=True)
class Review:
    id: int
    concept: str
    due_on: date
    status: str
    evidence: str | None
