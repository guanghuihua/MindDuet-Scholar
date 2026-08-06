"""Configuration kept separate from note content and credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Source notes are never written by this application."""

    project_root: Path
    notes_root: Path
    data_dir: Path
    ai_api_key: str | None = None
    ai_base_url: str = "https://api.openai.com/v1"
    ai_model: str | None = None

    @property
    def database_path(self) -> Path:
        return self.data_dir / "mindduet.sqlite3"

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "Settings":
        root = project_root or Path.cwd()
        default_notes = Path("E:/Guanghui/Notes")
        return cls(
            project_root=root,
            notes_root=_path_from_env("MINDDUET_NOTES_ROOT", default_notes),
            data_dir=_path_from_env("MINDDUET_DATA_DIR", root / ".mindduet"),
            ai_api_key=os.getenv("MINDDUET_AI_API_KEY"),
            ai_base_url=os.getenv("MINDDUET_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            ai_model=os.getenv("MINDDUET_AI_MODEL"),
        )
