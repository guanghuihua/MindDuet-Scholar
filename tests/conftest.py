from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mindduet_scholar.app import create_app
from mindduet_scholar.config import Settings


@pytest.fixture()
def notes_root(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "groups.md").write_text(
        "# Quotient groups\n\nLet $G$ be a group.\n\n## Universal property\n\nSee [exercise](exercise.md).\n\n$$G/N$$\n",
        encoding="utf-8",
    )
    (root / "algebra.tex").write_text("\\section{Homomorphisms}\n\\begin{equation} f(xy)=f(x)f(y) \\end{equation}\n", encoding="utf-8")
    return root


@pytest.fixture()
def settings(tmp_path: Path, notes_root: Path) -> Settings:
    return Settings(project_root=tmp_path, notes_root=notes_root, data_dir=tmp_path / ".mindduet")


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))
