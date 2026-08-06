from __future__ import annotations

import json
from pathlib import Path

from mindduet_scholar.indexer import NotesIndexer


def test_scan_extracts_supported_source_formats(notes_root: Path) -> None:
    notebook = {
        "cells": [{"cell_type": "markdown", "source": ["# Notebook theorem\n", "A short note."]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (notes_root / "work.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    original_text = (notes_root / "groups.md").read_text(encoding="utf-8")

    documents, result = NotesIndexer().scan(notes_root)

    assert result.errors == []
    assert result.indexed == 3
    by_path = {document.relative_path: document for document in documents}
    assert by_path["groups.md"].title == "Quotient groups"
    assert by_path["groups.md"].headings == ["Quotient groups", "Universal property"]
    assert by_path["groups.md"].math_blocks >= 2
    assert by_path["algebra.tex"].title == "Homomorphisms"
    assert by_path["work.ipynb"].title == "Notebook theorem"
    assert (notes_root / "groups.md").read_text(encoding="utf-8") == original_text


def test_missing_notes_root_reports_error(tmp_path: Path) -> None:
    documents, result = NotesIndexer().scan(tmp_path / "missing")

    assert documents == []
    assert result.indexed == 0
    assert "does not exist" in result.errors[0]
