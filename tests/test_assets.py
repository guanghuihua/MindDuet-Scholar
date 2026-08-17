from __future__ import annotations

from pathlib import Path

from mindduet_scholar.assets import build_english_catalog, classify_english_asset, resolve_english_asset


def test_english_assets_are_classified_by_skill(tmp_path: Path) -> None:
    root = tmp_path / "英语学习"
    listening = root / "剑桥雅思听力音频" / "test.mp3"
    writing = root / "IELTS" / "作文素材" / "task.pdf"
    reading = root / "经济学人" / "issue.pdf"
    for path in (listening, writing, reading):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"study material")

    catalog = build_english_catalog(tmp_path)

    assert catalog["total_count"] == 3
    assert {item["skill"] for item in catalog["assets"]} == {"listening", "writing", "reading"}
    assert classify_english_asset(Path("IELTS/口语素材/topic.pdf")) == "speaking"


def test_english_asset_resolution_stays_inside_library(tmp_path: Path) -> None:
    root = tmp_path / "英语学习"
    root.mkdir()
    audio = root / "practice.mp3"
    audio.write_bytes(b"audio")
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"private")

    assert resolve_english_asset(tmp_path, "practice.mp3") == audio.resolve()
    assert resolve_english_asset(tmp_path, "../private.pdf") is None
    assert resolve_english_asset(tmp_path, "missing.mp3") is None
