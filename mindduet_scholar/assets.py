"""Read-only catalog helpers for local study assets."""

from __future__ import annotations

from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any

ENGLISH_DIRECTORY = "英语学习"
ENGLISH_ASSET_SUFFIXES = {".docx", ".epub", ".mp3", ".pdf"}

ENGLISH_SKILLS = (
    {"key": "all", "label": "All materials", "goal": "Plan my next IELTS study session"},
    {"key": "listening", "label": "Listening", "goal": "Complete one IELTS listening practice and diagnose my errors"},
    {"key": "reading", "label": "Reading", "goal": "Read and analyse one IELTS-level English article"},
    {"key": "writing", "label": "Writing", "goal": "Write and revise one IELTS response"},
    {"key": "speaking", "label": "Speaking", "goal": "Practise one IELTS speaking topic and improve my response"},
    {"key": "vocabulary", "label": "Vocabulary", "goal": "Review and actively use a focused IELTS vocabulary set"},
    {"key": "general", "label": "General", "goal": "Organise one English study task"},
)


def format_bytes(byte_size: int) -> str:
    value = float(byte_size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{byte_size} B"


def classify_english_asset(relative_path: Path) -> str:
    path_text = relative_path.as_posix().lower()
    if relative_path.suffix.lower() == ".mp3" or "听力" in path_text:
        return "listening"
    if "作文" in path_text or "writing" in path_text:
        return "writing"
    if "口语" in path_text or "speaking" in path_text:
        return "speaking"
    if "经济学人" in path_text or "reading" in path_text:
        return "reading"
    if "单词" in path_text or "vocabulary" in path_text or relative_path.suffix.lower() == ".epub":
        return "vocabulary"
    return "general"


def build_english_catalog(
    project_root: Path,
    skill: str = "all",
    query: str = "",
    page: int = 1,
    page_size: int = 60,
) -> dict[str, Any]:
    root = project_root / ENGLISH_DIRECTORY
    allowed_skills = {item["key"] for item in ENGLISH_SKILLS}
    selected_skill = skill if skill in allowed_skills else "all"
    normalized_query = query.strip().casefold()
    items: list[dict[str, Any]] = []
    skill_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    group_bytes: Counter[str] = Counter()
    total_bytes = 0
    incomplete_files = 0

    if root.is_dir():
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in ENGLISH_ASSET_SUFFIXES:
                incomplete_files += 1
                continue
            relative_path = path.relative_to(root)
            item_skill = classify_english_asset(relative_path)
            byte_size = path.stat().st_size
            group = relative_path.parts[0] if len(relative_path.parts) > 1 else "根目录"
            total_bytes += byte_size
            skill_counts[item_skill] += 1
            group_counts[group] += 1
            group_bytes[group] += byte_size
            if selected_skill != "all" and item_skill != selected_skill:
                continue
            if normalized_query and normalized_query not in relative_path.as_posix().casefold():
                continue
            items.append(
                {
                    "name": path.name,
                    "relative_path": relative_path.as_posix(),
                    "project_path": f"{ENGLISH_DIRECTORY}/{relative_path.as_posix()}",
                    "skill": item_skill,
                    "kind": suffix.removeprefix(".").upper(),
                    "byte_size": byte_size,
                    "display_size": format_bytes(byte_size),
                }
            )

    groups = [
        {"name": name, "count": group_counts[name], "display_size": format_bytes(group_bytes[name])}
        for name in sorted(group_counts, key=str.casefold)
    ]
    skills = [
        {**item, "count": sum(skill_counts.values()) if item["key"] == "all" else skill_counts[item["key"]]}
        for item in ENGLISH_SKILLS
    ]
    matching_count = len(items)
    page_count = max(1, ceil(matching_count / page_size))
    selected_page = min(max(1, page), page_count)
    first_index = (selected_page - 1) * page_size
    last_index = min(first_index + page_size, matching_count)
    return {
        "available": root.is_dir(),
        "root": root,
        "selected_skill": selected_skill,
        "query": query,
        "assets": items[first_index:last_index],
        "matching_count": matching_count,
        "page": selected_page,
        "page_count": page_count,
        "first_item": first_index + 1 if matching_count else 0,
        "last_item": last_index,
        "total_count": sum(skill_counts.values()),
        "total_bytes": total_bytes,
        "display_size": format_bytes(total_bytes),
        "incomplete_files": incomplete_files,
        "skills": skills,
        "groups": groups,
    }


def resolve_english_asset(project_root: Path, relative_path: str) -> Path | None:
    root = (project_root / ENGLISH_DIRECTORY).resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in ENGLISH_ASSET_SUFFIXES:
        return None
    return candidate
