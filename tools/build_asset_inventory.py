"""Generate small tracked inventories for the local-only study libraries."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

ASSET_SUFFIXES = {
    ".azw",
    ".azw3",
    ".djvu",
    ".doc",
    ".docx",
    ".epub",
    ".mht",
    ".mobi",
    ".mp3",
    ".pdf",
    ".xlsx",
    ".zip",
}


def format_bytes(byte_size: int) -> str:
    value = float(byte_size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return f"{byte_size} B"


def english_skill(relative_path: Path) -> str:
    text = relative_path.as_posix().lower()
    if relative_path.suffix.lower() == ".mp3" or "听力" in text:
        return "listening"
    if "作文" in text or "writing" in text:
        return "writing"
    if "口语" in text or "speaking" in text:
        return "speaking"
    if "经济学人" in text or "reading" in text:
        return "reading"
    if "单词" in text or "vocabulary" in text or relative_path.suffix.lower() == ".epub":
        return "vocabulary"
    return "general"


def scan_assets(root: Path, library: str) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    if not root.is_dir():
        return records
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in ASSET_SUFFIXES:
            continue
        relative_path = path.relative_to(root)
        records.append(
            {
                "library": library,
                "category": relative_path.parts[0] if len(relative_path.parts) > 1 else "根目录",
                "study_area": english_skill(relative_path) if library == "english" else "math",
                "file_type": path.suffix.lower().removeprefix("."),
                "size_bytes": path.stat().st_size,
                "status": "variant-review" if "同书异版" in relative_path.parts else "active",
                "relative_path": relative_path.as_posix(),
            }
        )
    return records


def write_csv(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("library", "category", "study_area", "file_type", "size_bytes", "status", "relative_path")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def category_rows(records: list[dict[str, str | int]]) -> list[tuple[str, int, int]]:
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for record in records:
        category = str(record["category"])
        counts[category] += 1
        sizes[category] += int(record["size_bytes"])
    return [(category, counts[category], sizes[category]) for category in sorted(counts, key=str.casefold)]


def markdown_table(records: list[dict[str, str | int]]) -> str:
    lines = ["| 分类 | 文件数 | 大小 |", "| --- | ---: | ---: |"]
    lines.extend(f"| {category} | {count} | {format_bytes(size)} |" for category, count, size in category_rows(records))
    return "\n".join(lines)


def write_summary(path: Path, math_records: list[dict[str, str | int]], english_records: list[dict[str, str | int]]) -> None:
    math_bytes = sum(int(record["size_bytes"]) for record in math_records)
    english_bytes = sum(int(record["size_bytes"]) for record in english_records)
    variant_count = sum(record["status"] == "variant-review" for record in math_records)
    content = f"""# 本地学习资产台账

更新日期：{date.today().isoformat()}

本台账记录本机教材与英语学习资料的路径、类型和大小。资产文件本身由 `.gitignore` 排除，不上传 GitHub；CSV 台账可以进入 Git。

## 总览

| 资产库 | 文件数 | 大小 | 详细台账 |
| --- | ---: | ---: | --- |
| 数学教材 | {len(math_records)} | {format_bytes(math_bytes)} | [`inventory/math-assets.csv`](./inventory/math-assets.csv) |
| 英语学习 | {len(english_records)} | {format_bytes(english_bytes)} | [`inventory/english-assets.csv`](./inventory/english-assets.csv) |

数学教材中有 {variant_count} 本位于 `同书异版/`。这些文件名称相同但内容并不相同，在人工确认版本质量前全部保留。

## 数学教材分类

{markdown_table(math_records)}

## 英语资料分类

{markdown_table(english_records)}

## 更新命令

```bash
.venv/bin/python tools/build_asset_inventory.py
```

台账生成过程只读取资产元数据，不修改教材和英语资料。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    math_records = scan_assets(root / "数学教材", "math")
    english_records = scan_assets(root / "英语学习", "english")
    inventory_dir = root / "docs" / "inventory"
    write_csv(inventory_dir / "math-assets.csv", math_records)
    write_csv(inventory_dir / "english-assets.csv", english_records)
    write_summary(root / "docs" / "资产台账.md", math_records, english_records)
    print(
        f"Recorded {len(math_records)} math assets ({format_bytes(sum(int(item['size_bytes']) for item in math_records))}) "
        f"and {len(english_records)} English assets ({format_bytes(sum(int(item['size_bytes']) for item in english_records))})."
    )


if __name__ == "__main__":
    main()
