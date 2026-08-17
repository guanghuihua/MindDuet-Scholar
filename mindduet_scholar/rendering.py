"""Safe Markdown rendering that leaves LaTeX delimiters for MathJax."""

from __future__ import annotations

import html
import re

from markdown_it import MarkdownIt
from markupsafe import Markup


MATH_PATTERN = re.compile(r"\\\[(?:.|\n)*?\\\]|\\\((?:.|\n)*?\\\)")
MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": False})


def render_math_markdown(value: object) -> Markup:
    """Render untrusted Markdown while preserving MathJax's LaTeX syntax."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        token = f"MINDDUETMATHTOKEN{len(protected)}END"
        protected.append(match.group(0))
        return token

    rendered = MARKDOWN.render(MATH_PATTERN.sub(protect, text))
    for index, math_source in enumerate(protected):
        token = f"MINDDUETMATHTOKEN{index}END"
        rendered = rendered.replace(token, html.escape(math_source, quote=False))
    return Markup(rendered)
