from __future__ import annotations

from mindduet_scholar.rendering import render_math_markdown


def test_math_markdown_renders_structure_and_preserves_latex() -> None:
    rendered = str(render_math_markdown("**Key idea**\n\n- first\n- second\n\n\\[G/N\\]"))

    assert "<strong>Key idea</strong>" in rendered
    assert "<ul>" in rendered
    assert r"\[G/N\]" in rendered


def test_math_markdown_does_not_render_untrusted_html() -> None:
    rendered = str(render_math_markdown('<script>alert("no")</script>'))

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
