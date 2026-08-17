# MindDuet Scholar

MindDuet Scholar is a local-first mathematical learning companion. It indexes
the notes and study materials kept in this repository, records learning
attempts and gives small, evidence-oriented hints instead of complete
solutions.

中文操作手册见 [`使用说明.md`](./使用说明.md)。

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
mindduet index
mindduet serve
```

Open `http://127.0.0.1:8000` after starting the server. The application stores
its database and generated index under `.mindduet/`, which is deliberately not
tracked by Git. By default, `mindduet index` scans the project root, so the
code and personal study library can live together in one repository.

## Configuration

| Variable | Purpose |
| --- | --- |
| `MINDDUET_NOTES_ROOT` | Optional override for the source directory containing Markdown, TeX, PDF, and Notebook files. Defaults to the project root. |
| `MINDDUET_DATA_DIR` | Directory for local SQLite state. Defaults to `.mindduet`. |
| `MINDDUET_AI_API_KEY` | Optional OpenAI-compatible API key. Keep it in an environment variable, never in the repository. |
| `MINDDUET_AI_BASE_URL` | Optional API base URL. Defaults to the OpenAI responses endpoint. |
| `MINDDUET_AI_MODEL` | Optional model identifier for tier-one hints. |
| `MINDDUET_CODEX_MODEL` | Optional model override for the read-only Codex assistant embedded in the PDF reader. |

Reader Codex turns use ephemeral threads and keep their visible conversation
history in `.mindduet/mindduet.sqlite3`, so they do not mix with Codex records
created directly in the project workspace.

Learning sessions have two modes. **Ask & Understand** keeps a continuous,
direct Codex dialogue for exploring notation, definitions, and connections;
its messages are restored after a refresh. **Practice & Prove** keeps the
attempt, tier-one hint, diagnosis, and review workflow. Session chat also uses
ephemeral Codex threads, while its durable history stays only in MindDuet's
local SQLite database. Completed sessions can be reopened without losing their
conversation or learning evidence.

If no AI credentials are configured, the application provides a constrained
local tier-one prompt. This keeps the learning workflow usable offline and
never exposes a complete proof before the learner has attempted one.

## Development

```powershell
pytest
```

The MVP includes an indexer, a SQLite learning record, learning sessions,
attempt history, tier-one hints, misconception tracking, spaced reviews and a
weekly-report view. See `MINDDUET_SCHOLAR.md` for the product specification.

## Study library

This repository intentionally keeps the learning application and the personal
study library together. The current library is organized around:

- `ATVault/`: algebraic topology notes, problem attempts and related reading.
- `数学教材/`: textbooks and reference material used by the local indexer.
- `CTRW/`: graduation-thesis code, simulations, drafts and presentation files.
- `guanghui的Hamilton系统论文/`: Hamiltonian and symplectic geometry paper
  materials.

Generated indexes, local databases, credentials, Python caches, LaTeX build
products and machine-specific editor state stay outside Git.
