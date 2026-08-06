# MindDuet Scholar

MindDuet Scholar is a local-first mathematical learning companion. It indexes
your existing notes without changing them, records learning attempts and gives
small, evidence-oriented hints instead of complete solutions.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:MINDDUET_NOTES_ROOT = "E:\Guanghui\Notes"
mindduet index
mindduet serve
```

Open `http://127.0.0.1:8000` after starting the server. The application stores
its database and generated index under `.mindduet/`, which is deliberately not
tracked by Git.

## Configuration

| Variable | Purpose |
| --- | --- |
| `MINDDUET_NOTES_ROOT` | Read-only source directory containing Markdown, TeX, PDF, and Notebook files. |
| `MINDDUET_DATA_DIR` | Directory for local SQLite state. Defaults to `.mindduet`. |
| `MINDDUET_AI_API_KEY` | Optional OpenAI-compatible API key. Keep it in an environment variable, never in the repository. |
| `MINDDUET_AI_BASE_URL` | Optional API base URL. Defaults to the OpenAI responses endpoint. |
| `MINDDUET_AI_MODEL` | Optional model identifier for tier-one hints. |

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
