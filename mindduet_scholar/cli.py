"""Command line entry points for local indexing and serving."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import Settings
from .database import Database
from .indexer import NotesIndexer
from .repository import Repository


def main() -> None:
    parser = argparse.ArgumentParser(prog="mindduet")
    subcommands = parser.add_subparsers(dest="command", required=True)
    index_command = subcommands.add_parser("index", help="Read and index source notes without modifying them.")
    index_command.add_argument("--notes-root", type=Path, help="Override MINDDUET_NOTES_ROOT for this run.")
    serve_command = subcommands.add_parser("serve", help="Run the local web application.")
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings.from_environment(Path.cwd())
    if args.command == "index":
        notes_root = args.notes_root or settings.notes_root
        database = Database(settings.database_path)
        database.initialize()
        documents, result = NotesIndexer().scan(notes_root)
        if result.errors:
            for error in result.errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        Repository(database).replace_documents(documents)
        print(f"Indexed {result.indexed} documents; skipped {result.skipped} unsupported files.")
        return
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
