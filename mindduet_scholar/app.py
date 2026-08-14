"""FastAPI application for the first MindDuet Scholar learning loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings
from .database import Database
from .indexer import NotesIndexer
from .repository import Repository
from .tutor import Tutor

PACKAGE_ROOT = Path(__file__).parent


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment(Path.cwd())
    database = Database(settings.database_path)
    database.initialize()
    repository = Repository(database)
    tutor = Tutor(settings)
    templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))

    app = FastAPI(title="MindDuet Scholar", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "static")), name="static")
    app.state.settings = settings
    app.state.repository = repository

    def render(request: Request, name: str, **context: object) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"settings": settings, **context})

    def pdf_document(document_id: int) -> tuple[dict[str, object], Path]:
        document = repository.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Source document not found")
        if document["file_type"] != "pdf":
            raise HTTPException(status_code=400, detail="Source document is not a PDF")
        pdf_path = (settings.notes_root / str(document["relative_path"])).resolve()
        notes_root = settings.notes_root.resolve()
        if not pdf_path.is_file() or not pdf_path.is_relative_to(notes_root):
            raise HTTPException(status_code=404, detail="PDF file not found")
        return document, pdf_path

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return render(request, "dashboard.html", dashboard=repository.dashboard())

    @app.post("/index", response_class=HTMLResponse)
    def index_notes(request: Request) -> HTMLResponse:
        documents, result = NotesIndexer().scan(settings.notes_root)
        if not result.errors:
            repository.replace_documents(documents)
            message = f"Indexed {result.indexed} source documents; skipped {result.skipped} unsupported files."
            status = "success"
        else:
            message = "Index was not replaced: " + " | ".join(result.errors[:3])
            status = "error"
        if request.headers.get("HX-Request"):
            return render(request, "partials/index_status.html", message=message, status=status)
        return RedirectResponse(url=f"/?notice={quote(message)}", status_code=303)

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request, q: str = "") -> HTMLResponse:
        documents = repository.search_documents(q)
        if request.headers.get("HX-Request"):
            return render(request, "partials/document_rows.html", documents=documents)
        return render(request, "library.html", documents=documents, query=q)

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    def document_detail(request: Request, document_id: int) -> HTMLResponse:
        document = repository.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Source document not found")
        return render(request, "document_detail.html", document=document)

    @app.get("/reader", response_class=HTMLResponse)
    def pdf_reader(request: Request, document_id: int) -> HTMLResponse:
        document, _ = pdf_document(document_id)
        return render(request, "pdf_reader.html", document=document)

    @app.get("/reader/documents/{document_id}/file")
    def pdf_file(document_id: int) -> FileResponse:
        document, pdf_path = pdf_document(document_id)
        return FileResponse(pdf_path, media_type="application/pdf", filename=str(document["title"]))

    @app.get("/reader/context")
    def current_pdf_context() -> JSONResponse:
        context_path = settings.data_dir / "current_pdf_context.json"
        if not context_path.exists():
            return JSONResponse({"active": False})
        try:
            return JSONResponse(json.loads(context_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return JSONResponse({"active": False, "error": "Saved PDF context is unreadable"})

    @app.post("/reader/context")
    async def save_pdf_context(request: Request) -> JSONResponse:
        payload = await request.json()
        document_id = int(payload.get("document_id", 0))
        document, _ = pdf_document(document_id)
        page = max(1, int(payload.get("page", 1)))
        selected_text = str(payload.get("selected_text", "")).strip()
        page_text = str(payload.get("page_text", "")).strip()
        context = {
            "active": True,
            "document_id": document_id,
            "title": document["title"],
            "relative_path": document["relative_path"],
            "page": page,
            "scale": payload.get("scale"),
            "selected_text": selected_text[:12000],
            "page_text": page_text[:50000],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "instruction": "Explain this PDF context in Chinese with translation, mathematical meaning, proof dependencies, hidden steps, and one understanding-check question.",
        }
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "current_pdf_context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse(context)

    @app.get("/sessions/new", response_class=HTMLResponse)
    def new_session(request: Request, document_id: int | None = None) -> HTMLResponse:
        return render(request, "session_new.html", documents=repository.search_documents(), selected_document_id=document_id)

    @app.post("/sessions")
    def create_session(goal: str = Form(...), document_id: int | None = Form(None), source_anchor: str | None = Form(None)) -> RedirectResponse:
        cleaned_goal = goal.strip()
        if not cleaned_goal:
            return RedirectResponse(url="/sessions/new?notice=Goal+is+required", status_code=303)
        session_id = repository.start_session(cleaned_goal, document_id, source_anchor.strip() if source_anchor else None)
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @app.get("/sessions", response_class=HTMLResponse)
    def sessions(request: Request) -> HTMLResponse:
        return render(request, "sessions.html", sessions=repository.list_sessions())

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    def session_detail(request: Request, session_id: int) -> HTMLResponse:
        session = repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        return render(request, "session_detail.html", session=session)

    @app.post("/sessions/{session_id}/attempts")
    def add_attempt(session_id: int, body: str = Form(...)) -> RedirectResponse:
        session = repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        cleaned_body = body.strip()
        if cleaned_body:
            findings = tutor.diagnose(cleaned_body)
            repository.add_attempt(session_id, cleaned_body, findings[0][1] if findings else None, findings)
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @app.post("/sessions/{session_id}/hints")
    def request_hint(session_id: int) -> RedirectResponse:
        session = repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        if not session["attempts"]:
            return RedirectResponse(url=f"/sessions/{session_id}?notice=Write+an+attempt+before+asking+for+a+hint", status_code=303)
        latest_attempt = session["attempts"][-1]
        hint = tutor.tier_one_hint(session["goal"], latest_attempt["body"])
        repository.add_hint(session_id, latest_attempt["id"], hint.text, hint.provider)
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @app.post("/sessions/{session_id}/reviews")
    def schedule_review(session_id: int) -> RedirectResponse:
        session = repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Learning session not found")
        repository.schedule_reviews(session["goal"])
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @app.post("/sessions/{session_id}/complete")
    def complete_session(session_id: int) -> RedirectResponse:
        repository.complete_session(session_id)
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @app.get("/reports", response_class=HTMLResponse)
    def reports(request: Request) -> HTMLResponse:
        return render(request, "reports.html", report=repository.report())

    @app.post("/reviews/{review_id}/complete")
    def complete_review(review_id: int, evidence: str = Form("")) -> RedirectResponse:
        repository.complete_review(review_id, evidence.strip())
        return RedirectResponse(url="/reports", status_code=303)

    return app
