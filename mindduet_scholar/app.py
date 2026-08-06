"""FastAPI application for the first MindDuet Scholar learning loop."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
