from contextlib import asynccontextmanager, contextmanager
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from threading import Event, Lock, Thread
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from desk.rag import GeminiProvider, ProviderError, RagEngine, documents_fingerprint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "clothing"


def now():
    return datetime.now(timezone.utc).isoformat()


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DocumentInput(Input):
    title: Annotated[str, Field(min_length=1, max_length=160)]
    topic: Annotated[str, Field(min_length=1, max_length=30)]
    body: Annotated[str, Field(min_length=20, max_length=20000)]


class DocumentUpdate(DocumentInput):
    version: Annotated[int, Field(ge=1)]


class TicketInput(Input):
    customer_id: Annotated[str, Field(min_length=1, max_length=100)]
    subject: Annotated[str, Field(min_length=1, max_length=160)]
    body: Annotated[str, Field(min_length=5, max_length=5000)]


class RevisionInput(Input):
    revision: Annotated[int, Field(ge=1)]


class DraftInput(RevisionInput):
    guidance: Annotated[str, Field(max_length=1000)] = ""
    current_answer: Annotated[str, Field(max_length=6000)] = ""


class ResolveInput(RevisionInput):
    action: Literal["approve", "escalate"]
    answer: Annotated[str, Field(min_length=1, max_length=6000)]


class LimitBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 128 * 1024:
                return await JSONResponse({"detail": "요청이 너무 큽니다."}, 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        return await self.app(scope, bounded_receive, send)


def create_app(data_dir=None, provider_factory=None, seed=True, auto_draft=True):
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    key_file = data_dir / "gemini-key"
    engine = RagEngine(data_dir / "vectors")
    operation_lock = Lock()
    wake_worker = Event()
    stop_worker = Event()

    @contextmanager
    def database():
        connection = sqlite3.connect(data_dir / "desk.sqlite3", timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    with database() as db:
        for table in ("documents", "customers", "tickets"):
            db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if seed and not db.execute("SELECT 1 FROM metadata WHERE key='seeded'").fetchone():
            for table in ("documents", "customers"):
                for item in json.loads((ROOT / "demo" / f"{table}.json").read_text(encoding="utf-8")):
                    db.execute(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", (item["id"], json.dumps(item, ensure_ascii=False)))
            for item in reversed(json.loads((ROOT / "demo" / "tickets.json").read_text(encoding="utf-8"))):
                ticket = new_ticket(item)
                db.execute("INSERT INTO tickets VALUES (?, ?)", (ticket["id"], json.dumps(ticket, ensure_ascii=False)))
            db.execute("INSERT INTO metadata VALUES ('seeded', '1')")
        for row in db.execute("SELECT id, body FROM tickets").fetchall():
            ticket = json.loads(row["body"])
            finished = ticket.get("draft") is not None or ticket["status"] in {"resolved", "escalated"}
            job = ticket.get("auto_draft", {"state": "queued", "error": ""})
            if finished:
                job = {"state": "completed", "error": ""}
            elif job["state"] == "running":
                # A crash may occur after the paid request; do not silently repeat it.
                job = {"state": "failed", "error": "서버가 종료되어 자동 작성이 중단됐습니다. 재시도해 주세요."}
            ticket["auto_draft"] = job
            db.execute("UPDATE tickets SET body=? WHERE id=?", (json.dumps(ticket, ensure_ascii=False), ticket["id"]))

    @asynccontextmanager
    async def lifespan(app):
        worker = Thread(target=auto_worker, name="inquiry-first-drafts", daemon=True) if auto_draft else None
        if worker:
            worker.start()
            wake_worker.set()
        try:
            yield
        finally:
            stop_worker.set()
            wake_worker.set()
            if worker:
                await asyncio.to_thread(worker.join)
            engine.close()

    application = FastAPI(title="의류 FAQ 문의데스크 API", version="0.2.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    application.state.engine = engine
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    application.add_middleware(LimitBody)

    @application.middleware("http")
    async def local_only(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "다른 사이트에서 보낸 변경 요청은 허용하지 않습니다."}, 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        return response

    @application.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({"detail": str(exc)}, 502)

    def read_key():
        try:
            return key_file.read_text(encoding="utf-8").strip() if key_file.exists() else os.environ.get("GEMINI_API_KEY", "").strip()
        except OSError:
            raise HTTPException(503, "저장된 키를 읽지 못했습니다. 키를 다시 저장해 주세요.") from None

    def provider():
        if provider_factory:
            return provider_factory()
        key = read_key()
        if not key:
            raise HTTPException(409, "설정에서 Gemini 키를 저장한 뒤 FAQ 검색 색인을 만들어 주세요.")
        return GeminiProvider(key)

    def rows(table, db=None):
        if db is not None:
            return [json.loads(row["body"]) for row in db.execute(f"SELECT body FROM {table} ORDER BY rowid DESC")]
        with database() as connection:
            return rows(table, connection)

    def find(table, item_id, db):
        row = db.execute(f"SELECT body FROM {table} WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(404, "항목을 찾을 수 없습니다.")
        return json.loads(row["body"])

    def write(table, item, db):
        db.execute(f"UPDATE {table} SET body=? WHERE id=?", (json.dumps(item, ensure_ascii=False), item["id"]))

    def ticket_detail(ticket_id, db):
        ticket = find("tickets", ticket_id, db)
        ticket["customer"] = find("customers", ticket["customer_id"], db)
        return ticket

    def check_revision(ticket, revision):
        if ticket["revision"] != revision:
            raise HTTPException(409, "다른 작업에서 문의가 변경됐습니다. 새로고침한 뒤 다시 확인해 주세요.")
        if ticket["status"] in {"resolved", "escalated"}:
            raise HTTPException(409, "이미 처리한 문의입니다.")

    @contextmanager
    def operation(notify=True):
        # ponytail: one process and one AI job at a time; use a shared queue for multiple server workers.
        if not operation_lock.acquire(blocking=False):
            raise HTTPException(409, "AI 작업이 진행 중입니다. 완료된 뒤 다시 시도해 주세요.")
        try:
            yield
        finally:
            operation_lock.release()
            if notify:
                wake_worker.set()

    @application.get("/api/status")
    def status():
        documents = rows("documents")
        meta = engine.status()
        configured = bool(read_key()) or provider_factory is not None
        indexed = bool(meta.get("indexed")) and meta.get("index_fingerprint") == documents_fingerprint(documents)
        items = rows("tickets")
        jobs = [t.get("auto_draft", {}).get("state") for t in items if t["draft"] is None and t["status"] not in {"resolved", "escalated"}]
        queue = {state: jobs.count(state) for state in ("queued", "running", "failed")}
        queue["blocked_reason"] = "설정에서 Gemini 키를 저장해 주세요." if not configured else "FAQ 검색 색인을 갱신하면 자동 작성을 시작합니다." if not indexed else ""
        return {**meta, "configured": configured, "indexed": indexed, "document_count": len(documents), "ticket_count": len(items), "model": "gemini-2.5-flash-lite", "embedding_model": "gemini-embedding-001", "auto_draft": {"enabled": auto_draft, **queue}}

    @application.put("/api/settings/key")
    async def save_key(request: Request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise HTTPException(422, "키 입력 형식이 올바르지 않습니다.") from None
        if not isinstance(body, dict) or set(body) != {"api_key"} or not isinstance(body["api_key"], str):
            raise HTTPException(422, "키 입력 형식이 올바르지 않습니다.")
        key = body["api_key"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{20,200}", key):
            raise HTTPException(422, "키 전체를 입력해 주세요. 공백이나 줄바꿈이 포함될 수 없습니다.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=data_dir, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(key)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, key_file)
        except OSError:
            raise HTTPException(503, "키를 저장하지 못했습니다. 폴더의 쓰기 권한을 확인해 주세요.") from None
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        wake_worker.set()
        return {"saved": True}

    @application.get("/api/customers")
    def customers():
        return {"customers": rows("customers")}

    @application.get("/api/documents")
    def documents():
        return {"documents": rows("documents")}

    def reject_duplicate(payload, documents, current_id=None):
        if any(document["id"] != current_id and document["body"].strip() == payload.body.strip() for document in documents):
            raise HTTPException(409, "같은 내용의 FAQ가 이미 있습니다.")

    @application.post("/api/documents", status_code=201)
    def add_document(payload: DocumentInput):
        with database() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = rows("documents", db)
            if len(existing) >= 100:
                raise HTTPException(422, "이 데모는 FAQ를 최대 100개까지 지원합니다.")
            reject_duplicate(payload, existing)
            document = {**payload.model_dump(), "id": str(uuid4()), "version": 1}
            db.execute("INSERT INTO documents VALUES (?, ?)", (document["id"], json.dumps(document, ensure_ascii=False)))
        return document

    @application.put("/api/documents/{document_id}")
    def update_document(document_id: str, payload: DocumentUpdate):
        with database() as db:
            db.execute("BEGIN IMMEDIATE")
            document = find("documents", document_id, db)
            if document["version"] != payload.version:
                raise HTTPException(409, "문서가 변경됐습니다. 다시 열어 확인해 주세요.")
            reject_duplicate(payload, rows("documents", db), document_id)
            document.update(payload.model_dump())
            document["version"] += 1
            write("documents", document, db)
        return document

    @application.post("/api/index")
    def rebuild_index():
        with operation():
            result = engine.index(rows("documents"), provider())
        return result

    @application.get("/api/tickets")
    def tickets():
        return {"tickets": rows("tickets")}

    @application.post("/api/tickets", status_code=201)
    def add_ticket(payload: TicketInput):
        with database() as db:
            find("customers", payload.customer_id, db)
            ticket = new_ticket(payload.model_dump())
            db.execute("INSERT INTO tickets VALUES (?, ?)", (ticket["id"], json.dumps(ticket, ensure_ascii=False)))
            result = ticket_detail(ticket["id"], db)
        wake_worker.set()
        return result

    @application.get("/api/tickets/{ticket_id}")
    def get_ticket(ticket_id: str):
        with database() as db:
            return ticket_detail(ticket_id, db)

    @application.post("/api/tickets/{ticket_id}/draft")
    def generate_draft(ticket_id: str, payload: DraftInput):
        with operation():
            if auto_draft:
                with database() as db:
                    ticket = find("tickets", ticket_id, db)
                    if ticket["draft"] is None:
                        raise HTTPException(409, "1차 초안은 자동으로 작성됩니다. 작성 실패 시 재시도해 주세요.")
            return create_draft(ticket_id, payload)

    def create_draft(ticket_id, payload):
        with database() as db:
            ticket = ticket_detail(ticket_id, db)
            check_revision(ticket, payload.revision)
        fingerprint = documents_fingerprint(rows("documents"))
        if engine.status().get("index_fingerprint") != fingerprint:
            raise HTTPException(409, "FAQ가 아직 색인되지 않았거나 변경됐습니다. FAQ 자료에서 색인을 갱신해 주세요.")
        draft = engine.draft(ticket, ticket["customer"], provider(), guidance=payload.guidance, current_answer=payload.current_answer)
        draft["index_fingerprint"] = fingerprint
        with database() as db:
            db.execute("BEGIN IMMEDIATE")
            current = find("tickets", ticket_id, db)
            check_revision(current, payload.revision)
            if documents_fingerprint(rows("documents", db)) != fingerprint:
                raise HTTPException(409, "분석 중 정책이 변경되어 초안을 저장하지 않았습니다. 색인을 갱신해 주세요.")
            regenerated = current["draft"] is not None
            current.update(draft=draft, status="review" if draft["action"] == "answer" else "needs_info", updated_at=now(), revision=current["revision"] + 1, auto_draft={"state": "completed", "error": ""})
            event = {"at": now(), "event": "regenerated" if regenerated else "drafted", "detail": ("AI 초안 재생성: " if regenerated else "AI 초안 생성: ") + draft["action"]}
            if regenerated:
                event["guidance"] = payload.guidance
                if payload.guidance:
                    event["detail"] += " / 요청 방향: " + payload.guidance
            current["history"].append(event)
            write("tickets", current, db)
            return ticket_detail(ticket_id, db)

    @application.post("/api/tickets/{ticket_id}/retry-draft")
    def retry_draft(ticket_id: str, payload: RevisionInput):
        with database() as db:
            db.execute("BEGIN IMMEDIATE")
            ticket = find("tickets", ticket_id, db)
            check_revision(ticket, payload.revision)
            if ticket["draft"] is not None or ticket.get("auto_draft", {}).get("state") != "failed":
                raise HTTPException(409, "자동 작성에 실패한 문의만 재시도할 수 있습니다.")
            ticket["auto_draft"] = {"state": "queued", "error": ""}
            ticket["updated_at"] = now()
            ticket["history"].append({"at": now(), "event": "auto_draft_retry", "detail": "자동 초안 재시도 요청"})
            write("tickets", ticket, db)
            result = ticket_detail(ticket_id, db)
        wake_worker.set()
        return result

    def process_next_draft():
        try:
            # Do not compete with key setup or the first index build while unready.
            if not provider_factory and not read_key():
                return False
            if engine.status().get("index_fingerprint") != documents_fingerprint(rows("documents")):
                return False
            with operation(notify=False):
                if not provider_factory and not read_key():
                    return False
                if engine.status().get("index_fingerprint") != documents_fingerprint(rows("documents")):
                    return False
                with database() as db:
                    db.execute("BEGIN IMMEDIATE")
                    # ponytail: scan this small local inbox; index job state for a large multi-user inbox.
                    ticket = next((t for t in reversed(rows("tickets", db)) if t["draft"] is None
                                   and t["status"] == "new" and t.get("auto_draft", {}).get("state") == "queued"), None)
                    if ticket is None:
                        return False
                    ticket["auto_draft"] = {"state": "running", "error": ""}
                    write("tickets", ticket, db)
                try:
                    create_draft(ticket["id"], DraftInput(revision=ticket["revision"]))
                except Exception as exc:
                    message = (exc.detail if isinstance(exc, HTTPException) else str(exc) if isinstance(exc, ProviderError)
                               else "자동 초안 작성 중 오류가 발생했습니다. 재시도해 주세요.")
                    with database() as db:
                        db.execute("BEGIN IMMEDIATE")
                        current = find("tickets", ticket["id"], db)
                        if current["draft"] is None and current["status"] == "new":
                            current["auto_draft"] = {"state": "failed", "error": message}
                            current["updated_at"] = now()
                            current["history"].append({"at": now(), "event": "auto_draft_failed", "detail": message})
                            write("tickets", current, db)
                return True
        except (HTTPException, ProviderError):
            # Another AI operation, unreadable key or stale index leaves the queue untouched.
            return False

    def auto_worker():
        while not stop_worker.is_set():
            wake_worker.wait()
            wake_worker.clear()
            while not stop_worker.is_set() and process_next_draft():
                pass

    @application.post("/api/tickets/{ticket_id}/resolve")
    def resolve(ticket_id: str, payload: ResolveInput):
        with database() as db:
            db.execute("BEGIN IMMEDIATE")
            ticket = find("tickets", ticket_id, db)
            check_revision(ticket, payload.revision)
            if payload.action == "approve":
                draft = ticket.get("draft") or {}
                if draft.get("action") != "answer" or not draft.get("citations"):
                    raise HTTPException(409, "근거가 확인된 답변 초안만 승인할 수 있습니다. 필요한 정보를 확인하거나 담당자에게 이관해 주세요.")
                if draft.get("index_fingerprint") != documents_fingerprint(rows("documents", db)):
                    raise HTTPException(409, "정책이 변경되어 이 초안은 승인할 수 없습니다. 색인을 갱신하고 다시 작성해 주세요.")
            ticket.update(status="resolved" if payload.action == "approve" else "escalated", final_answer=payload.answer, updated_at=now(), revision=ticket["revision"] + 1, auto_draft={"state": "completed", "error": ""})
            ticket["history"].append({"at": now(), "event": payload.action, "detail": payload.answer})
            write("tickets", ticket, db)
            return ticket_detail(ticket_id, db)

    dist = ROOT / "frontend" / "dist"
    if (dist / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @application.get("/")
    def index():
        if not (dist / "index.html").is_file():
            raise HTTPException(503, "화면을 먼저 빌드해 주세요: cd frontend; npm run build")
        return FileResponse(dist / "index.html")

    return application


def new_ticket(values):
    timestamp = now()
    return {**values, "id": str(uuid4()), "status": "new", "revision": 1, "created_at": timestamp, "updated_at": timestamp, "draft": None, "auto_draft": {"state": "queued", "error": ""}, "final_answer": "", "history": [{"at": timestamp, "event": "received", "detail": "문의 접수"}]}
