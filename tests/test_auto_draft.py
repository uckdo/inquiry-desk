import json
import sqlite3
from threading import Event, Lock
from time import monotonic

from fastapi.testclient import TestClient
import pytest

from desk import main
from desk.rag import ProviderError
from tests.test_api import FakeEngine


@pytest.fixture
def engine_factory(monkeypatch):
    """Persistent mock index and counted model calls; no Gemini requests."""
    engines = []
    index_state = {}

    class CountingEngine(FakeEngine):
        def __init__(self, path):
            super().__init__(path)
            self.path = str(path)
            self.meta = index_state.get(self.path, self.meta)
            self.calls = []
            self.active = 0
            self.max_active = 0
            self.lock = Lock()
            self.started = Event()
            self.release = Event()
            self.release.set()
            self.failure = None
            engines.append(self)

        def index(self, documents, provider):
            result = super().index(documents, provider)
            index_state[self.path] = result
            return result

        def draft(self, ticket, *args, **kwargs):
            with self.lock:
                self.calls.append(ticket["id"])
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.started.set()
            try:
                assert self.release.wait(3), "Test did not release the mock generation"
                if self.failure:
                    raise ProviderError(self.failure)
                return super().draft(ticket, *args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    monkeypatch.setattr(main, "RagEngine", CountingEngine)
    return engines


def wait_for(read, predicate, timeout=3):
    deadline = monotonic() + timeout
    last = None
    while monotonic() < deadline:
        last = read()
        if predicate(last):
            return last
        Event().wait(0.01)
    pytest.fail(f"Timed out waiting for automatic drafting: {last!r}")


def all_tickets(client):
    return client.get("/api/tickets").json()["tickets"]


def wait_completed(client, count):
    return wait_for(
        lambda: all_tickets(client),
        lambda tickets: len(tickets) == count
        and all(ticket.get("auto_draft", {}).get("state") == "completed" for ticket in tickets),
    )


def change_stored_ticket(data_dir, ticket_id, update):
    with sqlite3.connect(data_dir / "desk.sqlite3") as connection:
        body = connection.execute("SELECT body FROM tickets WHERE id=?", (ticket_id,)).fetchone()[0]
        ticket = json.loads(body)
        update(ticket)
        connection.execute("UPDATE tickets SET body=? WHERE id=?", (json.dumps(ticket), ticket_id))


def test_every_seed_and_new_inquiry_gets_a_first_draft_without_click(tmp_path, engine_factory):
    app = main.create_app(tmp_path, provider_factory=lambda: object())
    with TestClient(app) as client:
        pending = all_tickets(client)
        assert len(pending) == 8
        assert all(ticket["draft"] is None for ticket in pending)
        assert all(ticket["auto_draft"]["state"] == "queued" for ticket in pending)
        assert engine_factory[-1].calls == []
        manual = client.post(f'/api/tickets/{pending[0]["id"]}/draft', json={"revision": pending[0]["revision"]})
        assert manual.status_code == 409
        assert client.post("/api/index").status_code == 200
        completed = wait_completed(client, 8)
        assert all(ticket["draft"] and ticket["status"] == "review" for ticket in completed)
        assert all(ticket["final_answer"] == "" for ticket in completed)
        assert len(engine_factory[-1].calls) == 8
        created = client.post("/api/tickets", json={
            "customer_id": completed[0]["customer_id"],
            "subject": "배송 확인 부탁드립니다",
            "body": "주문한 티셔츠가 아직 도착하지 않았습니다.",
        })
        assert created.status_code == 201
        wait_completed(client, 9)
        assert engine_factory[-1].calls.count(created.json()["id"]) == 1
        assert len(engine_factory[-1].calls) == 9


def test_background_worker_runs_without_ticket_reads_and_does_not_duplicate(tmp_path, engine_factory):
    app = main.create_app(tmp_path, provider_factory=lambda: object())
    engine = engine_factory[-1]
    engine.release.clear()
    with TestClient(app) as client:
        try:
            assert client.post("/api/index").status_code == 200
            assert engine.started.wait(3)
            for _ in range(10):
                tickets = all_tickets(client)
                client.get("/api/status")
            assert len(engine.calls) == 1
            assert sum(ticket["auto_draft"]["state"] == "running" for ticket in tickets) == 1
            assert client.post("/api/index").status_code == 409
            assert not any(ticket["auto_draft"]["state"] == "failed" for ticket in all_tickets(client))
        finally:
            engine.release.set()
        wait_completed(client, 8)
        assert len(engine.calls) == len(set(engine.calls)) == 8
        assert engine.max_active == 1


def test_missing_key_waits_without_failure_and_key_save_resumes(tmp_path, engine_factory, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "GeminiProvider", lambda key: object())
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object(), auto_draft=False)) as client:
        assert client.post("/api/index").status_code == 200
    with TestClient(main.create_app(tmp_path)) as client:
        pending = all_tickets(client)
        assert all(ticket["auto_draft"]["state"] == "queued" for ticket in pending)
        assert engine_factory[-1].calls == []
        assert not client.get("/api/status").json()["configured"]
        response = client.put("/api/settings/key", json={"api_key": "dummy_test_key_not_a_real_secret"})
        assert response.status_code == 200
        wait_completed(client, 8)
        assert len(engine_factory[-1].calls) == 8


def test_existing_draft_and_processed_inquiry_are_not_overwritten_on_startup(tmp_path, engine_factory):
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object(), auto_draft=False)) as client:
        assert client.post("/api/index").status_code == 200
        tickets = all_tickets(client)
        draft_path = f'/api/tickets/{tickets[0]["id"]}'
        existing = client.post(draft_path + "/draft", json={"revision": tickets[0]["revision"]}).json()
        closed_path = f'/api/tickets/{tickets[1]["id"]}'
        closed = client.post(closed_path + "/resolve", json={
            "revision": tickets[1]["revision"], "action": "escalate", "answer": "담당자가 직접 확인 중입니다.",
        }).json()
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object())) as client:
        wait_for(lambda: all_tickets(client), lambda values: sum(bool(ticket["draft"]) for ticket in values) == 7)
        preserved = client.get(draft_path).json()
        assert preserved["draft"] == existing["draft"]
        assert preserved["revision"] == existing["revision"]
        assert preserved["history"] == existing["history"]
        resolved = client.get(closed_path).json()
        assert resolved["status"] == "escalated"
        assert resolved["final_answer"] == closed["final_answer"]
        assert resolved["revision"] == closed["revision"]
        assert tickets[0]["id"] not in engine_factory[-1].calls
        assert tickets[1]["id"] not in engine_factory[-1].calls
        assert len(engine_factory[-1].calls) == 6


def test_failure_is_persisted_and_only_retries_on_explicit_request(tmp_path, engine_factory):
    app = main.create_app(tmp_path, provider_factory=lambda: object())
    engine_factory[-1].failure = "모의 공급자 실패입니다."
    with TestClient(app) as client:
        assert client.post("/api/index").status_code == 200
        failed = wait_for(lambda: all_tickets(client), lambda tickets: all(ticket["auto_draft"]["state"] == "failed" for ticket in tickets))
        assert len(engine_factory[-1].calls) == 8
        assert all(ticket["draft"] is None and ticket["auto_draft"]["error"] for ticket in failed)
        for _ in range(10):
            all_tickets(client)
        assert len(engine_factory[-1].calls) == 8
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object())) as client:
        failed_again = all_tickets(client)
        assert all(ticket["auto_draft"]["state"] == "failed" for ticket in failed_again)
        assert engine_factory[-1].calls == []
        retry = failed_again[0]
        path = f'/api/tickets/{retry["id"]}'
        assert client.post(path + "/retry-draft", json={"revision": retry["revision"] + 100}).status_code == 409
        assert client.post(path + "/retry-draft", json={"revision": retry["revision"]}).status_code == 200
        result = wait_for(lambda: client.get(path).json(), lambda ticket: ticket["auto_draft"]["state"] == "completed")
        assert result["draft"]
        assert result["auto_draft"]["error"] == ""
        assert engine_factory[-1].calls == [retry["id"]]
        assert client.post(path + "/retry-draft", json={"revision": result["revision"]}).status_code == 409


def test_interrupted_generation_does_not_silently_retry_on_restart(tmp_path, engine_factory):
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object(), auto_draft=False)) as client:
        assert client.post("/api/index").status_code == 200
        ticket = all_tickets(client)[0]
    change_stored_ticket(tmp_path, ticket["id"], lambda current: current.update(auto_draft={"state": "running", "error": ""}))
    with TestClient(main.create_app(tmp_path, provider_factory=lambda: object())) as client:
        wait_for(lambda: all_tickets(client), lambda tickets: sum(bool(item["draft"]) for item in tickets) == 7)
        interrupted = client.get(f'/api/tickets/{ticket["id"]}').json()
        assert interrupted["auto_draft"]["state"] == "failed"
        assert interrupted["auto_draft"]["error"]
        assert interrupted["draft"] is None
        assert ticket["id"] not in engine_factory[-1].calls


def test_stale_index_pauses_pending_drafts_until_rebuilt(tmp_path, engine_factory):
    app = main.create_app(tmp_path, provider_factory=lambda: object())
    engine = engine_factory[-1]
    engine.release.clear()
    with TestClient(app) as client:
        try:
            assert client.post("/api/index").status_code == 200
            assert engine.started.wait(3)
            running_id = engine.calls[0]
            document = client.get("/api/documents").json()["documents"][0]
            payload = {key: document[key] for key in ("title", "topic", "body", "version")}
            payload["body"] += "\n검토자가 생성 도중 추가한 안내입니다."
            assert client.put(f'/api/documents/{document["id"]}', json=payload).status_code == 200
        finally:
            engine.release.set()
        wait_for(lambda: client.get(f'/api/tickets/{running_id}').json(), lambda ticket: ticket["auto_draft"]["state"] != "running")
        pending = all_tickets(client)
        assert all(ticket["draft"] is None for ticket in pending)
        assert len(engine.calls) == 1
        assert sum(ticket["auto_draft"]["state"] == "queued" for ticket in pending) >= 7
        assert client.post("/api/index").status_code == 200
        wait_for(lambda: all_tickets(client), lambda tickets: sum(bool(ticket["draft"]) for ticket in tickets) >= 7)


def test_automatic_result_cannot_overwrite_concurrent_human_escalation(tmp_path, engine_factory):
    app = main.create_app(tmp_path, provider_factory=lambda: object())
    engine = engine_factory[-1]
    engine.release.clear()
    with TestClient(app) as client:
        try:
            assert client.post("/api/index").status_code == 200
            assert engine.started.wait(3)
            path = f'/api/tickets/{engine.calls[0]}'
            running = client.get(path).json()
            response = client.post(path + "/resolve", json={
                "revision": running["revision"], "action": "escalate", "answer": "오배송 확인을 담당자에게 요청했습니다.",
            })
            assert response.status_code == 200
            escalated = response.json()
        finally:
            engine.release.set()
        wait_for(lambda: all_tickets(client), lambda tickets: sum(bool(ticket["draft"]) for ticket in tickets) == 7)
        preserved = client.get(path).json()
        assert preserved["status"] == "escalated"
        assert preserved["draft"] is None
        assert preserved["final_answer"] == escalated["final_answer"]
        assert preserved["revision"] == escalated["revision"]
        assert preserved["history"] == escalated["history"]
