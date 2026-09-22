import copy
import json

import pytest

from desk.evaluate import DEMO_DIR, case_documents, evaluate, load_dataset, main
from desk.rag import ProviderError


class FakeEngine:
    paths = []
    indexed_ids = []
    closed = 0

    def __init__(self, path):
        self.paths.append(path)

    def index(self, documents, provider):
        self.indexed_ids.append([doc["id"] for doc in documents])

    def draft(self, ticket, customer, provider):
        return {
            "action": "answer", "answer": "모의 답변", "reason": "테스트 전용", "citations": [{"chunk_id": "chunk", "quote": "테스트"}],
            "evidence": [{"chunk_id": "chunk", "document_id": "shop-dispatch", "text": "테스트"}],
        }

    def close(self):
        type(self).closed += 1


@pytest.fixture(autouse=True)
def reset_fake():
    FakeEngine.paths = []
    FakeEngine.indexed_ids = []
    FakeEngine.closed = 0


def test_default_validates_all_cases_without_key_read_or_network(tmp_path, capsys):
    def forbidden(*args):
        raise AssertionError("외부 호출이 없어야 합니다.")

    assert main(["--key-file", str(tmp_path / "does-not-exist")], provider_factory=forbidden, engine_factory=forbidden) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "validation_only"
    assert report["document_count"] == 10
    assert report["customer_count"] == 6
    assert report["case_count"] == 30
    assert report["network_calls"] == 0
    assert report["semantic_accuracy"] is None


@pytest.mark.parametrize("changes", [
    {"name": ""}, {"order_number": None}, {"product_name": " "}, {"option": 100},
    {"order_status": "active"}, {"paid_amount_krw": -1}, {"paid_amount_krw": True},
    {"paid_amount_krw": 39000.5}, {"days_since_order": -1}, {"days_since_order": "1"},
    {"days_since_delivery": -1}, {"days_since_delivery": False}, {"days_since_delivery": "0"},
])
def test_invalid_order_profile_rejected_before_evaluation(tmp_path, changes):
    for filename in ("documents.json", "customers.json", "evaluation.json"):
        records = json.loads((DEMO_DIR / filename).read_text(encoding="utf-8"))
        if filename == "customers.json":
            records[0].update(changes)
        (tmp_path / filename).write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="고객 주문 프로필"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("delivery_days", [None, 0, 7])
def test_delivery_day_count_preserves_unknown_and_boundary_values(tmp_path, delivery_days):
    for filename in ("documents.json", "customers.json", "evaluation.json"):
        records = json.loads((DEMO_DIR / filename).read_text(encoding="utf-8"))
        if filename == "customers.json":
            records[0]["days_since_delivery"] = delivery_days
        (tmp_path / filename).write_text(json.dumps(records), encoding="utf-8")
    _, customers, _ = load_dataset(tmp_path)
    assert customers["shop-customer-01"]["days_since_delivery"] == delivery_days


def test_case_override_isolated_and_metrics_are_not_semantic_accuracy():
    documents, customers, cases = load_dataset()
    original = copy.deepcopy(documents)
    selected = [cases[0], cases[1], cases[28], cases[0]]
    report = evaluate(documents, customers, selected, object(), engine_factory=FakeEngine)
    assert documents == original
    assert len(FakeEngine.indexed_ids) == 3
    assert "shop-shipping-fees-conflict" not in FakeEngine.indexed_ids[0]
    assert "shop-shipping-fees-conflict" in FakeEngine.indexed_ids[1]
    assert "shop-shipping-fees" in FakeEngine.indexed_ids[1]
    assert "shop-shipping-fees-conflict" not in FakeEngine.indexed_ids[2]
    assert FakeEngine.closed == 1
    assert not FakeEngine.paths[0].exists()
    assert report["route_accuracy"] == 0.75
    assert report["candidate_source_coverage"] == pytest.approx(2 / 6)
    assert report["semantic_accuracy"] is None
    assert report["manual_review"] == "required"
    assert report["cases"][2]["missing_expected_document_ids"] == ["shop-shipping-fees", "shop-shipping-fees-conflict", "support-rules"]


def test_replacement_does_not_remove_other_documents():
    documents, _, _ = load_dataset()
    replacement = {**documents[0], "body": "변경 내용"}
    result = case_documents(documents, {"document_overrides": [replacement]})
    assert len(result) == len(documents)
    assert result[0]["body"] == "변경 내용"
    assert documents[0]["body"] != "변경 내용"


@pytest.mark.parametrize("failure_stage", ["index", "draft"])
def test_errors_sanitized_and_later_cases_still_run(failure_stage):
    class FailingEngine(FakeEngine):
        calls = 0

        def index(self, documents, provider):
            if failure_stage == "index":
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("do-not-output-SECRET")
            return super().index(documents, provider)

        def draft(self, ticket, customer, provider):
            if failure_stage == "draft":
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("do-not-output-SECRET")
            return super().draft(ticket, customer, provider)

    documents, customers, cases = load_dataset()
    report = evaluate(documents, customers, cases[:2], object(), engine_factory=FailingEngine)
    assert report["completed_count"] == 1
    assert report["route_accuracy"] == 0.5
    assert report["cases"][0]["error_stage"] == failure_stage
    assert "SECRET" not in json.dumps(report)


def test_live_explicit_key_file_and_case_selection(tmp_path, capsys):
    key_path = tmp_path / "key"
    key_path.write_text("TEST-ONLY-KEY", encoding="utf-8")
    captured = []

    def provider(key):
        captured.append(key)
        return object()

    assert main(["--live", "--key-file", str(key_path), "--case", "eval-01"], provider_factory=provider, engine_factory=FakeEngine) == 0
    output = capsys.readouterr().out
    assert captured == ["TEST-ONLY-KEY"]
    assert "TEST-ONLY-KEY" not in output
    assert json.loads(output)["case_count"] == 1


def test_no_key_refuses_live(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    report = tmp_path / "report.json"
    assert main(["--live", "--report", str(report)], provider_factory=lambda key: pytest.fail("키가 없습니다.")) == 2
    assert not report.exists()
    assert "오류 원문은 출력하지 않습니다" in capsys.readouterr().err


def test_existing_report_not_overwritten_or_billed(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")
    assert main(["--live", "--report", str(report)], provider_factory=lambda key: pytest.fail("호출 금지")) == 2
    assert report.read_text(encoding="utf-8") == "keep"
    assert "덮어쓰지 않습니다" in capsys.readouterr().err


def test_new_offline_report_and_unknown_case(tmp_path, capsys):
    report = tmp_path / "report.json"
    assert main(["--report", str(report), "--case", "eval-29"]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["case_ids"] == ["eval-29"]
    assert main(["--case", "missing"]) == 2


def test_invalid_cross_reference_rejected_before_provider(tmp_path):
    for name in ("documents.json", "customers.json", "evaluation.json"):
        data = json.loads((DEMO_DIR / name).read_text(encoding="utf-8"))
        if name == "evaluation.json":
            data[0]["expected_document_ids"] = ["not-a-document"]
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="출처 참조"):
        load_dataset(tmp_path)


def test_engine_initialization_error_does_not_print_secret(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "TEST-ONLY-KEY")

    def failing_engine(path):
        raise RuntimeError("SECRET-IN-ERROR")

    assert main(["--live"], provider_factory=lambda key: object(), engine_factory=failing_engine) == 2
    output = capsys.readouterr()
    assert "SECRET" not in output.out + output.err
