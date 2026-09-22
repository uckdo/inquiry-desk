"""Validate demo cases offline; --live explicitly enables paid Gemini requests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from desk.rag import EMBEDDING_MODEL, GeminiProvider, ProviderError, RagEngine, documents_fingerprint

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
ACTIONS = {"answer", "clarify", "escalate"}


def _records(directory: Path, filename: str) -> list[dict]:
    path = directory / filename
    if path.stat().st_size > 5_000_000:
        raise ValueError("평가 데이터 파일이 너무 큽니다.")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("평가 데이터는 1~1,000개 항목의 배열이어야 합니다.")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("평가 항목에 문자열 ID가 필요합니다.")
        if row["id"] in ids:
            raise ValueError("평가 데이터에 중복 ID가 있습니다.")
        ids.add(row["id"])
    return rows


def _document(doc: dict) -> None:
    if not isinstance(doc, dict) or any(
        not isinstance(doc.get(key), str) or not doc[key].strip() or len(doc[key]) > limit
        for key, limit in (("id", 100), ("title", 200), ("topic", 100), ("body", 20000))
    ) or type(doc.get("version")) is not int or doc["version"] < 1:
        raise ValueError("운영규정 형식이 올바르지 않습니다.")


def case_documents(documents: list[dict], case: dict) -> list[dict]:
    result = {doc["id"]: dict(doc) for doc in documents}
    result.update({doc["id"]: dict(doc) for doc in case.get("document_overrides", [])})
    return list(result.values())


def load_dataset(directory: Path = DEMO_DIR) -> tuple[list[dict], dict, list[dict]]:
    documents = _records(directory, "documents.json")
    customers = {row["id"]: row for row in _records(directory, "customers.json")}
    cases = _records(directory, "evaluation.json")
    for document in documents:
        _document(document)
    for customer in customers.values():
        if (
            any(not isinstance(customer.get(key), str) or not customer[key].strip()
                for key in ("name", "order_number", "product_name", "option"))
            or customer.get("order_status") not in ("paid", "preparing", "shipped", "delivered", "cancelled")
            or any(type(customer.get(key)) is not int or customer[key] < 0
                   for key in ("paid_amount_krw", "days_since_order"))
            or "days_since_delivery" not in customer
            or (customer["days_since_delivery"] is not None and (
                type(customer["days_since_delivery"]) is not int or customer["days_since_delivery"] < 0
            ))
        ):
            raise ValueError("고객 주문 프로필 형식이 올바르지 않습니다.")
    for case in cases:
        if (
            not isinstance(case.get("question"), str) or not 1 <= len(case["question"]) <= 5000
            or case.get("customer_id") not in customers
            or case.get("expected_action") not in ACTIONS
        ):
            raise ValueError("평가 질문 또는 고객 참조가 올바르지 않습니다.")
        for key in ("expected_document_ids", "required_facts"):
            if not isinstance(case.get(key), list) or any(
                not isinstance(value, str) or not value.strip() for value in case[key]
            ) or len(set(case[key])) != len(case[key]):
                raise ValueError("평가 기대값은 중복 없는 문자열 배열이어야 합니다.")
        overrides = case.get("document_overrides", [])
        if not isinstance(overrides, list):
            raise ValueError("사례별 문서 변형은 배열이어야 합니다.")
        for document in overrides:
            _document(document)
        if len({doc["id"] for doc in overrides}) != len(overrides):
            raise ValueError("사례별 문서 변형에 중복 ID가 있습니다.")
        current = case_documents(documents, case)
        if len(current) > 100 or not set(case["expected_document_ids"]) <= {doc["id"] for doc in current}:
            raise ValueError("평가 사례의 문서 개수 또는 출처 참조가 올바르지 않습니다.")
    return documents, customers, cases


def evaluate(documents, customers, cases, provider, *, engine_factory=RagEngine) -> dict:
    """Run only against a temporary index; never opens the application data directory."""
    results = []
    with TemporaryDirectory(prefix="inquiry-desk-evaluation-") as temporary:
        engine = engine_factory(Path(temporary))
        current_fingerprint = None
        try:
            for case in cases:
                expected = case["expected_document_ids"]
                result = {
                    "id": case["id"], "question": case["question"],
                    "customer_id": case["customer_id"], "expected_action": case["expected_action"],
                    "expected_document_ids": expected, "required_facts": case["required_facts"],
                    "manual_review": "required", "route_match": False,
                    "retrieved_document_ids": [], "cited_document_ids": [],
                    "missing_expected_document_ids": expected, "candidate_source_coverage": 0 if expected else None,
                }
                stage = "index"
                try:
                    current = case_documents(documents, case)
                    fingerprint = documents_fingerprint(current)
                    if fingerprint != current_fingerprint:
                        engine.index(current, provider)
                        current_fingerprint = fingerprint
                    stage = "draft"
                    draft = engine.draft(
                        {"subject": "", "body": case["question"]}, customers[case["customer_id"]], provider
                    )
                    evidence = {item["chunk_id"]: item["document_id"] for item in draft["evidence"]}
                    retrieved = sorted(set(evidence.values()))
                    found = set(expected) & set(retrieved)
                    result.update(
                        status="completed", actual_action=draft["action"], answer=draft["answer"],
                        reason=draft["reason"], citations=draft["citations"], evidence=draft["evidence"],
                        index_fingerprint=fingerprint, route_match=draft["action"] == case["expected_action"],
                        retrieved_document_ids=retrieved,
                        cited_document_ids=sorted({evidence[c["chunk_id"]] for c in draft["citations"] if c["chunk_id"] in evidence}),
                        missing_expected_document_ids=sorted(set(expected) - found),
                        candidate_source_coverage=len(found) / len(expected) if expected else None,
                    )
                except Exception as error:
                    result.update(
                        status="error", error_stage=stage,
                        error_code="provider_error" if isinstance(error, ProviderError) else "evaluation_error",
                        error="평가 실행에 실패했습니다. 원문 오류와 인증정보는 기록하지 않습니다.",
                    )
                results.append(result)
        finally:
            engine.close()
    expected_count = sum(len(row["expected_document_ids"]) for row in results)
    found_count = sum(len(row["expected_document_ids"]) - len(row["missing_expected_document_ids"]) for row in results)
    return {
        "mode": "executed", "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": getattr(provider, "model", "test-provider"), "embedding_model": EMBEDDING_MODEL,
        "case_count": len(results), "completed_count": sum(row["status"] == "completed" for row in results),
        "route_accuracy": sum(row["route_match"] for row in results) / len(results) if results else None,
        "candidate_source_coverage": found_count / expected_count if expected_count else None,
        "metric_notes": "분류 정확도는 오류 사례를 실패로 포함합니다. 출처 커버리지는 기대 출처 후보가 검색되었는지만 측정하며 인용 의무나 답변의 의미 정확도가 아닙니다.",
        "semantic_accuracy": None, "manual_review": "required", "cases": results,
    }


def main(argv=None, *, provider_factory=GeminiProvider, engine_factory=RagEngine) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEMO_DIR)
    parser.add_argument("--case", action="append", dest="case_ids", help="실행할 사례 ID. 반복 지정할 수 있습니다.")
    parser.add_argument("--live", action="store_true", help="Gemini 외부 호출을 허용합니다. API 사용료가 발생할 수 있습니다.")
    parser.add_argument("--key-file", type=Path, help="명시한 키 파일 또는 GEMINI_API_KEY 환경변수만 사용합니다.")
    parser.add_argument("--report", type=Path, help="새 JSON 보고서 경로. 기존 파일은 덮어쓰지 않습니다.")
    args = parser.parse_args(argv)
    handle = None
    try:
        documents, customers, cases = load_dataset(args.dataset_dir)
        if args.case_ids:
            selected = set(args.case_ids)
            if not selected <= {case["id"] for case in cases}:
                raise ValueError("요청한 평가 사례 ID가 없습니다.")
            cases = [case for case in cases if case["id"] in selected]
        if args.report and args.report.exists():
            raise FileExistsError
        if args.live:
            if args.key_file:
                if args.key_file.stat().st_size > 512:
                    raise ValueError("API 키 파일 형식이 올바르지 않습니다.")
                key = args.key_file.read_text(encoding="utf-8").strip()
            else:
                key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                raise ValueError("--live 실행에는 GEMINI_API_KEY 또는 --key-file이 필요합니다.")
            provider = provider_factory(key)
        if args.report:
            handle = args.report.open("x", encoding="utf-8")
        if args.live:
            report = evaluate(documents, customers, cases, provider, engine_factory=engine_factory)
            report["mode"] = "live"
        else:
            report = {
                "mode": "validation_only", "network_calls": 0,
                "document_count": len(documents), "customer_count": len(customers), "case_count": len(cases),
                "expected_actions": dict(Counter(case["expected_action"] for case in cases)),
                "case_ids": [case["id"] for case in cases], "semantic_accuracy": None,
                "note": "데이터 형식과 참조만 확인했습니다. 모델 평가 결과가 아닙니다. --live는 외부 API를 호출하며 비용이 발생할 수 있습니다.",
            }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if handle:
            handle.write(rendered + "\n")
        print(rendered)
        return 1 if report.get("completed_count", len(cases)) != len(cases) else 0
    except FileExistsError:
        print("기존 보고서는 덮어쓰지 않습니다. 새 경로를 지정해 주세요.", file=sys.stderr)
        return 2
    except Exception:
        print("평가를 시작하거나 보고서를 저장하지 못했습니다. 데이터·키 설정·출력 경로를 확인해 주세요. 오류 원문은 출력하지 않습니다.", file=sys.stderr)
        return 2
    finally:
        if handle:
            handle.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
