"""Disposable browser-test server. No real key or Gemini request is used."""
import argparse
import time
from tempfile import TemporaryDirectory

import uvicorn

from desk.main import create_app
from desk.rag import DIMENSIONS, ProviderError


class BrowserFixtureProvider:
    failed_once = set()

    def embed_documents(self, texts):
        return [[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * (DIMENSIONS - 1)

    def generate(self, ticket, customer, evidence, guidance="", current_answer=""):
        if ticket["subject"] == "초안 실패 검사" and ticket["id"] not in self.failed_once:
            self.failed_once.add(ticket["id"])
            raise ProviderError("브라우저 검사에서 만든 일시적인 초안 오류입니다.")
        if ticket["subject"].startswith("백그라운드"):
            time.sleep(2.5)
        return {
            "category": "자동화 검사", "action": "answer", "evidence_status": "supported",
            "answer": (
                f"[브라우저 테스트용 재생성] 입력 방향: {guidance}\n수정 전 답변: {current_answer}"
                if guidance or current_answer else
                "[브라우저 테스트용 모의 응답] 주문 정보와 FAQ 원문을 검토한 후 답변을 기록해 주세요. 실제 Gemini가 작성한 답변이 아닙니다."
            ),
            "reason": "화면 흐름을 검사하기 위한 모의 응답입니다. 모델 정확도 검사가 아닙니다.",
            "citations": [{"chunk_id": evidence[0]["chunk_id"], "quote": evidence[0]["text"][:90]}],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="inquiry-desk-browser-") as directory:
        uvicorn.run(create_app(directory, provider_factory=BrowserFixtureProvider), host="127.0.0.1", port=args.port)
