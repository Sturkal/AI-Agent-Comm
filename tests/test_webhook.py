import pytest

flask = pytest.importorskip("flask")

from flask import Flask

from app.api.routes import register_routes


class FakeResult:
    def __init__(self):
        self.answer = "ok"
        self.xml_hits = [{"id": "xml_rule:1", "document": "xml"}]
        self.pdf_hits = [{"id": "pdf:1", "document": "pdf"}]
        self.self_evaluation_summary = "Round 1: enough evidence from XML/PDF to answer. confidence=0.90"
        self.self_evaluation = {"confidence": 0.9, "needs_refinement": False}
        self.refinement_memory = [{"summary": "Round 0: prior note"}]


class FakeEngine:
    def __init__(self):
        self.calls = []

    def answer(self, message_body, user_phone_number=None):
        self.calls.append((message_body, user_phone_number))
        return FakeResult()


def test_webhook_returns_agent_response():
    app = Flask(__name__)
    register_routes(app)
    fake_engine = FakeEngine()
    app.extensions["sfim_engine"] = fake_engine

    client = app.test_client()
    response = client.post(
        "/webhook/whatsapp",
        json={
            "user_phone_number": "+10000000000",
            "message_body": "How is Rule X calculated?",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["reply_message"] == "ok"
    assert payload["self_evaluation_summary"].startswith("Round 1:")
    assert payload["self_evaluation"]["confidence"] == 0.9
    assert payload["refinement_memory"][0]["summary"] == "Round 0: prior note"
    assert fake_engine.calls == [("How is Rule X calculated?", "+10000000000")]


def test_webhook_rejects_missing_message_body():
    app = Flask(__name__)
    register_routes(app)
    app.extensions["sfim_engine"] = FakeEngine()

    client = app.test_client()
    response = client.post("/webhook/whatsapp", json={"user_phone_number": "+1"})

    assert response.status_code == 400
    assert response.get_json()["error"] == "message_body is required"
