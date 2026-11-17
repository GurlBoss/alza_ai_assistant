import base64
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gmail_webhook_accepts_pubsub_push() -> None:
    data = {"emailAddress": "bot@example.com", "historyId": 12345}
    payload = {
        "message": {
            "data": base64.b64encode(json.dumps(data).encode()).decode(),
            "messageId": "1",
            "publishTime": "2026-11-17T12:00:00Z",
        },
        "subscription": "projects/test/subscriptions/gmail-push",
    }
    response = client.post("/gmail/webhook", json=payload)
    assert response.status_code == 204


def test_gmail_webhook_rejects_invalid_payload() -> None:
    response = client.post("/gmail/webhook", json={"message": {}})
    assert response.status_code == 400
