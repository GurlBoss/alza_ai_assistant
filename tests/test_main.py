import base64
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.utils import auth
from app.utils.auth import verify_oidc
from app.main import app, get_mailbox
from app.utils.firestore_storage import COMPLETED
from tests.fakes import BOT, FakeGmail, make_email, make_mailbox

client = TestClient(app)


def push_body(email_address: str = BOT, history_id: int = 101) -> dict:
    data = {"emailAddress": email_address, "historyId": history_id}
    return {
        "message": {
            "data": base64.b64encode(json.dumps(data).encode()).decode(),
            "messageId": "1",
            "publishTime": "2026-11-17T12:00:00Z",
        },
        "subscription": "projects/test/subscriptions/gmail-push",
    }


@pytest.fixture
def mailbox():
    gmail = FakeGmail([])
    gmail.add(make_email("m1"))
    box = make_mailbox(gmail, history_id="100")
    app.dependency_overrides[get_mailbox] = lambda: box
    app.dependency_overrides[verify_oidc] = lambda: {"email": "invoker@test"}
    yield box
    app.dependency_overrides.clear()


def test_webhook_processes_new_mail(mailbox) -> None:
    response = client.post("/gmail/webhook", json=push_body())

    assert response.status_code == 204
    assert mailbox.storage.docs["m1"]["status"] == COMPLETED
    assert mailbox.storage.get_history_id() == "101"


class TestOidc:
    """Real verify_oidc, with Google's token verification mocked."""

    @pytest.fixture(autouse=True)
    def configured(self):
        with patch.multiple(
            auth.settings,
            SERVICE_URL="https://svc.run.app",
            INVOKER_SERVICE_ACCOUNT="invoker@p.iam.gserviceaccount.com",
        ):
            yield

    def test_wrong_service_account(self) -> None:
        claims = {"email": "someone@gmail.com", "email_verified": True}
        with patch.object(auth.id_token, "verify_oauth2_token", return_value=claims):
            response = client.post(
                "/internal/gmail/renew-watch", headers={"Authorization": "Bearer x"}
            )
        assert response.status_code == 403


def test_invalid_token_is_not_logged(caplog) -> None:
    secret_token = "garbage.secret-token-value"
    with patch.object(auth.settings, "SERVICE_URL", "https://svc.run.app"):
        with caplog.at_level("WARNING"):
            response = client.post(
                "/internal/gmail/recovery-check-inbox",
                headers={"Authorization": f"Bearer {secret_token}"},
            )

    assert response.status_code == 401
    assert "Invalid OIDC token: MalformedError" in caplog.text
    assert "secret-token-value" not in caplog.text
