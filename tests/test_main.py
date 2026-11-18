import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.utils import auth
from app.utils.auth import verify_oidc
from app import main
from app.main import app, decode_notification, get_mailbox
from app.mail_processing.store import COMPLETED
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


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_each_request_gets_its_own_gmail_client(monkeypatch) -> None:
    """The Gmail client's HTTP connection is not thread-safe."""
    shared = make_mailbox(FakeGmail([]))
    monkeypatch.setattr(main, "shared_mailbox", lambda: shared)
    monkeypatch.setattr(main, "gmail_credentials", lambda: "creds")
    monkeypatch.setattr(
        main.GmailClient, "from_credentials", lambda creds: FakeGmail([])
    )

    first, second = get_mailbox(), get_mailbox()

    assert first.gmail is not second.gmail
    assert first.store is second.store is shared.store
    assert first.gemini is second.gemini


def test_decode_notification() -> None:
    assert decode_notification(push_body(history_id=12345)) == (BOT, "12345")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": {}},
        {"message": {"data": "not base64!"}},
        {"message": {"data": base64.b64encode(b"not json").decode()}},
        {"message": {"data": base64.b64encode(b'{"emailAddress": "x"}').decode()}},
    ],
)
def test_invalid_payload_is_rejected(mailbox, body: dict) -> None:
    assert client.post("/gmail/webhook", json=body).status_code == 400


def test_webhook_processes_new_mail(mailbox) -> None:
    response = client.post("/gmail/webhook", json=push_body())

    assert response.status_code == 204
    assert mailbox.store.docs["m1"]["status"] == COMPLETED
    assert mailbox.sync_state.get_history_id() == "101"


def test_duplicate_notification_replies_once(mailbox) -> None:
    client.post("/gmail/webhook", json=push_body())
    response = client.post("/gmail/webhook", json=push_body())

    assert response.status_code == 204
    assert len(mailbox.gmail.sent) == 1


def test_notification_for_other_mailbox_is_ignored(mailbox) -> None:
    response = client.post("/gmail/webhook", json=push_body("other@example.com"))

    assert response.status_code == 204
    assert mailbox.gmail.sent == []


def test_recovery_sync(mailbox) -> None:
    response = client.post("/internal/gmail/recovery-sync")

    assert response.status_code == 200
    assert response.json()["results"] == {"m1": COMPLETED}


def test_renew_watch_stores_history_id_only_if_missing(mailbox) -> None:
    with patch.object(auth.settings, "PUBSUB_TOPIC", "projects/p/topics/t"):
        mailbox.sync_state.history_id = None
        assert client.post("/internal/gmail/renew-watch").status_code == 200
        assert mailbox.sync_state.get_history_id() == "101"

        mailbox.gmail.history_id = "999"
        client.post("/internal/gmail/renew-watch")
        assert mailbox.sync_state.get_history_id() == "101"


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

    def test_missing_token(self) -> None:
        assert client.post("/gmail/webhook", json=push_body()).status_code == 401

    def test_invalid_token(self) -> None:
        with patch.object(
            auth.id_token, "verify_oauth2_token", side_effect=ValueError("bad")
        ):
            response = client.post(
                "/internal/gmail/recovery-sync",
                headers={"Authorization": "Bearer x"},
            )
        assert response.status_code == 401

    def test_wrong_service_account(self) -> None:
        claims = {"email": "someone@gmail.com", "email_verified": True}
        with patch.object(auth.id_token, "verify_oauth2_token", return_value=claims):
            response = client.post(
                "/internal/gmail/renew-watch", headers={"Authorization": "Bearer x"}
            )
        assert response.status_code == 403

    @pytest.mark.parametrize(
        ("path", "audience"),
        [
            ("/gmail/webhook", "https://svc.run.app"),
            ("/gmail/webhook", "https://svc.run.app/gmail/webhook"),
            ("/internal/gmail/renew-watch", "https://svc.run.app"),
            (
                "/internal/gmail/renew-watch",
                "https://svc.run.app/internal/gmail/renew-watch",
            ),
        ],
    )
    def test_service_url_with_or_without_path_is_accepted(
        self, path: str, audience: str
    ) -> None:
        assert audience in self.accepted_audiences(path)

    def test_audience_for_another_path_or_service_is_not_accepted(self) -> None:
        accepted = self.accepted_audiences("/gmail/webhook")
        assert "https://svc.run.app/internal/gmail/renew-watch" not in accepted
        assert "https://other.run.app" not in accepted

    @staticmethod
    def accepted_audiences(path: str) -> list[str]:
        """The audience list verify_oidc passes to Google's token verification."""
        claims = {"email": "invoker@p.iam.gserviceaccount.com", "email_verified": True}
        request = MagicMock()
        request.headers = {"authorization": "Bearer tok"}
        request.url.path = path
        with patch.object(
            auth.id_token, "verify_oauth2_token", return_value=claims
        ) as verify:
            assert auth.verify_oidc(request) == claims
        return verify.call_args.kwargs["audience"]
