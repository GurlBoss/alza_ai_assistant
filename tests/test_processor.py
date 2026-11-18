import base64
from datetime import datetime, timezone
from email import message_from_bytes, policy
from unittest.mock import MagicMock

import pytest

from app.ai_core import llm
from app.config import Settings
from app.mail_processing.gmail import Email, GmailClient
from app.mail_processing.processor import (
    FALLBACK_REPLY,
    SKIPPED,
    Mailbox,
    process_email,
)
from app.mail_processing.store import (
    COMPLETED,
    FAILED,
    InMemoryStore,
    InMemorySyncState,
)

BOT = "bot@example.com"
SETTINGS = Settings(GEMINI_MODEL="gemini-test")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


def make_email() -> Email:
    return Email(
        id="msg-1",
        thread_id="thread-1",
        sender="Jan Novák <Jan@Example.com>",
        subject="Order status",
        body="Where is my order?",
        message_id="<abc@mail.example.com>",
        references="",
        received_at=datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc),
    )


def make_gmail() -> tuple[GmailClient, MagicMock]:
    service = MagicMock()
    service.users().messages().send().execute.return_value = {"id": "sent-1"}
    service.reset_mock()
    return GmailClient(service), service


def make_gemini(answer: str = "Gemini answer") -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=answer)
    return client


def run(gmail: GmailClient, store: InMemoryStore, gemini: MagicMock) -> str:
    mailbox = Mailbox(gmail, BOT, store, InMemorySyncState(), gemini, SETTINGS)
    return process_email(mailbox, make_email())


def sent_text(service: MagicMock) -> str:
    raw = service.users().messages().send.call_args.kwargs["body"]["raw"]
    message = message_from_bytes(base64.urlsafe_b64decode(raw), policy=policy.default)
    return message.get_content()


def test_new_email_is_answered_by_gemini() -> None:
    gmail, service = make_gmail()
    gemini = make_gemini()
    store = InMemoryStore()

    result = run(gmail, store, gemini)

    assert result == COMPLETED
    assert sent_text(service).strip() == "Gemini answer"
    assert gemini.models.generate_content.call_args.kwargs["model"] == "gemini-test"
    doc = store.docs["msg-1"]
    assert doc["status"] == COMPLETED
    assert doc["reply_message_id"] == "sent-1"
    assert doc["sender_domain"] == "example.com"
    assert doc["subject_length"] == len("Order status")
    assert doc["received_at"] == make_email().received_at
    assert not {"body", "created_at", "expire_at"} & doc.keys()
    service.users().messages().modify.assert_called_once()


def test_duplicate_email_is_skipped() -> None:
    gmail, service = make_gmail()
    gemini = make_gemini()
    store = InMemoryStore()
    run(gmail, store, gemini)
    service.reset_mock()
    gemini.reset_mock()

    assert run(gmail, store, gemini) == SKIPPED

    gemini.models.generate_content.assert_not_called()
    service.users().messages().send.assert_not_called()
    assert store.docs["msg-1"]["status"] == COMPLETED


def test_gemini_failure_sends_fallback_and_sets_failed() -> None:
    gmail, service = make_gmail()
    gemini = MagicMock()
    gemini.models.generate_content.side_effect = TimeoutError("prompt text")
    store = InMemoryStore()

    result = run(gmail, store, gemini)

    assert result == FAILED
    assert gemini.models.generate_content.call_count == 2  # one retry
    assert sent_text(service).strip() == FALLBACK_REPLY.strip()
    doc = store.docs["msg-1"]
    assert doc["status"] == FAILED
    assert doc["error_type"] == "Gemini:TimeoutError"
    assert doc["reply_message_id"] == "sent-1"
    assert "prompt text" not in str(doc)
    service.users().messages().modify.assert_called_once()


def test_send_failure_sets_failed() -> None:
    gmail, service = make_gmail()
    service.users().messages().send().execute.side_effect = ConnectionError(
        "secret details"
    )
    store = InMemoryStore()

    # Recorded as FAILED and not raised, so the caller can move on.
    result = run(gmail, store, make_gemini())

    assert result == FAILED

    doc = store.docs["msg-1"]
    assert doc["status"] == FAILED
    assert doc["error_type"] == "ConnectionError"
    assert "reply_message_id" not in doc
    assert "secret details" not in str(doc)
    service.users().messages().modify.assert_not_called()


def test_mark_read_failure_keeps_completed() -> None:
    gmail, service = make_gmail()
    service.users().messages().modify().execute.side_effect = ConnectionError()
    store = InMemoryStore()

    result = run(gmail, store, make_gemini())

    assert result == COMPLETED
    assert store.docs["msg-1"]["status"] == COMPLETED
