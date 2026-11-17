import base64
from datetime import datetime, timezone
from email import message_from_bytes, policy
from unittest.mock import MagicMock

from app.mail_processing.gmail import Email, GmailClient
from app.mail_processing.reply import build_reply, reply_subject, send_reply

BOT = "bot@example.com"


def make_email(**overrides: str) -> Email:
    fields: dict = {
        "id": "msg-1",
        "thread_id": "thread-1",
        "sender": "Jan Novák <jan@example.com>",
        "subject": "Order status",
        "body": "Where is my order?",
        "message_id": "<abc@mail.example.com>",
        "references": "<first@mail.example.com>",
        "received_at": datetime(2026, 11, 17, tzinfo=timezone.utc),
    }
    fields.update(overrides)
    return Email(**fields)


def decode(body: dict[str, str]):
    return message_from_bytes(
        base64.urlsafe_b64decode(body["raw"]), policy=policy.default
    )


def test_build_reply_keeps_thread_and_headers() -> None:
    body = build_reply(make_email(), "Test reply", BOT)
    message = decode(body)

    assert body["threadId"] == "thread-1"
    assert message["From"] == BOT
    assert message["To"] == "Jan Novák <jan@example.com>"
    assert message["Cc"] is None
    assert message["Subject"] == "Re: Order status"
    assert message["In-Reply-To"] == "<abc@mail.example.com>"
    assert message["References"] == "<first@mail.example.com> <abc@mail.example.com>"
    assert message.get_payload(decode=True).decode().strip() == "Test reply"


def test_build_reply_without_previous_references() -> None:
    message = decode(build_reply(make_email(references=""), "Hi", BOT))
    assert message["References"] == "<abc@mail.example.com>"


def test_reply_subject_does_not_double_prefix() -> None:
    assert reply_subject("RE: Order status") == "RE: Order status"
    assert reply_subject("Order status") == "Re: Order status"


def test_send_reply_calls_gmail_send() -> None:
    service = MagicMock()
    service.users().messages().send().execute.return_value = {"id": "sent-1"}

    sent_id = send_reply(GmailClient(service), make_email(), "Test reply", BOT)

    assert sent_id == "sent-1"
    _, kwargs = service.users().messages().send.call_args
    assert kwargs["userId"] == "me"
    assert kwargs["body"]["threadId"] == "thread-1"
