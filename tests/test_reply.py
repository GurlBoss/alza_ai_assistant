import base64
from datetime import datetime, timezone
from email import message_from_bytes, policy

from app.mail_processing.gmail import Email
from app.mail_processing.reply import (
    AI_GENERATED_TEXT,
    build_reply,
)

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
    assert message.get_content_type() == "multipart/alternative"
    plain = message.get_body(("plain",)).get_content()
    assert plain.strip() == f"Test reply\n\n--\n{AI_GENERATED_TEXT}"
