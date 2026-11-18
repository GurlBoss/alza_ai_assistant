"""In-memory Gmail, Gemini and Firestore doubles shared by the mailbox and webhook
tests."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

from googleapiclient.errors import HttpError

from app.config import Settings
from app.mail_processing.gmail import Email, HistoryExpiredError
from app.utils.firestore_storage import (
    can_retry,
    completed_record,
    failed_record,
    new_record,
    skipped_record,
)
from app.mail_processing.mailbox import Mailbox

BOT = "bot@example.com"


def make_email(message_id: str = "m1", **overrides) -> Email:
    fields = {
        "id": message_id,
        "thread_id": f"t-{message_id}",
        "sender": "Jan Novák <jan@example.com>",
        "subject": "Order status",
        "body": "Where is my order?",
        "message_id": f"<{message_id}@mail.example.com>",
        "references": "",
        "received_at": datetime(2026, 11, 17, tzinfo=timezone.utc),
        "label_ids": ["INBOX", "UNREAD"],
        "headers": {},
    }
    fields.update(overrides)
    return Email(**fields)


def not_found() -> HttpError:
    return HttpError(MagicMock(status=404), b"not found")


class FakeGmail:
    def __init__(self, emails: list[Email], history_id: str = "100") -> None:
        self.emails = {e.id: e for e in emails}
        self.history: list[tuple[str, str]] = []  # (historyId, message ID)
        self.history_id = history_id
        self.history_expired = False
        self.sent_per_thread: dict[str, int] = {}
        self.sent: list[dict] = []
        self.read: list[str] = []
        self.attachment_data: dict[str, bytes] = {}  # attachment ID -> bytes
        self.downloaded: list[str] = []

    def add(self, email: Email) -> None:
        """Simulates a new INBOX message arriving."""
        self.emails[email.id] = email
        self.history_id = str(int(self.history_id) + 1)
        self.history.append((self.history_id, email.id))

    def get_history_id(self) -> str:
        return self.history_id

    def list_history(self, start: str) -> tuple[list[str], str]:
        if self.history_expired:
            raise HistoryExpiredError(start)
        ids = [mid for hid, mid in self.history if int(hid) > int(start)]
        return ids, self.history_id

    def list_unread_ids(self) -> list[str]:
        return [e.id for e in self.emails.values() if "UNREAD" in e.label_ids]

    def get_email(self, message_id: str) -> Email:
        if message_id not in self.emails:
            raise not_found()
        return self.emails[message_id]

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes:
        self.downloaded.append(attachment_id)
        if attachment_id not in self.attachment_data:
            raise not_found()
        return self.attachment_data[attachment_id]

    def count_sent_in_thread(self, thread_id: str) -> int:
        return self.sent_per_thread.get(thread_id, 0)

    def send(self, body: dict) -> dict:
        self.sent.append(body)
        thread = body["threadId"]
        self.sent_per_thread[thread] = self.sent_per_thread.get(thread, 0) + 1
        return {"id": f"sent-{len(self.sent)}"}

    def mark_read(self, message_id: str) -> None:
        self.read.append(message_id)
        self.emails[message_id].label_ids.remove("UNREAD")

    def watch(self, topic: str) -> dict:
        return {"historyId": self.history_id, "expiration": "1795000000000"}


class InMemoryStorage:
    """Same behaviour as FirestoreStorage, in memory, for tests."""

    def __init__(self, history_id: str | None = None) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.history_id = history_id

    def create_and_claim_message(
        self, message_id: str, data: dict[str, Any], retry_failed: bool = False
    ) -> bool:
        """Claims the email for processing; False if it is already stored, unless
        retry_failed and the stored record can_retry."""
        existing = self.docs.get(message_id)
        claimed = existing is None or (retry_failed and can_retry(existing))
        if claimed:
            self.docs[message_id] = new_record(data)
        return claimed

    def complete_message(self, message_id: str, reply_message_id: str) -> None:
        """Marks the email as answered."""
        self.docs[message_id].update(completed_record(reply_message_id))

    def fail_message(
        self, message_id: str, error_type: str, reply_message_id: str | None = None
    ) -> None:
        """Marks the email as failed, with only the error type."""
        self.docs[message_id].update(failed_record(error_type, reply_message_id))

    def skip_message(self, message_id: str, data: dict[str, Any], reason: str) -> None:
        """Records an email the filters rejected, unless it already has a record."""
        self.docs.setdefault(message_id, skipped_record(data, reason))

    def get_history_id(self) -> str | None:
        """The stored history ID, or None before the first inbox check."""
        return self.history_id

    def save_history_id(self, history_id: str) -> None:
        """Stores the historyId unless a newer one is already stored."""
        if self.history_id is None or int(history_id) > int(self.history_id):
            self.history_id = history_id


def make_gemini(answer: str = "Gemini answer") -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=answer)
    return client


def make_mailbox(gmail: FakeGmail, history_id: str | None = "100") -> Mailbox:
    return Mailbox(
        gmail=gmail,
        own_address=BOT,
        storage=InMemoryStorage(history_id),
        gemini=make_gemini(),
        settings=Settings(GEMINI_MODEL="gemini-test"),
    )
