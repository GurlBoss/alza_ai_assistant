"""In-memory Gmail and Gemini doubles shared by the sync and webhook tests."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from googleapiclient.errors import HttpError

from app.config import Settings
from app.mail_processing.gmail import Email, HistoryExpiredError
from app.mail_processing.store import InMemoryStore, InMemorySyncState
from app.mail_processing.sync import Mailbox

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


def make_gemini(answer: str = "Gemini answer") -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=answer)
    return client


def make_mailbox(gmail: FakeGmail, history_id: str | None = "100") -> Mailbox:
    return Mailbox(
        gmail=gmail,
        own_address=BOT,
        store=InMemoryStore(),
        sync_state=InMemorySyncState(history_id),
        gemini=make_gemini(),
        settings=Settings(GEMINI_MODEL="gemini-test"),
    )
