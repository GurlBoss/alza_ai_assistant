"""Thin wrapper around the Gmail API and parsing of Gmail messages."""

import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import Settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


@dataclass
class Email:
    id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    message_id: str
    references: str
    received_at: datetime
    label_ids: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)  # lower-case names


class HistoryExpiredError(Exception):
    """The stored historyId is too old; Gmail no longer has history from it."""


def get_credentials(settings: Settings) -> Credentials:
    """Exchanges the refresh token for an access token."""
    missing = [
        name
        for name in (
            "GMAIL_CLIENT_ID",
            "GMAIL_CLIENT_SECRET",
            "GMAIL_OAUTH_REFRESH_TOKEN",
        )
        if not getattr(settings, name)
    ]
    if missing:
        raise RuntimeError(f"Missing settings: {', '.join(missing)}")
    creds = Credentials(
        token=None,
        refresh_token=settings.GMAIL_OAUTH_REFRESH_TOKEN,
        client_id=settings.GMAIL_CLIENT_ID,
        client_secret=settings.GMAIL_CLIENT_SECRET,
        token_uri=TOKEN_URI,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


class GmailClient:
    def __init__(self, service: Any) -> None:
        self.service = service

    @classmethod
    def from_credentials(cls, creds: Credentials) -> "GmailClient":
        return cls(build("gmail", "v1", credentials=creds, cache_discovery=False))

    def get_address(self) -> str:
        return self.service.users().getProfile(userId="me").execute()["emailAddress"]

    def get_history_id(self) -> str:
        return self.service.users().getProfile(userId="me").execute()["historyId"]

    def list_unread_ids(self) -> list[str]:
        """Unread INBOX message IDs, oldest first."""
        ids: list[str] = []
        request = (
            self.service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=500)
        )
        while request is not None:
            response = request.execute()
            ids.extend(m["id"] for m in response.get("messages", []))
            request = self.service.users().messages().list_next(request, response)
        return ids[::-1]  # Gmail lists newest first

    def iter_unread(self, own_address: str) -> Iterator[Email]:
        """Yields unread INBOX messages not sent by own_address, oldest first."""
        ids = self.list_unread_ids()
        log.info("Found %d unread message(s) in INBOX", len(ids))
        for message_id in ids:
            email = self.get_email(message_id)
            if parseaddr(email.sender)[1].lower() == own_address.lower():
                log.info("Skipping message %s sent by the bot itself", message_id)
                continue
            yield email

    def list_history(self, start_history_id: str) -> tuple[list[str], str]:
        """IDs of messages added to INBOX since start_history_id, and the latest historyId.

        Raises HistoryExpiredError when Gmail no longer has history that old.
        """
        ids: list[str] = []
        request = (
            self.service.users()
            .history()
            .list(
                userId="me",
                startHistoryId=start_history_id,
                historyTypes=["messageAdded"],
                labelId="INBOX",
                maxResults=500,
            )
        )
        latest = start_history_id
        while request is not None:
            try:
                response = request.execute()
            except HttpError as exc:
                if exc.resp.status == 404:
                    raise HistoryExpiredError(start_history_id) from exc
                raise
            latest = response.get("historyId", latest)
            for record in response.get("history", []):
                for added in record.get("messagesAdded", []):
                    if added["message"]["id"] not in ids:
                        ids.append(added["message"]["id"])
            request = self.service.users().history().list_next(request, response)
        return ids, latest

    def count_sent_in_thread(self, thread_id: str) -> int:
        """Number of messages in the thread sent from this mailbox."""
        thread = (
            self.service.users()
            .threads()
            .get(userId="me", id=thread_id, format="minimal")
            .execute()
        )
        return sum("SENT" in m.get("labelIds", []) for m in thread.get("messages", []))

    def watch(self, topic: str) -> dict[str, Any]:
        """Starts or renews push notifications; returns historyId and expiration."""
        return (
            self.service.users()
            .watch(userId="me", body={"topicName": topic, "labelIds": ["INBOX"]})
            .execute()
        )

    def get_email(self, message_id: str) -> Email:
        message = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return parse_message(message)

    def send(self, body: dict[str, str]) -> dict[str, Any]:
        return self.service.users().messages().send(userId="me", body=body).execute()

    def mark_read(self, message_id: str) -> None:
        self.service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()


def parse_message(message: dict[str, Any]) -> Email:
    payload = message["payload"]
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    return Email(
        id=message["id"],
        thread_id=message["threadId"],
        sender=headers.get("from", ""),
        subject=headers.get("subject", ""),
        body=_extract_body(payload),
        message_id=headers.get("message-id", ""),
        references=headers.get("references", ""),
        received_at=datetime.fromtimestamp(
            int(message["internalDate"]) / 1000, tz=timezone.utc
        ),
        label_ids=message.get("labelIds", []),
        headers=headers,
    )


def _extract_body(payload: dict[str, Any]) -> str:
    """Returns the text/plain part, or text/html with tags stripped as a fallback."""
    plain = _find_part(payload, "text/plain")
    if plain is not None:
        return plain
    html = _find_part(payload, "text/html")
    if html is not None:
        return re.sub(r"<[^>]+>", " ", html)
    return ""


def _find_part(part: dict[str, Any], mime_type: str) -> str | None:
    if part.get("mimeType") == mime_type and part.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(part["body"]["data"]).decode(
            "utf-8", errors="replace"
        )
    for child in part.get("parts", []):
        text = _find_part(child, mime_type)
        if text is not None:
            return text
    return None
