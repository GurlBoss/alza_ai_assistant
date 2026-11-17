"""Builds and sends a reply in the original Gmail thread."""

import base64
from email.message import EmailMessage

from app.mail_processing.gmail import Email, GmailClient


def reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def build_reply(original: Email, text: str, own_address: str) -> dict[str, str]:
    """Returns the Gmail API send body for a reply to the original's sender only."""
    message = EmailMessage()
    message["From"] = own_address
    message["To"] = original.sender
    message["Subject"] = reply_subject(original.subject)
    if original.message_id:
        message["In-Reply-To"] = original.message_id
        message["References"] = " ".join(
            filter(None, [original.references, original.message_id])
        )
    message.set_content(text)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw, "threadId": original.thread_id}


def send_reply(gmail: GmailClient, original: Email, text: str, own_address: str) -> str:
    """Sends the reply and returns the id of the sent message."""
    return gmail.send(build_reply(original, text, own_address))["id"]
