"""Downloads email attachments and decides which go to Gemini, using the rules in
filters.py. Everything stays in memory."""

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.mail_processing.filters import (
    DOWNLOAD_FAILED,
    SIGNATURE_IMAGE,
    content_rejection,
    filter_type,
    is_signature_image,
    size_rejection,
)
from app.mail_processing.gmail import AttachmentRef, Email, GmailClient

log = logging.getLogger(__name__)

# Attachment states.
ACCEPTED = "ACCEPTED"  # sent to Gemini
REJECTED = "REJECTED"  # not used; the reply mentions it with the reason


@dataclass
class Attachment:
    """An email attachment: ACCEPTED with its data and verified type, or REJECTED
    with the reason."""

    filename: str
    mime_type: str = ""  # verified by content
    data: bytes = b""
    state: str = ACCEPTED
    reason: str = ""  # why it was rejected


def load_attachments(
    gmail: GmailClient, email: Email, settings: Settings
) -> list[Attachment]:
    """Accepts or rejects each attachment; never raises for one attachment.

    The smallest are checked first, so as many as possible fit into the per-email
    limit; the result keeps the email's order. Signature images are left out.
    """
    started = time.monotonic()
    max_file = int(settings.MAX_ATTACHMENT_MB * 1024 * 1024)
    max_total = int(settings.MAX_TOTAL_ATTACHMENTS_MB * 1024 * 1024)
    loaded: dict[int, Attachment] = {}  # position in the email -> attachment

    numbered = enumerate(email.attachments, start=1)
    for number, ref in sorted(numbered, key=lambda pair: pair[1].size):
        if is_signature_image(ref, email.referenced_cids):
            _log(email, number, ref, f"skipped ({SIGNATURE_IMAGE})")
        else:
            accepted = [item for item in loaded.values() if item.state == ACCEPTED]
            room = max_total - sum(len(item.data) for item in accepted)
            attachment = _load_one(gmail, email, ref, max_file, room)
            loaded[number] = attachment
            if attachment.state == ACCEPTED:
                _log(email, number, ref, f"accepted as {attachment.mime_type}")
            else:
                _log(email, number, ref, f"rejected ({attachment.reason})")

    attachments = [loaded[number] for number in sorted(loaded)]
    log.info(
        "Attachments of %s: %d accepted, %d rejected in %.1f s",
        email.id,
        sum(item.state == ACCEPTED for item in attachments),
        sum(item.state == REJECTED for item in attachments),
        time.monotonic() - started,
    )
    return attachments


def _load_one(
    gmail: GmailClient,
    email: Email,
    ref: AttachmentRef,
    max_file: int,
    room: int,
) -> Attachment:
    """Checks the size, downloads the attachment and checks its content type."""
    data, mime_type = b"", None
    reason = size_rejection(ref, max_file, room)
    if reason is None:
        try:
            data = gmail.get_attachment(email.id, ref.attachment_id)
        except Exception as exc:
            log.warning("Attachment download failed: %s", type(exc).__name__)
            reason = DOWNLOAD_FAILED
    if reason is None:
        mime_type = filter_type(data, ref.mime_type)
        reason = content_rejection(mime_type)

    if reason is None:
        attachment = Attachment(ref.filename, mime_type, data)
    else:
        attachment = Attachment(ref.filename, state=REJECTED, reason=reason)
    return attachment


def _log(email: Email, number: int, ref: AttachmentRef, outcome: str) -> None:
    """Logs the decision about one attachment: its declared type, size and outcome,
    never its name or content."""
    log.info(
        "Attachment %d of %s: declared=%s size=%d -> %s",
        number,
        email.id,
        ref.mime_type or "-",
        ref.size,
        outcome,
    )
