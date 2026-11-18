"""Downloads and checks email attachments. Everything stays in memory."""

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.mail_processing.gmail import AttachmentRef, Email, GmailClient

log = logging.getLogger(__name__)

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
AUDIO_TYPES = {"audio/mpeg", "audio/mp4", "audio/wav"}
DOCUMENT_TYPES = {"application/pdf", "text/plain"}

# Larger images are always treated as real pictures (photos, pasted screenshots).
SIGNATURE_IMAGE_MAX_BYTES = 30 * 1024

UNSUPPORTED = "unsupported file type"
TOO_LARGE = "file is too large"
TOTAL_TOO_LARGE = "attachments together are too large"
EXTRA_AUDIO = "only one audio file per email can be listened to"
DOWNLOAD_FAILED = "could not be downloaded"

# MP4 container brands that mean M4A audio; any other brand (video) is unsupported.
M4A_BRANDS = {b"M4A ", b"M4B ", b"M4P ", b"F4A ", b"F4B "}


@dataclass
class Attachment:
    filename: str
    mime_type: str  # verified by content
    data: bytes


@dataclass
class RejectedAttachment:
    filename: str
    reason: str


@dataclass
class LoadedAttachments:
    accepted: list[Attachment] = field(default_factory=list)
    rejected: list[RejectedAttachment] = field(default_factory=list)


def detect_type(data: bytes, declared: str = "") -> str | None:
    """The real MIME type from the file content, or None when it is not one of
    the supported formats (images, MP3/M4A/WAV, PDF, plain text).

    declared only decides whether text-looking data is meant as plain text; it
    never makes binary data acceptable.
    """
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[4:8] == b"ftyp" and data[8:12] in M4A_BRANDS:
        return "audio/mp4"
    if data.startswith(b"ID3") or _is_mpeg_audio_frame(data):
        return "audio/mpeg"
    if declared.startswith("text/plain") and b"\x00" not in data[:4096]:
        return "text/plain"
    return None


def _is_mpeg_audio_frame(data: bytes) -> bool:
    # 11-bit frame sync, a valid MPEG version and a layer other than 00 (= AAC).
    return (
        len(data) >= 2
        and data[0] == 0xFF
        and data[1] & 0xE0 == 0xE0
        and (data[1] >> 3) & 0b11 != 0b01
        and (data[1] >> 1) & 0b11 != 0b00
    )


def is_signature_image(ref: AttachmentRef, referenced_cids: set[str]) -> bool:
    """Logos and icons from signatures: small images shown inline or by the HTML."""
    if not ref.mime_type.startswith("image/") or ref.size > SIGNATURE_IMAGE_MAX_BYTES:
        return False
    return ref.inline or (ref.content_id != "" and ref.content_id in referenced_cids)


def load_attachments(
    gmail: GmailClient, email: Email, settings: Settings
) -> LoadedAttachments:
    """Downloads the usable attachments, in order; never raises for one attachment.

    Rules: per-file and per-email size limits (checked from the metadata before
    downloading), types verified by content, at most one audio file.
    """
    started = time.monotonic()
    max_file = int(settings.MAX_ATTACHMENT_MB * 1024 * 1024)
    max_total = int(settings.MAX_TOTAL_ATTACHMENTS_MB * 1024 * 1024)
    result = LoadedAttachments()
    total = 0

    for number, ref in enumerate(email.attachments, start=1):
        if is_signature_image(ref, email.referenced_cids):
            _log(email, number, ref, None, "skipped (signature image)")
            continue

        detected = None
        if ref.size > max_file:
            reason = TOO_LARGE
        elif total + ref.size > max_total:
            reason = TOTAL_TOO_LARGE
        else:
            data = _download(gmail, email, ref)
            if data is None:
                reason = DOWNLOAD_FAILED
            else:
                detected = detect_type(data, ref.mime_type)
                reason = None if detected else UNSUPPORTED
                if reason is None and detected in AUDIO_TYPES and _has_audio(result):
                    reason = EXTRA_AUDIO
                if reason is None:
                    total += len(data)
                    result.accepted.append(Attachment(ref.filename, detected, data))
                    _log(email, number, ref, detected, "accepted")
                    continue

        result.rejected.append(RejectedAttachment(ref.filename, reason))
        _log(email, number, ref, detected, f"rejected ({reason})")

    log.info(
        "Attachments of %s: %d accepted, %d rejected in %.1f s",
        email.id,
        len(result.accepted),
        len(result.rejected),
        time.monotonic() - started,
    )
    return result


def _has_audio(result: LoadedAttachments) -> bool:
    return any(item.mime_type in AUDIO_TYPES for item in result.accepted)


def _download(gmail: GmailClient, email: Email, ref: AttachmentRef) -> bytes | None:
    try:
        return gmail.get_attachment(email.id, ref.attachment_id)
    except Exception as exc:
        log.warning("Attachment download failed: %s", type(exc).__name__)
        return None


def _log(
    email: Email, number: int, ref: AttachmentRef, detected: str | None, outcome: str
) -> None:
    # Type and size only: no file names or contents.
    log.info(
        "Attachment %d of %s: declared=%s detected=%s size=%d -> %s",
        number,
        email.id,
        ref.mime_type or "-",
        detected or "-",
        ref.size,
        outcome,
    )
