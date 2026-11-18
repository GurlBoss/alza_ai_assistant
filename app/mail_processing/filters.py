"""Decides which incoming emails the bot must not answer, and which attachments
it must not use."""

from email.utils import parseaddr

from app.mail_processing.gmail import AttachmentRef, Email, GmailClient

MAX_BOT_REPLIES_PER_THREAD = 5

SKIPPED_LABELS = {"SENT", "SPAM", "TRASH", "DRAFT", "CATEGORY_PROMOTIONS"}
BULK_PRECEDENCE = {"bulk", "junk", "list", "auto_reply"}
AUTO_REPLY_HEADERS = ("x-autoreply", "x-autorespond", "x-auto-response-suppress")
LIST_HEADERS = ("list-id", "list-unsubscribe")
NO_REPLY_PREFIXES = ("noreply", "no-reply", "donotreply", "mailer-daemon", "postmaster")

# Reasons why an email is not answered (stored and logged as "SKIPPED:<reason>")..
OWN_MESSAGE = "own message"
NO_REPLY_SENDER = "no-reply sender"
SKIPPED_LABEL = "label"  # followed by the label, e.g. "label SPAM"
ALREADY_READ = "already read"
AUTO_SUBMITTED = "auto-submitted"
AUTO_REPLY_HEADER = "auto-reply header"
BULK_MAIL = "bulk precedence"
MAILING_LIST = "mailing list"
BOUNCE = "bounce"
THREAD_REPLY_LIMIT = "thread reply limit"


def skip_reason(email: Email, own_address: str) -> str | None:
    """Returns why the email must not be answered, or None. Uses no Gmail calls."""
    address = parseaddr(email.sender)[1].lower()
    labels = SKIPPED_LABELS.intersection(email.label_ids)
    headers = email.headers

    reason = None
    if address == own_address.lower():
        reason = OWN_MESSAGE
    elif address.split("@")[0].startswith(NO_REPLY_PREFIXES):
        reason = NO_REPLY_SENDER
    elif labels:
        reason = f"{SKIPPED_LABEL} {sorted(labels)[0]}"
    elif "UNREAD" not in email.label_ids:
        reason = ALREADY_READ
    elif headers.get("auto-submitted", "no").lower() != "no":
        reason = AUTO_SUBMITTED
    elif any(name in headers for name in AUTO_REPLY_HEADERS):
        reason = AUTO_REPLY_HEADER
    elif headers.get("precedence", "").lower() in BULK_PRECEDENCE:
        reason = BULK_MAIL
    elif any(name in headers for name in LIST_HEADERS):
        reason = MAILING_LIST
    elif headers.get("return-path", "").strip() == "<>":
        reason = BOUNCE
    return reason


def thread_limit_reached(gmail: GmailClient, email: Email) -> bool:
    """Whether the bot has already sent MAX_BOT_REPLIES_PER_THREAD replies in this
    thread (stops reply loops)."""
    return gmail.count_sent_in_thread(email.thread_id) >= MAX_BOT_REPLIES_PER_THREAD


# --- attachments ------------------------------------------------------------------

# Supported: images (JPEG, PNG, WebP), audio (MP3, M4A, WAV), PDF and plain text.
# MP4 container brands that mean M4A audio; any other brand (video) is unsupported.
M4A_BRANDS = {b"M4A ", b"M4B ", b"M4P ", b"F4A ", b"F4B "}
# Larger images are always treated as real pictures (photos, pasted screenshots).
SIGNATURE_IMAGE_MAX_BYTES = 30 * 1024

# Reasons why an attachment is not used (mentioned in the reply).
UNSUPPORTED = "unsupported file type"
TOO_LARGE = "file is too large"
TOTAL_TOO_LARGE = "attachments together are too large"


def filter_type(data: bytes, declared: str = "") -> str | None:
    """The real MIME type from the file content, or None when it is not one of
    the supported formats (images, MP3/M4A/WAV, PDF, plain text).

    declared only decides whether text-looking data is meant as plain text; it
    never makes binary data acceptable.
    """
    mime_type = None
    if data.startswith(b"%PDF-"):
        mime_type = "application/pdf"
    elif data.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime_type = "image/webp"
    elif data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        mime_type = "audio/wav"
    elif data[4:8] == b"ftyp" and data[8:12] in M4A_BRANDS:
        mime_type = "audio/mp4"
    elif data.startswith(b"ID3") or _is_mpeg_audio_frame(data):
        mime_type = "audio/mpeg"
    elif declared.startswith("text/plain") and b"\x00" not in data[:4096]:
        mime_type = "text/plain"
    return mime_type


def _is_mpeg_audio_frame(data: bytes) -> bool:
    """Whether the data starts with an MPEG audio frame header (an MP3 without an ID3
    tag)."""
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
    shown = ref.inline or (ref.content_id != "" and ref.content_id in referenced_cids)
    return (
        ref.mime_type.startswith("image/")
        and ref.size <= SIGNATURE_IMAGE_MAX_BYTES
        and shown
    )


def size_rejection(ref: AttachmentRef, max_file: int, room: int) -> str | None:
    """Why the attachment is too big to download (per file, or for the room left in
    the per-email total), or None."""
    reason = None
    if ref.size > max_file:
        reason = TOO_LARGE
    elif ref.size > room:
        reason = TOTAL_TOO_LARGE
    return reason


def content_rejection(mime_type: str | None) -> str | None:
    """Why the downloaded attachment can't be used (its type from filter_type), or
    None."""
    reason = None
    if mime_type is None:
        reason = UNSUPPORTED
    return reason
