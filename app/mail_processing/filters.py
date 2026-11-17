"""Decides which incoming emails the bot must not answer."""

from email.utils import parseaddr

from app.mail_processing.gmail import Email, GmailClient

MAX_BOT_REPLIES_PER_THREAD = 5

SKIPPED_LABELS = {"SENT", "SPAM", "TRASH", "DRAFT", "CATEGORY_PROMOTIONS"}
BULK_PRECEDENCE = {"bulk", "junk", "list", "auto_reply"}
AUTO_REPLY_HEADERS = ("x-autoreply", "x-autorespond", "x-auto-response-suppress")
LIST_HEADERS = ("list-id", "list-unsubscribe")
NO_REPLY_PREFIXES = ("noreply", "no-reply", "donotreply", "mailer-daemon", "postmaster")


def skip_reason(email: Email, own_address: str) -> str | None:
    """Returns why the email must not be answered, or None. Uses no Gmail calls."""
    address = parseaddr(email.sender)[1].lower()
    if address == own_address.lower():
        return "own message"
    if address.split("@")[0].startswith(NO_REPLY_PREFIXES):
        return "no-reply sender"
    labels = SKIPPED_LABELS.intersection(email.label_ids)
    if labels:
        return f"label {sorted(labels)[0]}"
    if "UNREAD" not in email.label_ids:
        return "already read"

    headers = email.headers
    if headers.get("auto-submitted", "no").lower() != "no":
        return "auto-submitted"
    if any(name in headers for name in AUTO_REPLY_HEADERS):
        return "auto-reply header"
    if headers.get("precedence", "").lower() in BULK_PRECEDENCE:
        return "bulk precedence"
    if any(name in headers for name in LIST_HEADERS):
        return "mailing list"
    if headers.get("return-path", "").strip() == "<>":
        return "bounce"
    return None


def thread_limit_reached(gmail: GmailClient, email: Email) -> bool:
    return gmail.count_sent_in_thread(email.thread_id) >= MAX_BOT_REPLIES_PER_THREAD
