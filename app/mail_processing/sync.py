"""Processes everything that arrived in the INBOX since the stored historyId."""

import logging
from dataclasses import dataclass, field
from typing import Any

from googleapiclient.errors import HttpError

from app.config import Settings
from app.knowledge_base.rag import KnowledgeBase
from app.mail_processing.filters import skip_reason, thread_limit_reached
from app.mail_processing.gmail import GmailClient, HistoryExpiredError
from app.mail_processing.processor import process_email
from app.mail_processing.store import EmailStore, SyncState

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    start_history_id: str | None
    saved_history_id: str | None
    results: dict[str, str] = field(default_factory=dict)  # message ID -> outcome


@dataclass
class Mailbox:
    """Everything a sync needs; created once per instance in app.main."""

    gmail: GmailClient
    own_address: str
    store: EmailStore
    sync_state: SyncState
    gemini: Any
    settings: Settings
    knowledge: KnowledgeBase | None = None


def sync_mailbox(
    mailbox: Mailbox, notified_history_id: str | None = None
) -> SyncResult:
    """Handles all new INBOX messages, then stores the new historyId.

    Every message is answered, filtered out or recorded as FAILED before the
    historyId moves forward. Any error raised from here means something was not
    handled; the historyId stays put and the next notification retries it.
    """
    gmail = mailbox.gmail
    start = mailbox.sync_state.get_history_id()
    if start is None:
        # First run without a watch: start from now, nothing older is replayed.
        latest = notified_history_id or gmail.get_history_id()
        mailbox.sync_state.save_history_id(latest)
        log.warning("No stored historyId, starting from %s", latest)
        return SyncResult(None, latest)

    try:
        message_ids, latest = gmail.list_history(start)
    except HistoryExpiredError:
        # History is kept for about a week; fall back to all unread INBOX mail.
        # Already answered messages are skipped by the store.
        log.error("History from %s expired, falling back to unread INBOX", start)
        latest = gmail.get_history_id()
        message_ids = gmail.list_unread_ids()
    log.info("History %s -> %s: %d new message(s)", start, latest, len(message_ids))

    result = SyncResult(start, None)
    for message_id in message_ids:
        result.results[message_id] = handle_message(mailbox, message_id)

    new_history_id = max(latest, notified_history_id or latest, key=int)
    mailbox.sync_state.save_history_id(new_history_id)
    result.saved_history_id = new_history_id
    return result


def handle_message(mailbox: Mailbox, message_id: str) -> str:
    """Returns the outcome for one message; raises only when nothing was recorded."""
    try:
        email = mailbox.gmail.get_email(message_id)
    except HttpError as exc:
        if exc.resp.status == 404:
            log.info("Message %s no longer exists, skipping", message_id)
            return "DELETED"
        raise

    reason = skip_reason(email, mailbox.own_address)
    if reason is None and thread_limit_reached(mailbox.gmail, email):
        reason = "thread reply limit"
    if reason:
        log.info("Message %s filtered: %s", message_id, reason)
        return f"FILTERED:{reason}"

    return process_email(
        email,
        mailbox.gmail,
        mailbox.store,
        mailbox.settings,
        mailbox.own_address,
        mailbox.gemini,
        mailbox.knowledge,
    )
