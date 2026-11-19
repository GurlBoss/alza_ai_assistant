"""The bot's mailbox: answers everything new in the INBOX.

Mailbox.check_inbox -> handle_message -> process_email.
"""

import logging
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any

from googleapiclient.errors import HttpError

from app.ai_core.llm import generate_reply
from app.config import Settings
from app.ai_core.rag import build_rag_tool
from app.mail_processing.attachments import load_attachments
from app.mail_processing.filters import (
    ALREADY_READ,
    THREAD_REPLY_LIMIT,
    skip_reason,
    thread_limit_reached,
)
from app.mail_processing.gmail import Email, GmailClient, HistoryExpiredError
from app.mail_processing.reply import send_reply
from app.utils.firestore_storage import (
    COMPLETED,
    FAILED,
    PROCESSING,
    SKIPPED,
    Storage,
)

log = logging.getLogger(__name__)


FALLBACK_REPLY = """\
Dobrý den,
děkujeme za Váš e-mail.

Na Váš dotaz bohužel v tuto chvíli nedokážeme odpovědět automaticky. Vaši zprávu jsme proto předali našim operátorům, kteří se jí budou co nejdříve věnovat.
Jakmile to bude možné, ozve se Vám některý z našich kolegů.

Děkujeme za trpělivost.

Alza AI Assistant"""

# Outcomes of handle_message, besides COMPLETED and FAILED from the storage and
# SKIPPED:<reason> for emails the filters rejected, e.g. "SKIPPED:bounce".
DUPLICATE = "DUPLICATE"  # already in the storage, nothing done
DELETED = "DELETED"  # the message no longer exists in Gmail


@dataclass
class CheckInboxResult:
    """What one check_inbox run did: the history ID it started from, the one it
    saved, and the outcome per message."""

    start_history_id: str | None
    saved_history_id: str | None
    results: dict[str, str] = field(default_factory=dict)  # message ID -> outcome


def sender_domain(sender: str) -> str:
    """The domain of the sender's address, stored instead of the full address."""
    return parseaddr(sender)[1].rpartition("@")[2].lower()


def record_data(email: Email) -> dict[str, Any]:
    """What is stored about an email in processed_emails: no address or content."""
    return {
        "thread_id": email.thread_id,
        "sender_domain": sender_domain(email.sender),
        "subject_length": len(email.subject),
        "received_at": email.received_at,
    }


@dataclass
class Mailbox:
    """Everything needed to handle the bot's mailbox. Built per request in
    app.main: a new Gmail client each time, the other clients shared."""

    gmail: GmailClient
    own_address: str
    storage: Storage
    gemini: Any
    settings: Settings

    def check_inbox(
        self, notified_history_id: str | None = None, recovery: bool = False
    ) -> CheckInboxResult:
        """Handles all new INBOX messages, then stores the new history ID.

        Every message is answered, filtered out or recorded as FAILED before the
        history ID moves forward. Any error raised from here means something was not
        handled; the history ID stays put and the next notification retries it.
        """
        start = self.storage.get_history_id()
        message_ids, latest = self._messages_to_check(start, recovery)
        log.info("History %s -> %s: %d message(s)", start, latest, len(message_ids))

        result = CheckInboxResult(start, None)
        for message_id in message_ids:
            result.results[message_id] = self.handle_message(message_id, recovery)

        new_history_id = max(latest, notified_history_id or latest, key=int)
        self.storage.save_history_id(new_history_id)
        result.saved_history_id = new_history_id
        return result

    def _messages_to_check(
        self, start: str | None, recovery: bool = False
    ) -> tuple[list[str], str]:
        """Message IDs to check and the current history ID: the history since
        start, or all unread INBOX mail for a recovery or when there is no usable
        start."""
        if recovery:
            log.info("Recovery: checking all unread mail")
        elif start is None:
            log.warning("No stored history ID, checking all unread mail")
        else:
            try:
                return self.gmail.list_history(start)
            except HistoryExpiredError:
                # Gmail keeps history for about a week.
                log.error("History from %s expired, checking all unread mail", start)

        # Answered emails are read, so unread = not handled yet; duplicates are
        # skipped by their processed_emails record.
        latest = self.gmail.get_history_id()
        return self.gmail.list_unread_ids(), latest

    def handle_message(self, message_id: str, recovery: bool = False) -> str:
        """Returns the outcome for one message; raises only when nothing was
        recorded. In a recovery, an email that FAILED without any reply is retried."""
        outcome = None
        try:
            email = self.gmail.get_email(message_id)
        except HttpError as exc:
            if exc.resp.status != 404:
                raise
            log.info("Message %s: %s (no longer in Gmail)", message_id, DELETED)
            outcome = DELETED

        if outcome is None:
            reason = skip_reason(email, self.own_address)
            if reason is None and thread_limit_reached(self.gmail, email):
                reason = THREAD_REPLY_LIMIT
            if reason:
                outcome = self.skip_email(email, reason)
            else:
                outcome = self.process_email(email, retry_failed=recovery)
        return outcome

    def skip_email(self, email: Email, reason: str) -> str:
        """Records an email the filters rejected as SKIPPED and marks it as read, so
        a recovery does not check it again. Already read emails are left alone."""
        log.info("Message %s: %s:%s", email.id, SKIPPED, reason)
        if reason != ALREADY_READ:
            self.storage.skip_message(email.id, record_data(email), reason)
            try:
                self.gmail.mark_read(email.id)
            except Exception as exc:
                log.warning(
                    "Could not mark %s as read: %s", email.id, type(exc).__name__
                )
        return f"{SKIPPED}:{reason}"

    def process_email(self, email: Email, retry_failed: bool = False) -> str:
        """Returns COMPLETED, FAILED or DUPLICATE (already in the storage). With
        retry_failed, an email that FAILED without any reply is claimed again.

        Once the email is claimed, errors are recorded as FAILED and not raised, so
        the caller can treat the email as handled. Errors before that (e.g.
        Firestore unavailable) are raised: nothing was recorded and the email must
        be retried.
        """
        claimed = self.storage.create_and_claim_message(
            email.id, record_data(email), retry_failed=retry_failed
        )
        if not claimed:
            log.info("Message %s: %s (already in the storage)", email.id, DUPLICATE)
            return DUPLICATE
        log.info("Message %s claimed in the storage (%s)", email.id, PROCESSING)

        gemini_error = None
        try:
            attachments = load_attachments(self.gmail, email, self.settings)
            reply_text = generate_reply(
                self.gemini,
                self.settings.GEMINI_MODEL,
                email.subject,
                email.body,
                attachments,
                rag_tool=build_rag_tool(self.settings),
                rag_debug=self.settings.RAG_DEBUG,
            )
        except Exception as exc:
            # Only the exception type: messages can contain email content or tokens.
            gemini_error = f"Gemini:{type(exc).__name__}"
            log.error(
                "Gemini failed for %s (%s), sending fallback", email.id, gemini_error
            )
            reply_text = FALLBACK_REPLY

        try:
            reply_id = send_reply(self.gmail, email, reply_text, self.own_address)
        except Exception as exc:
            self.storage.fail_message(email.id, type(exc).__name__)
            log.error("Message %s %s: %s", email.id, FAILED, type(exc).__name__)
            return FAILED

        if gemini_error:
            self.storage.fail_message(email.id, gemini_error, reply_id)
            log.info("Fallback reply sent: id=%s (%s)", reply_id, FAILED)
        else:
            self.storage.complete_message(email.id, reply_id)
            log.info(
                "Reply sent: id=%s threadId=%s (%s)",
                reply_id,
                email.thread_id,
                COMPLETED,
            )

        # The sender got a reply either way; if this fails, nothing is resent.
        try:
            self.gmail.mark_read(email.id)
            log.info("Original message %s marked as read", email.id)
        except Exception as exc:
            log.warning("Could not mark %s as read: %s", email.id, type(exc).__name__)
        return FAILED if gemini_error else COMPLETED
