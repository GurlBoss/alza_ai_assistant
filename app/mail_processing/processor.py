"""Answers one email: claim in the store, Gemini reply in the thread, mark as read."""

import logging
from email.utils import parseaddr
from typing import Any

from app.config import Settings
from app.mail_processing.attachments import load_attachments
from app.mail_processing.gmail import Email, GmailClient
from app.ai_core.llm import generate_reply
from app.knowledge_base.rag import build_rag_tool
from app.mail_processing.reply import send_reply
from app.mail_processing.store import COMPLETED, FAILED, EmailStore

log = logging.getLogger(__name__)


FALLBACK_REPLY = """\
Dobrý den,
děkujeme za Váš e-mail. Momentálně na něj nedokážeme automaticky odpovědět,
ozve se Vám náš kolega.

Hello,
thank you for your email. We can't answer it automatically right now,
a colleague will get back to you.

Alza AI Assistant"""


SKIPPED = "SKIPPED"  # already in the store, nothing done


def sender_domain(sender: str) -> str:
    return parseaddr(sender)[1].rpartition("@")[2].lower()


def process_email(
    email: Email,
    gmail: GmailClient,
    store: EmailStore,
    settings: Settings,
    own_address: str,
    gemini: Any,
) -> str:
    """Returns COMPLETED, FAILED or SKIPPED (already in the store).

    Once the email is claimed, errors are recorded as FAILED and not raised, so the
    caller can treat the email as handled. Errors before that (e.g. Firestore
    unavailable) are raised: nothing was recorded and the email must be retried.
    """
    claimed = store.claim(
        email.id,
        {
            "thread_id": email.thread_id,
            "sender_domain": sender_domain(email.sender),
            "subject_length": len(email.subject),
            "received_at": email.received_at,
        },
    )
    if not claimed:
        log.info("Message %s already processed or failed, skipping", email.id)
        return SKIPPED
    log.info("Message %s claimed in the store (PROCESSING)", email.id)

    gemini_error = None
    try:
        loaded = load_attachments(gmail, email, settings)
        reply_text = generate_reply(
            gemini,
            settings.GEMINI_MODEL,
            email.subject,
            email.body,
            loaded.accepted,
            loaded.rejected,
            rag_tool=build_rag_tool(settings),
            rag_debug=settings.RAG_DEBUG,
        )
    except Exception as exc:
        # Only the exception type: messages can contain email content or tokens.
        gemini_error = f"Gemini:{type(exc).__name__}"
        log.error("Gemini failed for %s (%s), sending fallback", email.id, gemini_error)
        reply_text = FALLBACK_REPLY

    try:
        reply_id = send_reply(gmail, email, reply_text, own_address)
    except Exception as exc:
        store.fail(email.id, type(exc).__name__)
        log.error("Message %s FAILED: %s", email.id, type(exc).__name__)
        return FAILED

    if gemini_error:
        store.fail(email.id, gemini_error, reply_id)
        log.info("Fallback reply sent: id=%s (FAILED)", reply_id)
    else:
        store.complete(email.id, reply_id)
        log.info("Reply sent: id=%s threadId=%s (COMPLETED)", reply_id, email.thread_id)

    # The customer got a reply either way; if this fails, nothing is resent.
    try:
        gmail.mark_read(email.id)
        log.info("Original message %s marked as read", email.id)
    except Exception as exc:
        log.warning("Could not mark %s as read: %s", email.id, type(exc).__name__)
    return FAILED if gemini_error else COMPLETED
