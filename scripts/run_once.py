"""Answers the oldest unread email once, end to end, with Gemini.

Run from the repository root: python -m scripts.run_once
"""

import logging

from app.config import Settings
from app.mail_processing.gmail import GmailClient, get_credentials
from app.ai_core.llm import make_client
from app.mail_processing.processor import SKIPPED, process_email
from app.mail_processing.store import FirestoreStore

log = logging.getLogger("run_once")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()
    if not settings.GCP_PROJECT_ID:
        raise SystemExit("Set GCP_PROJECT_ID in .env")
    log.info(
        "1. Config loaded: project=%s region=%s",
        settings.GCP_PROJECT_ID,
        settings.REGION,
    )

    creds = get_credentials(settings)
    log.info("2. Gmail access token obtained (expires %s UTC)", creds.expiry)

    gmail = GmailClient.from_credentials(creds)
    own_address = gmail.get_address()
    log.info("   Mailbox: %s", own_address)
    store = FirestoreStore(settings.GCP_PROJECT_ID)
    gemini = make_client(settings)
    log.info(
        "   Gemini: model=%s region=%s",
        settings.GEMINI_MODEL,
        settings.GEMINI_REGION,
    )

    # Oldest unread first; emails already in the store (done or failed) are skipped.
    for email in gmail.iter_unread(own_address):
        log.info("3. Unread message: id=%s", email.id)
        log.info(
            "4. From=%s subject=%r threadId=%s Message-ID=%s references=%d body_chars=%d",
            email.sender,
            email.subject,
            email.thread_id,
            email.message_id,
            len(email.references.split()),
            len(email.body),
        )
        result = process_email(email, gmail, store, settings, own_address, gemini)
        if result != SKIPPED:
            return
    print("No unread email")


if __name__ == "__main__":
    main()
