"""Answers the oldest unread email once, end to end, from the terminal.

Run from the repository root: python -m scripts.run_once
Goes through unread INBOX emails, oldest first, with the same filters and
processing as the webhook, and stops after the first one it answers.
"""

import logging

from app.ai_core.llm import make_client
from app.config import Settings
from app.mail_processing.gmail import GmailClient, get_credentials
from app.mail_processing.processor import Mailbox
from app.mail_processing.store import (
    COMPLETED,
    FAILED,
    FirestoreStore,
    FirestoreSyncState,
)
from app.mail_processing.sync import handle_message

log = logging.getLogger("run_once")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()
    if not settings.GCP_PROJECT_ID:
        raise SystemExit("Set GCP_PROJECT_ID in .env")

    gmail = GmailClient.from_credentials(get_credentials(settings))
    mailbox = Mailbox(
        gmail=gmail,
        own_address=gmail.get_address(),
        store=FirestoreStore(settings.GCP_PROJECT_ID),
        sync_state=FirestoreSyncState(settings.GCP_PROJECT_ID),
        gemini=make_client(settings),
        settings=settings,
    )
    log.info(
        "Mailbox %s, Gemini %s (%s)",
        mailbox.own_address,
        settings.GEMINI_MODEL,
        settings.GEMINI_REGION,
    )

    for message_id in gmail.list_unread_ids():
        result = handle_message(mailbox, message_id)
        log.info("Message %s: %s", message_id, result)
        if result in (COMPLETED, FAILED):
            return
    print("No unread email to answer")


if __name__ == "__main__":
    main()
