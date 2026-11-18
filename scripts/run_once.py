"""Answers the oldest unread email once, end to end, from the terminal.

Run from the repository root: python -m scripts.run_once
Goes through unread INBOX emails, oldest first, with the same filters and
processing as the webhook, and stops after the first one it answers.
"""

import logging

from app.ai_core.llm import make_client
from app.config import Settings
from app.mail_processing.gmail import GmailClient, get_credentials
from app.mail_processing.mailbox import Mailbox
from app.utils.firestore_storage import COMPLETED, FAILED, FirestoreStorage

log = logging.getLogger("run_once")


def main(recovery: bool = False) -> None:
    """Builds the mailbox from .env and answers the oldest unread email that passes the
    filters. With recovery, an email that FAILED without any reply is retried."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()

    gmail = GmailClient.from_credentials(get_credentials(settings))
    mailbox = Mailbox(
        gmail=gmail,
        own_address=gmail.get_address(),
        storage=FirestoreStorage(settings.GCP_PROJECT_ID),
        gemini=make_client(settings),
        settings=settings,
    )
    log.info(
        "Mailbox %s, Gemini %s (%s), recovery %s",
        mailbox.own_address,
        settings.GEMINI_MODEL,
        settings.GEMINI_REGION,
        "on" if recovery else "off",
    )

    for message_id in gmail.list_unread_ids():
        result = mailbox.handle_message(message_id, recovery)
        log.info("Message %s: %s", message_id, result)
        if result in (COMPLETED, FAILED):
            return
    print("No unread email to answer")


if __name__ == "__main__":
    main()
