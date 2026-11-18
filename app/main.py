import base64
import binascii
import json
import logging
from dataclasses import replace
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from google.oauth2.credentials import Credentials
from starlette.concurrency import run_in_threadpool

from app.ai_core.llm import make_client
from app.config import settings
from app.mail_processing.gmail import GmailClient, get_credentials
from app.mail_processing.mailbox import Mailbox
from app.utils.auth import verify_oidc
from app.utils.firestore_storage import FirestoreStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

app = FastAPI(title="alza-ai-assistant")


@lru_cache(maxsize=1)
def gmail_credentials() -> Credentials:
    """Gmail OAuth credentials, created on first use and cached, so /health works before
    any Gmail call."""
    return get_credentials(settings)


@lru_cache(maxsize=1)
def shared_mailbox() -> Mailbox:
    """Clients that are safe to share between concurrent requests."""
    gmail = GmailClient.from_credentials(gmail_credentials())
    own_address = gmail.get_address()
    log.info("Mailbox %s ready", own_address)
    return Mailbox(
        gmail=gmail,
        own_address=own_address,
        storage=FirestoreStorage(settings.GCP_PROJECT_ID),
        gemini=make_client(settings),
        settings=settings,
    )


def get_mailbox() -> Mailbox:
    """Per request: a new Gmail client, because its HTTP connection is not
    thread-safe. Everything else is shared."""
    return replace(
        shared_mailbox(), gmail=GmailClient.from_credentials(gmail_credentials())
    )


def decode_notification(body: dict) -> tuple[str, str]:
    """Returns (emailAddress, historyId) from a Pub/Sub push body."""
    try:
        data = json.loads(base64.b64decode(body["message"]["data"], validate=True))
        return data["emailAddress"], str(int(data["historyId"]))
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub payload") from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; needs no authentication or configuration of external services."""
    return {"status": "ok"}


# Pub/Sub acks on 2xx and redelivers otherwise. An error escaping check_inbox
# becomes a 500, so the notification is retried; handled messages are skipped.
@app.post("/gmail/webhook", status_code=204, dependencies=[Depends(verify_oidc)])
async def gmail_webhook(
    request: Request, mailbox: Mailbox = Depends(get_mailbox)
) -> Response:
    """Pub/Sub push endpoint for Gmail notifications: answers every new INBOX email.
    Returns 204 when all are handled; any other status makes Pub/Sub redeliver."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Body is not JSON") from exc
    email_address, history_id = decode_notification(body)
    log.info(
        "Gmail notification: emailAddress=%s historyId=%s", email_address, history_id
    )

    if email_address.lower() != mailbox.own_address.lower():
        log.warning("Notification for another mailbox, ignoring")
        return Response(status_code=204)

    # The Google clients block, so the work runs in a worker thread.
    result = await run_in_threadpool(mailbox.check_inbox, history_id)
    log.info("Inbox processed: %s", result.results)
    return Response(status_code=204)


@app.post("/internal/gmail/renew-watch", dependencies=[Depends(verify_oidc)])
def renew_watch(mailbox: Mailbox = Depends(get_mailbox)) -> dict[str, str]:
    """Starts or renews the Gmail watch (it expires after 7 days); called daily by Cloud
    Scheduler."""
    response = mailbox.gmail.watch(settings.PUBSUB_TOPIC)
    if mailbox.storage.get_history_id() is None:
        mailbox.storage.save_history_id(response["historyId"])
    log.info(
        "Gmail watch renewed: historyId=%s expiration=%s",
        response["historyId"],
        response["expiration"],
    )
    return {"historyId": response["historyId"], "expiration": response["expiration"]}


@app.post("/internal/gmail/recovery-check-inbox", dependencies=[Depends(verify_oidc)])
async def recovery_check_inbox(mailbox: Mailbox = Depends(get_mailbox)) -> dict:
    """Recovery: checks all unread INBOX mail instead of the history since the
    stored history ID, e.g. after lost notifications."""
    result = await run_in_threadpool(mailbox.check_inbox, recovery=True)
    return {
        "startHistoryId": result.start_history_id,
        "savedHistoryId": result.saved_history_id,
        "results": result.results,
    }
