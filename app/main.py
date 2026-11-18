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
from app.mail_processing.processor import Mailbox
from app.mail_processing.store import FirestoreStore, FirestoreSyncState
from app.mail_processing.sync import SyncResult, sync_mailbox
from app.utils.auth import verify_oidc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

app = FastAPI(title="alza-ai-assistant")


# Built on first use, so /health works even before the secrets are set.
@lru_cache(maxsize=1)
def gmail_credentials() -> Credentials:
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
        store=FirestoreStore(settings.GCP_PROJECT_ID),
        sync_state=FirestoreSyncState(settings.GCP_PROJECT_ID),
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
    return {"status": "ok"}


# Pub/Sub acks on 2xx and redelivers otherwise. An error escaping sync_mailbox
# becomes a 500, so the notification is retried; handled messages are skipped.
@app.post("/gmail/webhook", status_code=204, dependencies=[Depends(verify_oidc)])
async def gmail_webhook(
    request: Request, mailbox: Mailbox = Depends(get_mailbox)
) -> Response:
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

    result = await run_sync(mailbox, history_id)
    log.info("Sync done: %s", result.results)
    return Response(status_code=204)


@app.post("/internal/gmail/renew-watch", dependencies=[Depends(verify_oidc)])
def renew_watch(mailbox: Mailbox = Depends(get_mailbox)) -> dict[str, str]:
    if not settings.PUBSUB_TOPIC:
        raise HTTPException(status_code=503, detail="PUBSUB_TOPIC is not set")
    response = mailbox.gmail.watch(settings.PUBSUB_TOPIC)
    if mailbox.sync_state.get_history_id() is None:
        mailbox.sync_state.save_history_id(response["historyId"])
    log.info(
        "Gmail watch renewed: historyId=%s expiration=%s",
        response["historyId"],
        response["expiration"],
    )
    return {"historyId": response["historyId"], "expiration": response["expiration"]}


@app.post("/internal/gmail/recovery-sync", dependencies=[Depends(verify_oidc)])
async def recovery_sync(mailbox: Mailbox = Depends(get_mailbox)) -> dict:
    result = await run_sync(mailbox, None)
    return {
        "startHistoryId": result.start_history_id,
        "savedHistoryId": result.saved_history_id,
        "results": result.results,
    }


async def run_sync(mailbox: Mailbox, history_id: str | None) -> SyncResult:
    # Gmail, Firestore and Gemini clients are blocking; keep the event loop free.
    return await run_in_threadpool(sync_mailbox, mailbox, history_id)
