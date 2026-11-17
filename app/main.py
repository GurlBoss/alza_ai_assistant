import base64
import json
import logging

from fastapi import FastAPI, HTTPException, Request, Response

from app.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="email-bot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/gmail/webhook", status_code=204)
async def gmail_webhook(request: Request) -> Response:
    # TODO: verify the Pub/Sub OIDC token before trusting the request.
    body = await request.json()
    try:
        data = json.loads(base64.b64decode(body["message"]["data"]))
        email_address = data["emailAddress"]
        history_id = data["historyId"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub payload") from exc

    log.info(
        "Gmail notification: emailAddress=%s historyId=%s project=%s",
        email_address,
        history_id,
        settings.GCP_PROJECT_ID,
    )
    # TODO: fetch new messages since historyId and process them.
    return Response(status_code=204)
