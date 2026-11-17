"""OIDC check for requests from Pub/Sub push and Cloud Scheduler."""

import logging

from fastapi import HTTPException, Request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.config import settings

log = logging.getLogger(__name__)

_transport = google_requests.Request()


def verify_oidc(request: Request) -> dict:
    """FastAPI dependency: the request must carry a Google-signed ID token
    for this service, issued to INVOKER_SERVICE_ACCOUNT when that is set.

    Accepted audiences: SERVICE_URL, or SERVICE_URL + the called path. The latter
    is what Pub/Sub and Cloud Scheduler send when no audience is configured.
    """
    if not settings.SERVICE_URL:
        log.error("SERVICE_URL is not set, rejecting the request")
        raise HTTPException(status_code=503, detail="Service not configured")

    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    service_url = settings.SERVICE_URL.rstrip("/")
    audiences = [service_url, service_url + request.url.path]
    try:
        claims = id_token.verify_oauth2_token(
            authorization[len("bearer ") :], _transport, audience=audiences
        )
    except ValueError as exc:
        log.warning("Invalid OIDC token: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    if not claims.get("email_verified"):
        raise HTTPException(status_code=403, detail="Unverified token email")
    expected = settings.INVOKER_SERVICE_ACCOUNT
    if expected and claims.get("email", "").lower() != expected.lower():
        log.warning("OIDC token from unexpected account %s", claims.get("email"))
        raise HTTPException(status_code=403, detail="Unexpected caller")
    return claims
