# token-refresh/get_token.py - standalone, not part of the app.
#
# Issues a new Gmail refresh token and stores the OAuth credentials in Secret Manager
# as three secrets (see SECRETS). Re-run whenever the token expires (every 7 days
# while the OAuth consent screen is in Testing mode).
#
# How to run (from the token-refresh/ folder):
#   1. One time: install the dependencies.
#        python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   2. One time: put the OAuth client JSON (downloaded from the OAuth client in the
#      console) at secrets/client_secret.json, and set GOOGLE_CLOUD_PROJECT in .env
#      (see .env.example).
#   3. Log in to Google Cloud, so the script can write the secrets:
#        gcloud auth application-default login
#   4. Run the script; port 8080 must be free for the sign-in redirect:
#        .venv/bin/python get_token.py
#      In the browser, sign in as the bot's Gmail account and allow access ("Google
#      hasn't verified this app" -> Continue).
#   5. Run the gcloud run services update command the script prints at the end, so
#      Cloud Run starts a new revision with the new token.
#   6. For local runs (scripts/run_once.py), copy the new values into the root .env
#      too (see the README, "Test locally").
#
# "No refresh token returned": remove the app's access in the bot's Google account
# (Security -> Third-party connections) and run it again.
import json
import os
import time
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from google.api_core.exceptions import AlreadyExists, FailedPrecondition, NotFound
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import secretmanager
from google_auth_oauthlib.flow import InstalledAppFlow

HERE = Path(__file__).resolve().parent
CLIENT_FILE = HERE / "secrets" / "client_secret.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
PORT = 8080  # must match the redirect URI http://localhost:8080/

# Cloud Run env var -> Secret Manager secret name.
SECRETS = {
    "GMAIL_CLIENT_ID": "gmail-client-id",
    "GMAIL_CLIENT_SECRET": "gmail-client-secret",
    "GMAIL_OAUTH_REFRESH_TOKEN": "gmail-oauth-refresh-token",
}


def save_secret(client, project: str, secret_id: str, value: str) -> str:
    """Makes value the latest version of the secret, creating the secret if needed.

    Adds a version only when the value changed, so the client ID and secret don't
    pile up identical versions on every token refresh.
    """
    name = f"projects/{project}/secrets/{secret_id}"
    try:
        client.create_secret(
            request={
                "parent": f"projects/{project}",
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )
    except AlreadyExists:
        try:
            latest = client.access_secret_version(
                request={"name": f"{name}/versions/latest"}
            )
            if latest.payload.data.decode() == value:
                return f"{secret_id}: unchanged"
        except (NotFound, FailedPrecondition):  # no enabled version yet
            pass
    version = client.add_secret_version(
        request={"parent": name, "payload": {"data": value.encode()}}
    )
    return f"{secret_id}: new version {version.name.rsplit('/', 1)[-1]}"


def main() -> None:
    load_dotenv(HERE / ".env")
    # Check the Google Cloud login first, so a missing one doesn't waste the Gmail sign-in.
    try:
        _, adc_project = google.auth.default()
    except DefaultCredentialsError:
        raise SystemExit(
            "No Google Cloud credentials for Secret Manager. "
            "Run: gcloud auth application-default login"
        )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
    if not project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT to the project holding the secrets.")

    if not CLIENT_FILE.exists():
        raise SystemExit(f"Missing {CLIENT_FILE}. Download it from the OAuth client.")

    data = json.loads(CLIENT_FILE.read_text())
    if not (data.get("web") or data.get("installed")):
        raise SystemExit("Unexpected JSON format: no 'web' or 'installed' section.")

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
    creds = flow.run_local_server(
        host="localhost",
        port=PORT,
        access_type="offline",
        prompt="consent",  # forces Google to issue a refresh token
        open_browser=True,
    )

    if not creds.refresh_token:
        raise SystemExit(
            "No refresh token returned. Remove the app's access "
            "in the bot's Google account and run again."
        )

    values = {
        "GMAIL_CLIENT_ID": creds.client_id,
        "GMAIL_CLIENT_SECRET": creds.client_secret,
        "GMAIL_OAUTH_REFRESH_TOKEN": creds.refresh_token,
    }
    client = secretmanager.SecretManagerServiceClient()
    print("\nSign in was successful. Secret Manager:")
    for env_var, secret_id in SECRETS.items():
        print("  " + save_secret(client, project, secret_id, values[env_var]))

    # Cloud Run resolves ":latest" only when an instance starts; changing an env var
    # forces a new revision so the service picks up the new token.
    # --update-secrets keeps the service's other env vars and secrets.
    mapping = ",".join(f"{env}={sid}:latest" for env, sid in SECRETS.items())
    print("\nRoll out a new Cloud Run revision so the service picks it up:")
    print(
        f"  gcloud run services update alza-ai-assistant --update-secrets {mapping} "
        f"--update-env-vars GMAIL_TOKEN_ROTATED_AT={int(time.time())}"
    )


if __name__ == "__main__":
    main()
