# alza_ai_assistant

AI email assistant for Gmail: reads incoming emails with PDF, image and audio attachments, answers with Gemini on Google Cloud using RAG over a knowledge base, and replies in the original thread. Serverless on Cloud Run, Pub/Sub and Firestore.

**Current state:** minimal skeleton for verifying the GCP setup (Cloud Build → Artifact Registry → Cloud Run). Gmail and Gemini logic come later.

```
app/main.py                       FastAPI: /health, /gmail/webhook (Pub/Sub push),
                                  /internal/gmail/renew-watch, /internal/gmail/recovery-sync
app/config.py                     settings from env vars (GCP_PROJECT_ID, REGION, GMAIL_*, GEMINI_*, ...)

app/mail_processing/gmail.py      Gmail API: access token, history, messages, send, mark read, watch
app/mail_processing/sync.py       Gmail history since the stored historyId (gmail_sync/state)
app/mail_processing/filters.py    own/SENT/SPAM/promotions/auto-reply/bulk mail, 5 replies per thread
app/mail_processing/processor.py  one email: Firestore claim, reply (fallback if Gemini fails), mark read
app/mail_processing/reply.py      reply in the original thread (threadId, In-Reply-To, References)
app/mail_processing/store.py      Firestore state (processed_emails, gmail_sync) + in-memory versions

app/ai_core/llm.py                Gemini on Vertex AI: system instruction, EMAIL block, timeout, one retry

app/utils/auth.py                 OIDC check (audience SERVICE_URL, caller INVOKER_SERVICE_ACCOUNT)

scripts/run_once.py               answers the oldest unread email once, from the terminal
tests/                            pytest tests (Gmail, Firestore and Gemini mocked)
cloudbuild.yaml                   black check → tests → docker build → push → deploy to Cloud Run
token-refresh/                    one-time script that gets the Gmail OAuth refresh token
```

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-cloudbuild.txt
cp .env.example .env            # fill in values
.venv/bin/uvicorn app.main:app --reload --port 8080
curl localhost:8080/health
```

With Docker:

```bash
docker build -t email-bot .
docker run --rm -p 8080:8080 --env-file .env email-bot
```

## End-to-end test

`scripts/run_once.py` answers the oldest unread INBOX email that is not yet in
Firestore with a Gemini reply in the same thread and marks it as read.
Set `GEMINI_MODEL` in `.env`; locally Firestore and Vertex AI use your
Application Default Credentials.
Fill the Gmail values in `.env` from Secret Manager (values are not printed):

```bash
for pair in GMAIL_CLIENT_ID:gmail-client-id GMAIL_CLIENT_SECRET:gmail-client-secret \
            GMAIL_OAUTH_REFRESH_TOKEN:gmail-oauth-refresh-token; do
  echo "${pair%%:*}=$(gcloud secrets versions access latest --secret=${pair#*:})" >> .env
done
.venv/bin/python -m scripts.run_once
```

## Tests and formatting

```bash
.venv/bin/black .               # format; CI runs black --check
.venv/bin/pytest
```

## Deploy

Every push to `main` triggers Cloud Build with `cloudbuild.yaml`:

1. `black --check` and `pytest` – an unformatted push or a failing test stops here and is not deployed,
2. build the Docker image and push it to
   `europe-west1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/alza-ai-assistant`,
3. deploy the `alza-ai-assistant` Cloud Run service in `europe-west1` as `alza-ai-assistant-runtime`,
   authentication required.

One-time setup:

* Cloud Build trigger on this repo, branch `^main$`, configuration *Cloud Build configuration file*, path `/cloudbuild.yaml`.
* The trigger's service account needs **Cloud Run Admin**, **Artifact Registry Writer** and
  **Service Account User** on `alza-ai-assistant-runtime`.
* The Artifact Registry repository `cloud-run-source-deploy` must exist in `europe-west1`.

Check the deployed service (authentication is required):

```bash
URL=$(gcloud run services describe alza-ai-assistant --region europe-west1 --format 'value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" $URL/health
```
