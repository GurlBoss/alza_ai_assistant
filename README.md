# alza_ai_assistant

AI email assistant for Gmail: reads incoming emails with PDF, image and audio attachments, answers with Gemini on Google Cloud using RAG over a knowledge base, and replies in the original thread. Serverless on Cloud Run, Pub/Sub and Firestore.

**Current state:** minimal skeleton for verifying the GCP setup (Cloud Build → Artifact Registry → Cloud Run). Gmail and Gemini logic come later.

```
app/main.py                       FastAPI: /health, /gmail/webhook (Pub/Sub push),
                                  /internal/gmail/renew-watch, /internal/gmail/recovery-check-inbox
app/config.py                     settings from env vars; required vs optional: see .env.example

app/mail_processing/gmail.py      Gmail API: access token, history, messages, send, mark read, watch
app/mail_processing/mailbox.py    Mailbox: check_inbox -> handle_message -> process_email
app/mail_processing/filters.py    which emails not to answer (own/SPAM/auto-reply/bulk, 5 replies per
                                  thread) and attachment rules (type by content, size, signature images)
app/mail_processing/reply.py      reply in the original thread (threadId, In-Reply-To, References)
app/mail_processing/attachments.py  downloads attachments, accepted/rejected by the rules in filters.py

app/ai_core/llm.py                Gemini on Vertex AI: builds the prompt, attachments as inline parts, retry
app/ai_core/rag.py                RAG Engine retrieval tool for the Gemini call, grounding logs
app/ai_core/instructions.py       all texts for Gemini: system instruction, task, block markers

app/utils/firestore_storage.py    all Firestore access: email records + history ID
app/utils/auth.py                 OIDC check (audience SERVICE_URL, caller INVOKER_SERVICE_ACCOUNT)

scripts/run_once.py               answers the oldest unread email once, from the terminal
scripts/run_once_recovery.py      the same, but also retries emails that FAILED without a reply
scripts/get_rag_info.py           RAG Engine corpus: settings, files and chunks per file (--chunks)
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
Fill the required values in `.env` (see `.env.example`); locally Firestore and Vertex AI use your
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
