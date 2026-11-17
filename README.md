# alza_ai_assistant

AI email assistant for Gmail: reads incoming emails with PDF, image and audio attachments, answers with Gemini on Google Cloud using RAG over a knowledge base, and replies in the original thread. Serverless on Cloud Run, Pub/Sub and Firestore.

**Current state:** minimal skeleton for verifying the GCP setup (Cloud Build → Artifact Registry → Cloud Run). Gmail and Gemini logic come later.

```
app/main.py      FastAPI app: GET /health, POST /gmail/webhook (Pub/Sub push)
app/config.py    settings from env vars (GCP_PROJECT_ID, REGION, PUBSUB_TOPIC)
tests/           pytest tests
cloudbuild.yaml  black check → docker build → push → deploy to Cloud Run
token-refresh/   one-time script that gets the Gmail OAuth refresh token
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

## Tests and formatting

```bash
.venv/bin/black .               # format; CI runs black --check
.venv/bin/pytest
```

## Deploy

Every push to `main` triggers Cloud Build with `cloudbuild.yaml`:

1. `black --check` – an unformatted push stops here and is not deployed,
2. build the Docker image and push it to
   `europe-west1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/email-bot`,
3. deploy the `email-bot` Cloud Run service in `europe-west1` as `email-bot-runtime`,
   authentication required.

One-time setup:

* Cloud Build trigger on this repo, branch `^main$`, configuration *Cloud Build configuration file*, path `/cloudbuild.yaml`.
* The trigger's service account needs **Cloud Run Admin**, **Artifact Registry Writer** and
  **Service Account User** on `email-bot-runtime`.
* The Artifact Registry repository `cloud-run-source-deploy` must exist in `europe-west1`.

Check the deployed service (authentication is required):

```bash
URL=$(gcloud run services describe email-bot --region europe-west1 --format 'value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" $URL/health
```
