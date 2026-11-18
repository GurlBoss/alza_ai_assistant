"""Dummy values for the required settings, so the tests run without a real
configuration. Set before any app module creates its Settings."""

import os

TEST_ENV = {
    "GCP_PROJECT_ID": "test-project",
    "GMAIL_CLIENT_ID": "test-client-id",
    "GMAIL_CLIENT_SECRET": "test-client-secret",
    "GMAIL_OAUTH_REFRESH_TOKEN": "test-refresh-token",
    "SERVICE_URL": "https://test-service.run.app",
    "PUBSUB_TOPIC": "projects/test-project/topics/test-topic",
    "GEMINI_MODEL": "gemini-test",
    "GEMINI_REGION": "global",
    "INVOKER_SERVICE_ACCOUNT": "invoker@test-project.iam.gserviceaccount.com",
    "RAG_CORPUS": "projects/test-project/locations/europe-west4/ragCorpora/test",
}
# Environment variables win over a local .env, so real values never leak in.
os.environ.update(TEST_ENV)
