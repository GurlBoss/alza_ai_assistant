from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GCP_PROJECT_ID: str = ""
    REGION: str = "europe-west1"
    PUBSUB_TOPIC: str = ""
    # Cloud Run URL; the OIDC token audience for Pub/Sub push and Scheduler.
    SERVICE_URL: str = ""
    # Only tokens issued to this account are accepted (empty = any verified).
    INVOKER_SERVICE_ACCOUNT: str = ""
    GEMINI_MODEL: str = ""
    # Newer Gemini models are served only from the global endpoint, not europe-west1.
    GEMINI_REGION: str = "global"

    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_OAUTH_REFRESH_TOKEN: str = ""


settings = Settings()
