from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from environment variables (Cloud Run) or .env (locally).

    Fields without a default are required: if one is missing or empty, the app
    fails at startup with an error naming it. Everything else has a working default.
    """

    # env_ignore_empty: "VAR=" counts as not set, so an empty required value fails
    # and an empty optional one falls back to its default.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    # --- required -----------------------------------------------------------------
    GCP_PROJECT_ID: str
    # Gmail OAuth of the bot's mailbox (Secret Manager in Cloud Run).
    GMAIL_CLIENT_ID: str
    GMAIL_CLIENT_SECRET: str
    GMAIL_OAUTH_REFRESH_TOKEN: str
    # Cloud Run URL that Pub/Sub push and Scheduler call: the accepted OIDC
    # token audience.
    SERVICE_URL: str
    # Topic Gmail publishes new-mail notifications to (used by renew-watch).
    PUBSUB_TOPIC: str

    # gemini model
    GEMINI_MODEL: str
    # Location of the model.
    GEMINI_REGION: str
    # The only account whose OIDC tokens are accepted (Pub/Sub push, Scheduler).
    INVOKER_SERVICE_ACCOUNT: str
    # Knowledge base: full RAG Engine corpus name; every Gemini call retrieves from it.
    RAG_CORPUS: str

    # --- optional, with defaults --------------------------------------------------
    # Attachment limits in MB: per file and per email. The total stays below 18 MB.
    MAX_ATTACHMENT_MB: float = 10
    MAX_TOTAL_ATTACHMENTS_MB: float = 18
    RAG_TOP_K: int = 5
    RAG_DISTANCE_THRESHOLD: float = 0.5
    # Also log the text of retrieved chunks (the knowledge base has no personal data).
    RAG_DEBUG: bool = False


settings = Settings()
