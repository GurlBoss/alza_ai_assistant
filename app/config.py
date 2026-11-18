from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GCP_PROJECT_ID: str = ""
    REGION: str = "europe-west1"
    PUBSUB_TOPIC: str = ""
    # Cloud Run URL(s), comma-separated; the accepted OIDC token audiences for
    # Pub/Sub push and Scheduler.
    SERVICE_URL: str = ""
    # Only tokens issued to this account are accepted (empty = any verified).
    INVOKER_SERVICE_ACCOUNT: str = ""
    GEMINI_MODEL: str = ""
    # Newer Gemini models are served only from the global endpoint, not europe-west1.
    GEMINI_REGION: str = "global"

    # Attachment limits: per file and per email, in megabytes. The total stays
    # below Gemini's 20 MB inline request limit, leaving room for the prompt.
    MAX_ATTACHMENT_MB: float = 10
    MAX_TOTAL_ATTACHMENTS_MB: float = 18

    # Knowledge base: Vertex AI RAG Engine corpus, used as a retrieval tool by
    # Gemini. Full resource name; empty = replies go out without the knowledge base.
    RAG_CORPUS: str = ""
    RAG_TOP_K: int = 5
    RAG_DISTANCE_THRESHOLD: float = 0.5
    # Also log the text of retrieved chunks (the knowledge base has no personal data).
    RAG_DEBUG: bool = False

    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_OAUTH_REFRESH_TOKEN: str = ""


settings = Settings()
