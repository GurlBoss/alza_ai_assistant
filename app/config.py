from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GCP_PROJECT_ID: str = ""
    REGION: str = "europe-west1"
    PUBSUB_TOPIC: str = ""
    # Cloud Run URL(s), comma-separated; the accepted OIDC token audiences for
    # Pub/Sub push, Scheduler and Eventarc.
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

    # Knowledge base (RAG). Without KB_BUCKET and the Vector Search IDs, replies
    # go out without knowledge.
    KB_BUCKET: str = ""
    VS_INDEX_ID: str = ""
    VS_ENDPOINT_ID: str = ""
    VS_DEPLOYED_INDEX_ID: str = ""
    # Served from REGION (europe-west1); not available on the global endpoint.
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    # Minimum cosine similarity of a chunk to be used.
    MIN_SCORE: float = 0.6

    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_OAUTH_REFRESH_TOKEN: str = ""


settings = Settings()
