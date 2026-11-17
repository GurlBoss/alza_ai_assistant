from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GCP_PROJECT_ID: str = ""
    REGION: str = "europe-west1"
    PUBSUB_TOPIC: str = ""


settings = Settings()
