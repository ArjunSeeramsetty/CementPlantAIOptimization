from functools import lru_cache
from typing import Optional

try:
    from pydantic import BaseSettings, Field
except Exception:  # pragma: no cover - allow import before deps installed
    # Lightweight fallback so imports don't break before dependencies are installed
    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, **_kwargs):  # type: ignore
        return default


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables.

    Environment variables (examples):
      - CEMENT_PROJECT_NAME=cement-ai-platform
      - CEMENT_ENV=dev
      - CEMENT_GCP_PROJECT=my-gcp-project
      - CEMENT_GCP_REGION=us-central1
      - CEMENT_BQ_DATASET=cement_analytics
      - CEMENT_VERTEX_BUCKET=gs://my-staging-bucket
      - CEMENT_FIREBASE_PROJECT_ID=my-firebase-project
      - CEMENT_LOG_LEVEL=INFO
    """

    project_name: str = Field("cement-ai-platform", env="CEMENT_PROJECT_NAME")
    env: str = Field("dev", env="CEMENT_ENV")

    # Google Cloud
    gcp_project: Optional[str] = Field(default=None, env="CEMENT_GCP_PROJECT")
    gcp_region: str = Field("us-central1", env="CEMENT_GCP_REGION")
    bq_dataset: Optional[str] = Field(default=None, env="CEMENT_BQ_DATASET")
    vertex_bucket: Optional[str] = Field(default=None, env="CEMENT_VERTEX_BUCKET")
    firebase_project_id: Optional[str] = Field(default=None, env="CEMENT_FIREBASE_PROJECT_ID")

    # Logging
    log_level: str = Field("INFO", env="CEMENT_LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached AppSettings instance."""
    try:
        # Optional .env support for local development
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # dotenv is optional; ignore if not installed
        pass

    return AppSettings()  # type: ignore[call-arg]



