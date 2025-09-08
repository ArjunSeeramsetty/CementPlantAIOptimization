from typing import Optional

from .settings import get_settings


def get_bigquery_client(project_id: Optional[str] = None):
    """Return an authenticated BigQuery client using default credentials."""
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Add it to requirements.txt"
        ) from exc

    settings = get_settings()
    project = project_id or settings.gcp_project
    return bigquery.Client(project=project)


def get_storage_client():
    """Return an authenticated Cloud Storage client using default credentials."""
    try:
        from google.cloud import storage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-storage is not installed. Add it to requirements.txt"
        ) from exc

    return storage.Client()


def init_vertex_ai(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
):
    """Initialize Vertex AI SDK with provided or default settings."""
    try:
        from google.cloud import aiplatform  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-aiplatform is not installed. Add it to requirements.txt"
        ) from exc

    settings = get_settings()
    aiplatform.init(
        project=project_id or settings.gcp_project,
        location=location or settings.gcp_region,
        staging_bucket=staging_bucket or settings.vertex_bucket,
    )



