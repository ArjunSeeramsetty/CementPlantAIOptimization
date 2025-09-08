from typing import Optional

from ..config.google_cloud_config import init_vertex_ai


def submit_custom_training_job(
    display_name: str,
    script_path: str,
    container_uri: str,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
):
    """Submit a custom training job to Vertex AI.

    Note: This is a minimal scaffold; extend with datasets, machine types, etc.
    """
    try:
        from google.cloud import aiplatform  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-aiplatform not installed") from exc

    init_vertex_ai(project_id=project_id, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=script_path,
        python_module_name="trainer.task",
        container_uri=container_uri,
    )
    return job



