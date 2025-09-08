from typing import Dict, Optional

from ..config.google_cloud_config import init_vertex_ai


def submit_custom_training_job(
    display_name: str,
    package_gcs_uri: str,
    module_name: str,
    container_uri: str,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    worker_pool_specs: Optional[list] = None,
    args: Optional[list] = None,
):
    """Create and run a custom training job on Vertex AI.

    Returns the Job object after submission. Caller may inspect `job.resource_name`.
    """
    try:
        from google.cloud import aiplatform  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-aiplatform not installed") from exc

    init_vertex_ai(project_id=project_id, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=package_gcs_uri,
        python_module_name=module_name,
        container_uri=container_uri,
    )

    # Defaults suitable for a quick run if not provided
    run_kwargs: Dict[str, object] = {
        "replica_count": 1,
        "args": args or [],
    }
    if worker_pool_specs is not None:
        run_kwargs["worker_pool_specs"] = worker_pool_specs

    job = job.run(sync=False, **run_kwargs)  # non-blocking submit
    return job



