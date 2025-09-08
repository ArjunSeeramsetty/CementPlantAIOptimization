from typing import Optional

import pandas as pd

from ...config.google_cloud_config import get_bigquery_client
from ...config.settings import get_settings


def run_bigquery(query: str, project_id: Optional[str] = None) -> pd.DataFrame:
    """Execute a BigQuery SQL query and return a DataFrame."""
    client = get_bigquery_client(project_id)
    job = client.query(query)
    return job.result().to_dataframe(create_bqstorage_client=True)


def table_to_dataframe(table: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read a BigQuery table into a DataFrame.

    Args:
        table: Fully-qualified table id (project.dataset.table) or dataset.table (uses default project)
        limit: Optional row limit
    """
    settings = get_settings()
    if table.count(".") == 1 and settings.gcp_project:
        table = f"{settings.gcp_project}.{table}"

    query = f"SELECT * FROM `{table}`"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return run_bigquery(query)



