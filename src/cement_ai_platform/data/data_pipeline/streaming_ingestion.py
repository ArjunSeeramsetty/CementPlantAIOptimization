"""BigQuery streaming ingestion for real-time sensor data."""

from __future__ import annotations

from typing import Iterable, Mapping

from ...config.google_cloud_config import get_bigquery_client
from ...config.settings import get_settings


class BigQueryStreamingIngestion:
    """Stream JSON rows into a configured BigQuery table.

    Expects env var `CEMENT_BQ_DATASET` to be set. Table is provided at call time.
    """

    def __init__(self, project_id: str | None = None):
        self.settings = get_settings()
        self.client = get_bigquery_client(project_id)

    def stream_rows(self, table_name: str, rows: Iterable[Mapping]):
        """Insert JSON rows into dataset.table using streaming API.

        Returns number of successfully ingested rows.
        """
        if not self.settings.bq_dataset:
            raise RuntimeError("CEMENT_BQ_DATASET is not configured")

        rows_list = list(rows)
        full_table_id = f"{self.settings.bq_dataset}.{table_name}"
        errors = self.client.insert_rows_json(full_table_id, rows_list)
        # insert_rows_json returns a list of errors per row; empty list means success
        return 0 if errors else len(rows_list)


