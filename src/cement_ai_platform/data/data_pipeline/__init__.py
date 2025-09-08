"""Data pipeline connectors (e.g., BigQuery, synthetic generation, streaming)."""

from .bigquery_connector import run_bigquery, table_to_dataframe  # noqa: F401
from .synthetic_data_generator import (
    generate_synthetic_dataset,
    generate_synthetic_temperature_data,
)  # noqa: F401
from .streaming_ingestion import BigQueryStreamingIngestion  # noqa: F401
