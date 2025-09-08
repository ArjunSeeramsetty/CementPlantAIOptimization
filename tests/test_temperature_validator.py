import pandas as pd

from cement_ai_platform.data.processors.temperature_validator import (
    validate_temperature_profile_consistency,
)


def test_temperature_validator_autodetect_columns():
    df = pd.DataFrame({
        "process_temp": [80, 82, 81, 85, 83],
        "curing_temp": [22, 23, 21, 24, 23],
    })
    result = validate_temperature_profile_consistency(df)
    assert result["total_samples"] == len(df)
    assert len(result["temperature_columns_checked"]) >= 2
    assert "temperature_statistics" in result



