"""Unit tests for the Upgraded SOTA Predictive Maintenance RUL module."""

import os
import pytest
import joblib
import pandas as pd
from scripts.train_pdm_rul import generate_pdm_mock_data, train_rul_model
from cement_ai_platform.maintenance.predictive_maintenance import PredictiveMaintenanceEngine


def test_pdm_mock_data_generation():
    """Test generating turbofan bearing wear data."""
    df = generate_pdm_mock_data(num_units=3)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "vibration" in df.columns
    assert "rul" in df.columns
    assert df["unit_id"].nunique() == 3


def test_train_pdm_model():
    """Test training and evaluating the RandomForest RUL model."""
    df = generate_pdm_mock_data(num_units=5)
    model, metrics = train_rul_model(df)
    
    assert model is not None
    assert "mean_absolute_error" in metrics
    assert "r2_score" in metrics
    assert isinstance(metrics["mean_absolute_error"], float)


def test_pdm_engine_model_integration():
    """Test that the maintenance engine loads and runs predictions with the SOTA RUL model."""
    # Ensure model file exists (we generated it earlier)
    model_path = os.path.join(os.getcwd(), "models", "pdm_rul_model.joblib")
    assert os.path.exists(model_path), f"SOTA model file missing at {model_path}"
    
    engine = PredictiveMaintenanceEngine()
    assert engine.sota_rul_model is not None, "Engine failed to load SOTA RUL model"
    
    # Mock sensor data for a kiln bearing
    equipment_data = {
        "equipment_id": "KILN_BEARING_001",
        "equipment_type": "kiln",
        "equipment_name": "Kiln Main Bearing",
        "operating_hours": 12000.0,
        "maintenance_age": 1200.0,
        "load_factor": 0.88,
        "kiln_vibration": 3.8,
        "kiln_temperature": 82.5,
        "kiln_current": 105.0,
        "kiln_torque": 45.0,
        "cycle": 150.0,
        "rpm": 1460.0
    }
    
    recommendation = engine.predict_equipment_failure(equipment_data)
    
    assert recommendation is not None
    assert recommendation.equipment_id == "KILN_BEARING_001"
    assert recommendation.time_to_failure_hours is not None
    assert recommendation.time_to_failure_hours > 0
    assert recommendation.confidence >= 0.5
