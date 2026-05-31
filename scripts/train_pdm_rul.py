"""Predictive Maintenance Remaining Useful Life (RUL) Training Pipeline.

Trains a RandomForest model on simulated sensor profiles and attempts to
register it to the Google Cloud Vertex AI Model Registry.
"""

import os
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Try importing Vertex AI SDK
try:
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("Warning: google-cloud-aiplatform not installed. Local-only mode enabled.")


def generate_pdm_mock_data(num_units: int = 20) -> pd.DataFrame:
    """
    Generate synthetic bearing degradation data mimicking the NASA C-MAPSS profile.
    
    Each unit runs for a random number of cycles until failure.
    As cycles increase, vibration and temperature drift upwards.
    """
    np.random.seed(42)
    data_list = []
    
    for unit_id in range(1, num_units + 1):
        # Max lifetime cycles between 120 and 250
        max_cycles = np.random.randint(120, 250)
        
        # Initial healthy baselines
        vib_base = np.random.uniform(1.2, 1.8)
        temp_base = np.random.uniform(45.0, 52.0)
        pressure_base = np.random.uniform(320.0, 360.0)
        
        for cycle in range(1, max_cycles + 1):
            # Calculate degradation factor (exponential near failure)
            deg_factor = (cycle / float(max_cycles)) ** 3
            
            # Add noise and degradation to features
            vibration = vib_base + (deg_factor * np.random.uniform(1.5, 3.5)) + np.random.normal(0, 0.1)
            temperature = temp_base + (deg_factor * np.random.uniform(25.0, 45.0)) + np.random.normal(0, 0.5)
            pressure = pressure_base - (deg_factor * np.random.uniform(40.0, 80.0)) + np.random.normal(0, 3.0)
            
            # Target variable: Remaining Useful Life (RUL) in cycles
            rul = max_cycles - cycle
            
            data_list.append({
                "unit_id": unit_id,
                "cycle": cycle,
                "vibration": round(vibration, 3),
                "temperature": round(temperature, 2),
                "oil_pressure": round(pressure, 2),
                "rpm": round(1500.0 + np.random.normal(0, 10.0) - (deg_factor * 120), 1),
                "rul": rul
            })
            
    return pd.DataFrame(data_list)


def train_rul_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train the RUL model and evaluate performance."""
    # Define features and target
    features = ["cycle", "vibration", "temperature", "oil_pressure", "rpm"]
    X = df[features]
    y = df["rul"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training RandomForestRegressor on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mean_absolute_error": float(mae),
        "r2_score": float(r2)
    }
    
    print(f"Model evaluation results: MAE = {mae:.2f} cycles, R2 = {r2:.3f}")
    return model, metrics


def register_model_in_vertex(
    model_path: str,
    project_id: str,
    region: str,
    model_name: str = "cement-pdm-rul-model"
) -> None:
    """Register the model in Google Cloud Vertex AI Model Registry."""
    if not VERTEX_AI_AVAILABLE:
        print("[Error] Cannot register: Vertex AI SDK is not available.")
        return
        
    # Check for GCP credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
        print("[Warning] GCP credentials not detected. Skipping Vertex AI registration.")
        return
        
    try:
        print(f"[Info] Initializing Vertex AI client for project: {project_id} in {region}...")
        aiplatform.init(project=project_id, location=region)
        
        # Verify if model registry is accessible and upload
        print(f"[Info] Uploading {model_name} to Vertex AI Model Registry...")
        
        # Note: In production this uploads the containerized model metadata.
        # We simulate the registration success for local pipelines.
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=os.path.dirname(os.path.abspath(model_path)),
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
            sync=True
        )
        print(f"[Success] Registered in Vertex AI: Name = {model.display_name}, Resource = {model.resource_name}")
        
    except Exception as e:
        print(f"[Warning] Vertex AI model registry upload failed (falling back to local): {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register PdM RUL model.")
    parser.add_argument("--project-id", type=str, default="cement-ai-optimization", help="GCP Project ID")
    parser.add_argument("--region", type=str, default="us-central1", help="GCP Region")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save local models")
    args = parser.parse_args()
    
    # Ensure models dir exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 1. Generate Data
    print("Generating simulated turbofan degradation data...")
    df = generate_pdm_mock_data(num_units=30)
    
    # 2. Train Model
    model, metrics = train_rul_model(df)
    
    # 3. Save Model Locally
    model_path = os.path.join(args.model_dir, "pdm_rul_model.joblib")
    print(f"Saving model locally to {model_path}...")
    joblib.dump(model, model_path)
    print("[Success] Local model saved successfully.")
    
    # 4. Register in GCP Vertex AI
    register_model_in_vertex(
        model_path=model_path,
        project_id=args.project_id,
        region=args.region
    )


if __name__ == "__main__":
    main()
