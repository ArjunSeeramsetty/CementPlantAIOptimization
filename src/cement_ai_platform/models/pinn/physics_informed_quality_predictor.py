from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Import the proper PINA-based PINN implementation
from .pina_cement_pinn import PinaCementPINN


class PhysicsInformedQualityPredictor:
    """
    Physics-Informed Neural Network for cement quality prediction.
    
    Uses the PINA framework for proper PINN implementation.
    Incorporates cement chemistry physics constraints:
    - Heat and mass balance equations
    - Chemical reaction kinetics
    - Thermodynamic equilibrium constraints
    """
    
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [64, 64, 32]):
        # Use the proper PINA-based PINN implementation
        self.pina_pinn = PinaCementPINN(input_dim, hidden_dims)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
    
    def prepare_data(self, data: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with normalization."""
        return self.pina_pinn.prepare_data(data, feature_cols, target_col)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              physics_weight: float = 0.1) -> Dict[str, Any]:
        """Train the PINN with physics constraints using PINA."""
        return self.pina_pinn.train(X, y, epochs, physics_weight)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.pina_pinn.predict(X)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale."""
        return self.pina_pinn.inverse_transform_predictions(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        return self.pina_pinn.get_feature_importance()
    
    def explain_prediction(self, X: np.ndarray, prediction_idx: int = 0) -> Dict[str, Any]:
        """Provide physics-based explanation for a prediction."""
        return self.pina_pinn.explain_prediction(X, prediction_idx)


def create_pinn_quality_predictor(input_dim: int = 5) -> PhysicsInformedQualityPredictor:
    """Factory function to create a PINN quality predictor."""
    return PhysicsInformedQualityPredictor(input_dim)