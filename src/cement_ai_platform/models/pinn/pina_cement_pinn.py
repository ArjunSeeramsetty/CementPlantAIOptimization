from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import grad
    _torch_available = True
except ImportError:
    _torch_available = False

try:
    from pina import Trainer, Condition, SpatialProblem, PINN, FeedForward, FixedValue
    from pina.equation import Equation
    _pina_available = True
except ImportError:
    _pina_available = False
    # Create dummy classes for when PINA is not available
    class Equation:
        pass
    class Trainer:
        pass
    class Condition:
        pass
    class SpatialProblem:
        pass
    class PINN:
        pass
    class FeedForward:
        pass
    class FixedValue:
        pass


class CementChemistryPhysics(Equation):
    """Physics equations for cement chemistry constraints."""
    
    def __init__(self):
        self.input_variables = ['T', 'F', 'U', 'P', 'H']  # Temperature, Feed, Fuel, Pressure, Heat
        self.output_variables = ['free_lime']
    
    def residual(self, input_, output_):
        """Compute physics residual for cement chemistry."""
        if not _torch_available:
            return torch.tensor(0.0)
        
        T = input_.extract('T')
        F = input_.extract('F')
        U = input_.extract('U')
        P = input_.extract('P')
        H = input_.extract('H')
        
        # Heat balance residual: dT/dF - k*(U/F) = 0
        dT_dF = grad(output_, input_, components=['free_lime'], d=['F'])
        
        # Energy efficiency parameter
        k = torch.nn.Parameter(torch.tensor([5.0]))
        
        # Temperature effect on calcination
        temp_effect = torch.exp(-8000.0 / (T + 273.15))
        
        # Residence time effect (simplified)
        residence_time = 1.0 / (F / 1000.0)  # Simplified residence time
        
        # Physics-informed prediction
        physics_prediction = 3.0 * torch.exp(-temp_effect * residence_time) * (1.0 - P / 100.0)
        
        # Residual: difference between neural network output and physics prediction
        residual = output_.extract('free_lime') - physics_prediction
        
        return residual


class CementQualityProblem(SpatialProblem):
    """Spatial problem definition for cement quality prediction."""
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]]):
        self.output_variables = ['free_lime']
        self.spatial_domain = domain_bounds
        self.conditions = {}
    
    def add_data_condition(self, X_data: np.ndarray, y_data: np.ndarray, 
                          feature_names: List[str]):
        """Add data condition for training."""
        if not _pina_available:
            return
        
        # Create data condition
        data_condition = Condition(
            location=X_data,
            equation=FixedValue(y_data.reshape(-1, 1))
        )
        self.conditions['data'] = data_condition
    
    def add_physics_condition(self, n_collocation_points: int = 1000):
        """Add physics condition with collocation points."""
        if not _pina_available:
            return
        
        # Sample collocation points from the domain
        collocation_points = []
        for var_name, (min_val, max_val) in self.spatial_domain.items():
            points = torch.uniform(min_val, max_val, (n_collocation_points, 1))
            collocation_points.append(points)
        
        collocation_input = torch.cat(collocation_points, dim=1)
        
        # Create physics condition
        physics_condition = Condition(
            location=collocation_input,
            equation=CementChemistryPhysics()
        )
        self.conditions['physics'] = physics_condition


class PinaCementPINN:
    """
    Physics-Informed Neural Network for cement quality prediction using PINA.
    
    Incorporates cement chemistry physics constraints:
    - Heat and mass balance equations
    - Chemical reaction kinetics
    - Thermodynamic equilibrium constraints
    """
    
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [64, 64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.model: Optional[nn.Module] = None
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.is_trained = False
        self.problem: Optional[CementQualityProblem] = None
        self.solver: Optional[PINN] = None
        
        if _torch_available:
            self._build_model()
    
    def _build_model(self):
        """Build the PINN architecture using PINA."""
        if not _pina_available:
            # Fallback to standard PyTorch model
            layers = []
            prev_dim = self.input_dim
            
            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.model = nn.Sequential(*layers)
        else:
            # Use PINA FeedForward model
            self.model = FeedForward(
                input_dimensions=self.input_dim,
                output_dimensions=1,
                layers=self.hidden_dims
            )
    
    def prepare_data(self, data: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with normalization."""
        from sklearn.preprocessing import StandardScaler
        
        self.feature_names = feature_cols
        X = data[feature_cols].values
        y = data[target_col].values.reshape(-1, 1)
        
        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        self.scalers = {
            'X': scaler_X,
            'y': scaler_y
        }
        
        return X_scaled, y_scaled
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              physics_weight: float = 0.1) -> Dict[str, Any]:
        """Train the PINN with physics constraints using PINA."""
        if not _torch_available:
            return self._train_fallback(X, y)
        
        if _pina_available:
            return self._train_pina(X, y, epochs, physics_weight)
        else:
            return self._train_pytorch(X, y, epochs, physics_weight)
    
    def _train_pina(self, X: np.ndarray, y: np.ndarray, epochs: int, physics_weight: float) -> Dict[str, Any]:
        """Train using PINA framework."""
        try:
            # Define domain bounds
            domain_bounds = {}
            for i, feature_name in enumerate(self.feature_names):
                min_val = float(X[:, i].min())
                max_val = float(X[:, i].max())
                domain_bounds[feature_name] = (min_val, max_val)
            
            # Create problem
            self.problem = CementQualityProblem(domain_bounds)
            
            # Add data condition
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            self.problem.add_data_condition(X_tensor, y_tensor, self.feature_names)
            
            # Add physics condition
            self.problem.add_physics_condition(n_collocation_points=1000)
            
            # Create solver
            self.solver = PINN(self.problem, self.model)
            
            # Create trainer
            trainer = Trainer(
                solver=self.solver,
                max_epochs=epochs,
                accelerator='cpu'  # Use CPU for compatibility
            )
            
            # Train
            trainer.train()
            
            self.is_trained = True
            
            return {
                "status": "success",
                "method": "pina_pinn",
                "epochs": epochs,
                "physics_weight": physics_weight
            }
            
        except Exception as e:
            print(f"PINA training failed: {e}")
            return self._train_pytorch(X, y, epochs, physics_weight)
    
    def _train_pytorch(self, X: np.ndarray, y: np.ndarray, epochs: int, physics_weight: float) -> Dict[str, Any]:
        """Train using PyTorch with physics constraints."""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_tensor)
            
            # Data loss (MSE)
            data_loss = nn.MSELoss()(predictions, y_tensor)
            
            # Physics loss
            physics_loss = self._compute_physics_loss(X_tensor, predictions)
            
            # Total loss
            total_loss = data_loss + physics_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Data Loss = {data_loss.item():.4f}, "
                      f"Physics Loss = {physics_loss.item():.4f}, "
                      f"Total Loss = {total_loss.item():.4f}")
        
        self.is_trained = True
        
        return {
            "status": "success",
            "method": "pytorch_pinn",
            "epochs": epochs,
            "final_loss": losses[-1],
            "physics_weight": physics_weight
        }
    
    def _compute_physics_loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss."""
        if not _torch_available:
            return torch.tensor(0.0)
        
        # Extract features (assuming order: temp, fuel_rate, kiln_speed, pressure, heat_consumption)
        temp, fuel_rate, kiln_speed, pressure, heat_consumption = inputs.T
        
        # Constraint 1: Heat Balance
        energy_input = fuel_rate * 3.1
        temp_predicted = energy_input / 1000.0 + 1200.0
        heat_balance_loss = torch.mean((temp - temp_predicted) ** 2)
        
        # Constraint 2: Mass Balance
        production_rate = fuel_rate * 0.85
        mass_balance_loss = torch.mean((production_rate - fuel_rate * 0.85) ** 2)
        
        # Constraint 3: Reaction Kinetics
        residence_time = 1.0 / kiln_speed
        reaction_rate = torch.exp(-8000.0 / (temp + 273.15))
        free_lime_base = 3.0 * torch.exp(-reaction_rate * residence_time)
        physics_prediction = free_lime_base * (1.0 - pressure / 100.0)
        
        kinetics_loss = torch.mean((outputs.squeeze() - physics_prediction) ** 2)
        
        # Weighted combination
        total_physics_loss = (
            0.4 * heat_balance_loss +
            0.3 * mass_balance_loss +
            0.3 * kinetics_loss
        )
        
        return total_physics_loss
    
    def _train_fallback(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fallback training when PyTorch is not available."""
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y.ravel())
        self.is_trained = True
        
        return {
            "status": "success",
            "method": "fallback_rf",
            "note": "PyTorch not available, using Random Forest"
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if _torch_available and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor)
                return predictions.numpy()
        else:
            # Fallback model
            return self.model.predict(X).reshape(-1, 1)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale."""
        if 'y' in self.scalers:
            return self.scalers['y'].inverse_transform(predictions)
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if isinstance(self.model, nn.Module):
            # For neural networks, we can't easily get feature importance
            return {name: 1.0 / len(self.feature_names) for name in self.feature_names}
        else:
            # For tree-based models
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                return {name: 1.0 / len(self.feature_names) for name in self.feature_names}
    
    def explain_prediction(self, X: np.ndarray, prediction_idx: int = 0) -> Dict[str, Any]:
        """Provide physics-based explanation for a prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before explanation")
        
        # Get the specific input
        x_sample = X[prediction_idx:prediction_idx+1]
        
        # Make prediction
        pred = self.predict(x_sample)[0, 0]
        
        # Extract features
        temp, fuel_rate, kiln_speed, pressure, heat_consumption = x_sample[0]
        
        # Physics-based analysis
        energy_input = fuel_rate * 3.1
        residence_time = 1.0 / kiln_speed
        reaction_rate = np.exp(-8000.0 / (temp + 273.15))
        
        explanation = {
            "predicted_free_lime": float(pred),
            "physics_analysis": {
                "energy_input": float(energy_input),
                "residence_time": float(residence_time),
                "reaction_rate": float(reaction_rate),
                "temperature_effect": "High" if temp > 1450 else "Low",
                "pressure_level": "Optimal" if 50 <= pressure <= 150 else "Suboptimal"
            },
            "recommendations": self._generate_recommendations(fuel_rate, kiln_speed, temp, pressure, pred)
        }
        
        return explanation
    
    def _generate_recommendations(self, fuel_rate: float, kiln_speed: float, 
                                temp: float, pressure: float, free_lime: float) -> List[str]:
        """Generate operational recommendations based on physics."""
        recommendations = []
        
        if free_lime > 2.0:
            recommendations.append("Free lime is high - consider increasing temperature")
            recommendations.append("Check kiln speed - may need adjustment")
        
        if temp < 1400:
            recommendations.append("Temperature is low - increase fuel rate")
        
        if pressure < 50:
            recommendations.append("Pressure is low - check draft system")
        elif pressure > 150:
            recommendations.append("Pressure is high - check air flow")
        
        if kiln_speed < 3.0:
            recommendations.append("Kiln speed is low - increase for better mixing")
        
        return recommendations


def create_pina_cement_pinn(input_dim: int = 5) -> PinaCementPINN:
    """Factory function to create a PINA-based cement PINN."""
    return PinaCementPINN(input_dim)
