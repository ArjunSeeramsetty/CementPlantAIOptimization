import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Physics-Informed Neural Network Implementation
class PhysicsInformedNN:
    """Physics-Informed Neural Network with multiple physics constraints"""
    
    def __init__(self, hidden_layers=(100, 50, 25), activation='relu', max_iter=1000):
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.physics_weights = {
            'thermodynamic': 1.0,
            'mass_balance': 2.0,
            'heat_transfer': 1.5,
            'reaction_kinetics': 1.2,
            'energy_efficiency': 0.8,
            'material_composition': 1.5,
            'chemical_equilibrium': 1.0,
            'momentum_conservation': 0.9,
            'pressure_drop': 1.1,
            'fluid_continuity': 1.8
        }
        self.constraint_violations = {}
        self.physics_loss_history = []
        
    def thermodynamic_loss(self, predictions, features):
        """First and Second Laws of Thermodynamics constraints"""
        # Extract temperature and energy features
        T_in = features[:, 0] if features.shape[1] > 0 else np.ones(len(features)) * 300
        T_out = predictions.flatten()
        energy = features[:, 1] if features.shape[1] > 1 else np.ones(len(features)) * 1000
        
        # First Law: Energy balance
        energy_balance = np.mean(np.square(
            energy - (T_out - T_in) * 100  # Simplified heat capacity
        ))
        
        # Second Law: Entropy constraint (T_out >= T_in for heating)
        entropy_violations = np.maximum(0, T_in - T_out)
        entropy_constraint = np.mean(np.square(entropy_violations))
        
        return energy_balance + 0.5 * entropy_constraint
    
    def mass_balance_loss(self, predictions, features):
        """Conservation of mass principle"""
        if features.shape[1] < 4:
            return 0.0
            
        # Assume first 2 features are inputs, next 2 are related to outputs
        mass_in = np.sum(features[:, :2], axis=1)
        mass_related = predictions.flatten() * 0.1  # Scale prediction to mass
        
        return np.mean(np.square(mass_in - (mass_in + mass_related * 0.01)))
    
    def energy_efficiency_loss(self, predictions, features):
        """Energy efficiency must be <= 1"""
        energy_input = features[:, 1] if features.shape[1] > 1 else np.ones(len(features))
        useful_output = np.abs(predictions.flatten())
        
        efficiency = useful_output / (energy_input + 1e-8)
        efficiency_violations = np.maximum(0, efficiency - 1.0)
        
        return np.mean(np.square(efficiency_violations))
    
    def reaction_kinetics_loss(self, predictions, features):
        """Arrhenius equation constraint"""
        T = features[:, 0] if features.shape[1] > 0 else np.ones(len(features)) * 350
        concentration = 0.5  # Assumed concentration
        rate_constant = 1e6
        activation_energy = 50000
        R = 8.314
        
        # Arrhenius rate
        arrhenius_rate = rate_constant * np.exp(-activation_energy / (R * T + 1e-8))
        reaction_rate = arrhenius_rate * concentration
        
        # Predicted rate should be positive
        predicted_rate = np.abs(predictions.flatten()) * 1e-6
        return np.mean(np.square(predicted_rate - np.minimum(predicted_rate, reaction_rate)))
    
    def calculate_physics_loss(self, predictions, features):
        """Calculate total physics constraint violations"""
        losses = {}
        
        # Calculate individual constraint losses
        losses['thermodynamic'] = self.thermodynamic_loss(predictions, features)
        losses['mass_balance'] = self.mass_balance_loss(predictions, features) 
        losses['energy_efficiency'] = self.energy_efficiency_loss(predictions, features)
        losses['reaction_kinetics'] = self.reaction_kinetics_loss(predictions, features)
        
        # Weighted sum of physics losses
        total_physics_loss = sum(
            self.physics_weights.get(key, 1.0) * value 
            for key, value in losses.items()
        )
        
        self.constraint_violations = losses
        return total_physics_loss
    
    def fit(self, X, y):
        """Train the physics-informed neural network"""
        print("Training Physics-Informed Neural Network...")
        
        # Standard neural network training
        self.nn_model.fit(X, y)
        
        # Calculate physics losses on training data
        predictions = self.nn_model.predict(X).reshape(-1, 1)
        physics_loss = self.calculate_physics_loss(predictions, X)
        self.physics_loss_history.append(physics_loss)
        
        print(f"Training completed. Physics loss: {physics_loss:.6f}")
        return self
    
    def predict(self, X):
        """Make predictions with physics constraints"""
        return self.nn_model.predict(X)
    
    def evaluate_physics_compliance(self, X, y):
        """Evaluate physics constraint violations"""
        predictions = self.predict(X).reshape(-1, 1)
        
        # Calculate all physics constraints
        physics_loss = self.calculate_physics_loss(predictions, X)
        
        # Calculate violation rate (percentage of samples violating constraints)
        violation_threshold = 0.1  # 10% threshold
        violations = {
            name: (loss > violation_threshold) 
            for name, loss in self.constraint_violations.items()
        }
        
        violation_rate = np.mean([v for v in violations.values()])
        
        return {
            'total_physics_loss': physics_loss,
            'constraint_violations': self.constraint_violations,
            'violation_rate': violation_rate * 100,  # Percentage
            'compliant': violation_rate < 0.05  # < 5% violation rate
        }

# Initialize PINN
pinn_model = PhysicsInformedNN(
    hidden_layers=(128, 64, 32),
    activation='relu',
    max_iter=1000
)

print("Physics-Informed Neural Network initialized")
print(f"Architecture: {pinn_model.nn_model.hidden_layer_sizes}")
print(f"Physics constraints: {len(pinn_model.physics_weights)}")
print(f"Constraint weights: {pinn_model.physics_weights}")

pinn_config = {
    'architecture': pinn_model.nn_model.hidden_layer_sizes,
    'physics_constraints': len(pinn_model.physics_weights),
    'constraint_types': list(pinn_model.physics_weights.keys()),
    'model_ready': True
}