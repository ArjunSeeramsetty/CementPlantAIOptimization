import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define physics constraint functions for the PINN
class PhysicsConstraints:
    """Physics-based constraint functions for cement manufacturing process"""

    @staticmethod
    def thermodynamic_constraints(T_in, T_out, energy_consumption, Q_supplied):
        """First and Second Laws of Thermodynamics"""
        # First Law: Energy balance ΔU = Q - W
        energy_balance = torch.mean(torch.square(
            energy_consumption - Q_supplied * (T_out - T_in) / (T_in + 1e-8)
        ))

        # Second Law: Entropy cannot decrease (T_out >= T_in for heating)
        entropy_constraint = torch.mean(torch.square(
            torch.nn.functional.relu(T_in - T_out)  # Penalty if T_out < T_in
        ))

        return energy_balance + 0.5 * entropy_constraint

    @staticmethod
    def mass_balance_constraints(inputs, outputs):
        """Conservation of mass principle"""
        # Total mass in = Total mass out
        mass_in = torch.sum(inputs, dim=-1, keepdim=True)
        mass_out = torch.sum(outputs, dim=-1, keepdim=True)

        return torch.mean(torch.square(mass_in - mass_out))

    @staticmethod
    def heat_transfer_constraints(T_hot, T_cold, heat_transfer_coeff, area):
        """Newton's law of cooling"""
        # Q = U * A * ΔT
        predicted_heat = heat_transfer_coeff * area * (T_hot - T_cold)
        # Heat transfer should be proportional to temperature difference
        return torch.mean(torch.square(torch.nn.functional.relu(-predicted_heat)))

    @staticmethod
    def reaction_kinetics_constraints(T, concentration, rate_constant, activation_energy):
        """Arrhenius equation for reaction kinetics"""
        R = 8.314  # Gas constant

        # k = A * exp(-Ea / (R*T))
        arrhenius_rate = rate_constant * torch.exp(-activation_energy / (R * T + 1e-8))
        reaction_rate = arrhenius_rate * concentration

        # Rate must be positive and follow Arrhenius behavior
        return torch.mean(torch.square(torch.nn.functional.relu(-reaction_rate)))

    @staticmethod
    def energy_efficiency_constraints(energy_input, useful_energy_output):
        """Energy efficiency must be <= 1"""
        efficiency = useful_energy_output / (energy_input + 1e-8)

        # Penalty for efficiency > 1 (impossible)
        return torch.mean(torch.square(torch.nn.functional.relu(efficiency - 1.0)))

    @staticmethod
    def material_composition_constraints(components):
        """Material composition must sum to 1"""
        total_composition = torch.sum(components, dim=-1)

        return torch.mean(torch.square(total_composition - 1.0))

    @staticmethod
    def chemical_equilibrium_constraints(reactants, products, K_eq, temperature):
        """Chemical equilibrium constraint: K = [products]/[reactants]"""
        # Le Chatelier's principle - equilibrium constant depends on temperature
        K_T = K_eq * torch.exp((temperature - 298.15) / 298.15)  # Temperature dependence

        product_ratio = torch.prod(products, dim=-1)
        reactant_ratio = torch.prod(reactants, dim=-1) + 1e-8

        equilibrium_violation = torch.mean(torch.square(
            product_ratio / reactant_ratio - K_T
        ))

        return equilibrium_violation

    @staticmethod
    def momentum_conservation_constraints(velocity_in, velocity_out, mass_flow):
        """Conservation of momentum in fluid flow"""
        momentum_in = mass_flow * velocity_in
        momentum_out = mass_flow * velocity_out

        return torch.mean(torch.square(momentum_in - momentum_out))

# Initialize physics constraints
physics_constraints = PhysicsConstraints()

print("Physics constraints framework initialized with PyTorch")
print("Available constraints:")
print("1. Thermodynamic constraints (1st & 2nd laws)")
print("2. Mass balance constraints")
print("3. Heat transfer constraints (Newton's law)")
print("4. Reaction kinetics constraints (Arrhenius)")
print("5. Energy efficiency constraints")
print("6. Material composition constraints")
print("7. Chemical equilibrium constraints")
print("8. Momentum conservation constraints")

# Define constraint violation penalty weights
constraint_weights = {
    'thermodynamic': 1.0,
    'mass_balance': 2.0,
    'heat_transfer': 1.5,
    'reaction_kinetics': 1.2,
    'energy_efficiency': 0.8,
    'material_composition': 1.5,
    'chemical_equilibrium': 1.0,
    'momentum_conservation': 0.9
}

print(f"\nConstraint weights: {constraint_weights}")
print(f"Total physics constraints implemented: {len(constraint_weights)}")

# Verify PyTorch is working
test_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"\nPyTorch test: {test_tensor.mean().item():.2f}")
print("Physics constraints framework ready!")