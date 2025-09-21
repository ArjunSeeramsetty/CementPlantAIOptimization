import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize
from scipy.constants import R  # Gas constant
import warnings
warnings.filterwarnings('ignore')

class ThermodynamicFramework:
    """Core thermodynamic modeling framework for cement processes"""

    def __init__(self):
        # Thermodynamic constants
        self.R = R  # 8.314 J/(mol·K)

        # Component properties for cement raw materials
        self.components = {
            'CaCO3': {'MW': 100.09, 'Hf': -1206.9e3},  # kJ/mol
            'SiO2': {'MW': 60.08, 'Hf': -910.9e3},
            'Al2O3': {'MW': 101.96, 'Hf': -1675.7e3},
            'Fe2O3': {'MW': 159.69, 'Hf': -824.2e3},
            'CaO': {'MW': 56.08, 'Hf': -635.1e3},
            'CO2': {'MW': 44.01, 'Hf': -393.5e3}
        }

        # Clinker phase properties
        self.clinker_phases = {
            'C3S': {'MW': 228.32, 'Hf': -2307.8e3},  # Tricalcium silicate
            'C2S': {'MW': 172.24, 'Hf': -2316.6e3},  # Dicalcium silicate
            'C3A': {'MW': 270.20, 'Hf': -3587.8e3},  # Tricalcium aluminate
            'C4AF': {'MW': 485.96, 'Hf': -5091.2e3}  # Tetracalcium aluminoferrite
        }

    def antoine_vapor_pressure(self, T, A, B, C):
        """Calculate vapor pressure using Antoine equation"""
        return 10**(A - B/(C + T))  # T in °C, P in mmHg

    def calculate_enthalpy(self, component, T, T_ref=298.15):
        """Calculate enthalpy at temperature T"""
        # Simplified heat capacity correlation: Cp = a + bT + cT²
        cp_params = {
            'CaCO3': {'a': 104.5, 'b': 0.0218, 'c': -2.59e-6},
            'CaO': {'a': 49.62, 'b': 0.00467, 'c': -1.28e-6},
            'CO2': {'a': 28.46, 'b': 0.00365, 'c': -0.196e-6}
        }

        if component in cp_params:
            params = cp_params[component]
            delta_H = (params['a'] * (T - T_ref) +
                      params['b']/2 * (T**2 - T_ref**2) +
                      params['c']/3 * (T**3 - T_ref**3))
            return self.components[component]['Hf'] + delta_H
        else:
            # Simple estimation for missing components
            return self.components[component]['Hf']

    def calcination_equilibrium(self, T, P_CO2=1.0):
        """Calculate CaCO3 calcination equilibrium"""
        # CaCO3 ⇌ CaO + CO2
        # Using simplified Van't Hoff equation
        delta_H = 178.3e3  # J/mol (heat of calcination)
        T_eq = 1156  # Equilibrium temperature at 1 atm

        ln_Keq = -delta_H/self.R * (1/T - 1/T_eq)
        Keq = np.exp(ln_Keq)

        # For CaCO3 ⇌ CaO + CO2, Keq = P_CO2
        conversion = min(1.0, max(0.0, (Keq - P_CO2) / Keq))
        return conversion

    def clinker_formation_kinetics(self, T, time, composition):
        """Model clinker formation kinetics"""
        # Simplified Arrhenius kinetics for C3S formation
        # Rate equation: r = k * exp(-Ea/RT) * f(composition)

        Ea_C3S = 250e3  # Activation energy J/mol
        A_C3S = 1e8     # Pre-exponential factor

        k = A_C3S * np.exp(-Ea_C3S / (self.R * T))

        # Composition factor (simplified)
        CaO_excess = composition.get('CaO_excess', 0.1)
        comp_factor = min(1.0, CaO_excess * 2.0)

        # Formation rate
        rate = k * comp_factor * time
        C3S_formation = 1 - np.exp(-rate)

        return {
            'C3S_fraction': C3S_formation,
            'rate_constant': k,
            'formation_rate': rate
        }

# Initialize framework
thermo_framework = ThermodynamicFramework()

# Test calculations
test_temp = 1723  # 1450°C in Kelvin
test_composition = {'CaO_excess': 0.15}

calcination_conv = thermo_framework.calcination_equilibrium(test_temp)
clinker_kinetics = thermo_framework.clinker_formation_kinetics(
    test_temp, 3600, test_composition  # 1 hour residence time
)

print("=== Thermodynamic Framework Initialized ===")
print(f"Calcination conversion at {test_temp-273}°C: {calcination_conv:.3f}")
print(f"C3S formation rate: {clinker_kinetics['formation_rate']:.2e}")
print(f"C3S fraction formed: {clinker_kinetics['C3S_fraction']:.3f}")

# Store results for downstream use
thermo_results = {
    'framework': thermo_framework,
    'test_temperature': test_temp,
    'calcination_conversion': calcination_conv,
    'clinker_kinetics': clinker_kinetics
}