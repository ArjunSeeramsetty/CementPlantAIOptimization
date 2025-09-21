import numpy as np
import pandas as pd
from scipy.stats import norm, uniform, multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class PhysicsCompliantDataGenerator:
    """Generate realistic synthetic data that follows physics-based constraints"""

    def __init__(self, thermo_framework, kiln_simulator, process_units):
        self.thermo = thermo_framework
        self.kiln = kiln_simulator
        self.units = process_units

        # Operating ranges for key variables (realistic industrial values)
        self.operating_ranges = {
            'kiln_feed_rate': (120, 180),      # t/h
            'fuel_rate': (4000, 6000),         # kg/h
            'kiln_rotation_speed': (1.5, 3.0), # rpm
            'primary_air_temp': (850, 1100),   # °C
            'coal_moisture': (2, 8),           # %
            'raw_meal_fineness': (8, 15),      # % residue on 90μm
            'kiln_inlet_temp': (1400, 1600),   # °C
            'cooler_air_flow': (150, 250),     # m³/min per t clinker
            'preheater_stages': (4, 6),        # number
            'limestone_content': (70, 85),     # % in raw meal
            'silica_ratio': (2.3, 2.8),       # SiO2/(Al2O3+Fe2O3)
            'alumina_ratio': (1.2, 2.5)       # Al2O3/Fe2O3
        }

        # Correlation structure for realistic relationships
        self.correlation_matrix = np.array([
            [1.00, 0.85, 0.45, -0.30, 0.20],  # feed_rate
            [0.85, 1.00, 0.40, -0.25, 0.15],  # fuel_rate
            [0.45, 0.40, 1.00, -0.60, 0.35],  # kiln_speed
            [-0.30, -0.25, -0.60, 1.00, -0.40], # primary_air_temp
            [0.20, 0.15, 0.35, -0.40, 1.00]   # coal_moisture
        ])

    def generate_operating_conditions(self, n_samples=1000):
        """Generate realistic operating condition scenarios"""

        # Generate correlated variables using multivariate normal
        mean_vals = np.array([150, 5000, 2.25, 975, 5])
        std_vals = np.array([15, 500, 0.375, 62.5, 1.5])

        # Scale correlation matrix to covariance
        cov_matrix = np.outer(std_vals, std_vals) * self.correlation_matrix

        samples = multivariate_normal.rvs(mean_vals, cov_matrix, n_samples)

        # Apply realistic bounds
        operating_data = pd.DataFrame({
            'kiln_feed_rate_tph': np.clip(samples[:, 0], *self.operating_ranges['kiln_feed_rate']),
            'fuel_rate_kgh': np.clip(samples[:, 1], *self.operating_ranges['fuel_rate']),
            'kiln_rotation_rpm': np.clip(samples[:, 2], *self.operating_ranges['kiln_rotation_speed']),
            'primary_air_temp_C': np.clip(samples[:, 3], *self.operating_ranges['primary_air_temp']),
            'coal_moisture_pct': np.clip(samples[:, 4], *self.operating_ranges['coal_moisture'])
        })

        # Add additional independent variables
        operating_data['raw_meal_fineness_pct'] = uniform.rvs(
            *self.operating_ranges['raw_meal_fineness'], size=n_samples
        )
        operating_data['limestone_content_pct'] = uniform.rvs(
            *self.operating_ranges['limestone_content'], size=n_samples
        )
        operating_data['silica_ratio'] = uniform.rvs(
            *self.operating_ranges['silica_ratio'], size=n_samples
        )
        operating_data['alumina_ratio'] = uniform.rvs(
            *self.operating_ranges['alumina_ratio'], size=n_samples
        )

        return operating_data

    def calculate_physics_responses(self, operating_data):
        """Calculate physics-compliant response variables"""

        n_samples = len(operating_data)
        responses = pd.DataFrame(index=operating_data.index)

        # Calculate responses for each sample
        for idx, row in operating_data.iterrows():

            # Kiln thermal efficiency (physics-based)
            feed_rate = row['kiln_feed_rate_tph']
            fuel_rate = row['fuel_rate_kgh']
            coal_moisture = row['coal_moisture_pct']

            # Heat balance calculations
            fuel_hv = 30000 - coal_moisture * 200  # kJ/kg (moisture reduces heating value)
            heat_input = fuel_rate * fuel_hv / 3600  # kW

            # Heat requirements
            sensible_heat = feed_rate * 1000 * 1.0 * (1450 - 100) / 3600  # kW
            calcination_heat = feed_rate * 1000 * 0.75 * 1780 / 3600 / 100  # kW (75% limestone)
            heat_losses = 0.15 * heat_input  # 15% heat loss

            thermal_efficiency = min(0.95, (sensible_heat + calcination_heat) /
                                   max(heat_input - heat_losses, 1))

            # Clinker quality based on chemistry and thermal profile
            lime_saturation = (row['limestone_content_pct'] - 75) / 10  # Normalized
            thermal_factor = thermal_efficiency
            fineness_factor = (15 - row['raw_meal_fineness_pct']) / 7  # Normalized

            # C3S content (key clinker mineral)
            c3s_content = 55 + 8 * lime_saturation + 5 * thermal_factor + 3 * fineness_factor
            c3s_content = np.clip(c3s_content, 45, 70)

            # SO3 in clinker (affected by fuel sulfur and process conditions)
            fuel_sulfur = 1.0 + 0.3 * np.random.normal()  # % sulfur in coal
            so3_content = fuel_sulfur * 0.4 + 0.2 * (row['primary_air_temp_C'] - 900) / 100
            so3_content = np.clip(so3_content, 0.5, 3.5)

            # Specific heat consumption
            theoretical_heat = 1750  # kJ/kg clinker (theoretical minimum)
            process_inefficiency = (100 - thermal_efficiency * 100) * 15
            shc = theoretical_heat + process_inefficiency + np.random.normal(0, 50)

            # Free lime (unreacted CaO - quality indicator)
            mixing_quality = 1.0 - row['raw_meal_fineness_pct'] / 20
            free_lime = 2.5 - 1.5 * thermal_efficiency - mixing_quality
            free_lime = np.clip(free_lime + np.random.normal(0, 0.3), 0.2, 4.0)

            # Clinker cooler performance
            cooler_air = 200 + 30 * (feed_rate - 150) / 30  # Scaled air flow
            clinker_temp_out = 90 + 40 * np.exp(-(cooler_air - 150) / 50)

            # Store responses
            responses.loc[idx, 'thermal_efficiency_pct'] = thermal_efficiency * 100
            responses.loc[idx, 'c3s_content_pct'] = c3s_content
            responses.loc[idx, 'so3_content_pct'] = so3_content
            responses.loc[idx, 'specific_heat_consumption'] = shc
            responses.loc[idx, 'free_lime_pct'] = free_lime
            responses.loc[idx, 'clinker_temp_out_C'] = clinker_temp_out

            # Add some process noise (measurement uncertainty)
            for col in responses.columns:
                if col in responses.loc[idx]:
                    noise_level = abs(responses.loc[idx, col]) * 0.02  # 2% noise
                    responses.loc[idx, col] += np.random.normal(0, noise_level)

        return responses

    def add_process_disturbances(self, data, disturbance_level=0.05):
        """Add realistic process disturbances and measurement noise"""

        disturbed_data = data.copy()

        # Add systematic disturbances (equipment wear, seasonal effects)
        n_samples = len(data)

        # Equipment efficiency drift
        efficiency_drift = np.linspace(1.0, 0.95, n_samples)  # 5% degradation
        thermal_efficiency_cols = [col for col in data.columns if 'efficiency' in col]
        for col in thermal_efficiency_cols:
            disturbed_data[col] *= efficiency_drift

        # Feed variability (realistic raw material variation)
        feed_variation = 1 + disturbance_level * np.random.normal(0, 1, n_samples)
        chemistry_cols = ['limestone_content_pct', 'silica_ratio', 'alumina_ratio']
        for col in chemistry_cols:
            if col in disturbed_data.columns:
                disturbed_data[col] *= feed_variation

        # Measurement noise on all variables
        for col in disturbed_data.select_dtypes(include=[np.number]).columns:
            noise_std = abs(disturbed_data[col].mean()) * disturbance_level * 0.5
            noise = np.random.normal(0, noise_std, n_samples)
            disturbed_data[col] += noise

        return disturbed_data

    def generate_complete_dataset(self, n_samples=1000):
        """Generate complete physics-compliant synthetic dataset"""

        print(f"Generating {n_samples} physics-compliant samples...")

        # Step 1: Generate operating conditions
        operating_conditions = self.generate_operating_conditions(n_samples)

        # Step 2: Calculate physics-based responses
        physics_responses = self.calculate_physics_responses(operating_conditions)

        # Step 3: Combine into complete dataset
        complete_data = pd.concat([operating_conditions, physics_responses], axis=1)

        # Step 4: Add realistic disturbances
        final_data = self.add_process_disturbances(complete_data)

        return final_data

# Initialize data generator
data_generator = PhysicsCompliantDataGenerator(thermo_framework, kiln_simulator, process_units)

# Generate synthetic dataset
synthetic_physics_data = data_generator.generate_complete_dataset(n_samples=1000)

print("=== Physics-Compliant Dataset Generated ===")
print(f"Dataset shape: {synthetic_physics_data.shape}")
print(f"Variables: {list(synthetic_physics_data.columns)}")
print("\nSample statistics:")
print(synthetic_physics_data.describe().round(2))

# Check physics constraints
print("\n=== Physics Validation ===")
thermal_eff_range = (synthetic_physics_data['thermal_efficiency_pct'].min(),
                    synthetic_physics_data['thermal_efficiency_pct'].max())
print(f"Thermal efficiency range: {thermal_eff_range[0]:.1f}% - {thermal_eff_range[1]:.1f}%")

c3s_range = (synthetic_physics_data['c3s_content_pct'].min(),
            synthetic_physics_data['c3s_content_pct'].max())
print(f"C3S content range: {c3s_range[0]:.1f}% - {c3s_range[1]:.1f}%")

heat_consumption_avg = synthetic_physics_data['specific_heat_consumption'].mean()
print(f"Average specific heat consumption: {heat_consumption_avg:.0f} kJ/kg")

# Store results
physics_dataset = synthetic_physics_data
physics_generator = data_generator