import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

print("🏭 GENERATING COMPREHENSIVE CEMENT PLANT DATASET")
print("=" * 60)

# Generate 168 hours (1 week) of continuous operation data
# With multiple samples per hour to reach 1000+ records
hours = 168
samples_per_hour = 6  # Every 10 minutes
total_samples = hours * samples_per_hour
print(f"📊 Target: {total_samples} records across {hours} hours of operation")

# Create time-based index
start_time = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=10*i) for i in range(total_samples)]

# Create base DataFrame
cement_data = pd.DataFrame({'timestamp': timestamps})
cement_data['hour_of_day'] = cement_data['timestamp'].dt.hour
cement_data['day_of_week'] = cement_data['timestamp'].dt.dayofweek

print("✅ Time series framework created")

# 15 Critical Cement Plant Parameters
print("\n🎯 Generating 15 Critical Plant Parameters:")

# 1. Burning Zone Temperature (°C) - Main kiln temperature
base_temp = 1450  # Typical burning zone temp
temp_variation = np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) * 50  # Daily cycle
cement_data['burning_zone_temp'] = base_temp + temp_variation + np.random.normal(0, 15, total_samples)
print("   1. ✓ Burning Zone Temperature")

# 2. Preheater Temperature (°C) 
cement_data['preheater_temp'] = 850 + np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) * 30 + np.random.normal(0, 20, total_samples)
print("   2. ✓ Preheater Temperature")

# 3. Thermal Energy Consumption (GJ/ton)
base_thermal = 3.2
thermal_efficiency = 0.9 + 0.1 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24)
cement_data['thermal_energy'] = base_thermal / thermal_efficiency + np.random.normal(0, 0.15, total_samples)
print("   3. ✓ Thermal Energy Consumption")

# 4. Primary Fuel Rate (kg/h)
cement_data['primary_fuel_rate'] = 2800 + 200 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) + np.random.normal(0, 100, total_samples)
print("   4. ✓ Primary Fuel Rate")

# 5. Alternative Fuel Rate (%) - Waste-derived fuels
alt_fuel_base = 25  # 25% alternative fuel
cement_data['alternative_fuel_rate'] = np.clip(alt_fuel_base + np.random.normal(0, 5, total_samples), 0, 45)
print("   5. ✓ Alternative Fuel Rate")

# 6. Oxygen Content (%) - Critical for combustion
cement_data['oxygen_content'] = 2.5 + 1.0 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) + np.random.normal(0, 0.3, total_samples)
print("   6. ✓ Oxygen Content")

# 7. Raw Material Feed Rate (ton/h)
cement_data['raw_material_feed'] = 420 + 50 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) + np.random.normal(0, 20, total_samples)
print("   7. ✓ Raw Material Feed Rate")

# 8. Kiln Speed (RPM)
cement_data['kiln_speed'] = 3.2 + 0.3 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24) + np.random.normal(0, 0.1, total_samples)
print("   8. ✓ Kiln Speed")

# 9. Clinker Production Rate (ton/h)
# Correlated with feed rate and efficiency
efficiency_factor = 0.85 + 0.05 * np.sin(2 * np.pi * cement_data['hour_of_day'] / 24)
cement_data['clinker_production'] = cement_data['raw_material_feed'] * efficiency_factor + np.random.normal(0, 15, total_samples)
print("   9. ✓ Clinker Production Rate")

# 10. Clinker Quality - Free Lime Content (%)
cement_data['clinker_free_lime'] = np.clip(1.2 + 0.5 * np.random.normal(0, 1, total_samples), 0.3, 3.0)
print("   10. ✓ Clinker Free Lime Content")

# 11. Clinker Quality - C3S Content (%)
cement_data['clinker_c3s'] = np.clip(60 + np.random.normal(0, 3, total_samples), 50, 70)
print("   11. ✓ Clinker C3S Content")

# 12. Stack Emission - NOx (mg/m³)
cement_data['nox_emissions'] = 450 + 100 * (cement_data['alternative_fuel_rate'] / 50) + np.random.normal(0, 50, total_samples)
print("   12. ✓ NOx Emissions")

# 13. Stack Emission - SO2 (mg/m³)
cement_data['so2_emissions'] = 120 + 80 * (cement_data['alternative_fuel_rate'] / 50) + np.random.normal(0, 30, total_samples)
print("   13. ✓ SO2 Emissions")

# 14. Mill Power Consumption (kW)
cement_data['mill_power'] = 8500 + 1000 * (cement_data['raw_material_feed'] / 450) + np.random.normal(0, 200, total_samples)
print("   14. ✓ Mill Power Consumption")

# 15. Specific Energy Consumption (kWh/ton) - TARGET VARIABLE
# Physics-based relationship with other parameters
base_energy = 110
temp_penalty = (cement_data['burning_zone_temp'] - 1450) * 0.02
fuel_efficiency = (cement_data['thermal_energy'] - 3.2) * 15
production_efficiency = (450 - cement_data['clinker_production']) * 0.1
cement_data['specific_energy_consumption'] = (
    base_energy + temp_penalty + fuel_efficiency + production_efficiency + 
    np.random.normal(0, 5, total_samples)
)
print("   15. ✓ Specific Energy Consumption (Target)")

print(f"\n📋 Dataset Summary:")
print(f"   • Total Records: {len(cement_data):,}")
print(f"   • Time Period: {hours} hours ({hours/24:.1f} days)")
print(f"   • Parameters: 15 critical cement plant variables")
print(f"   • Sampling Rate: Every 10 minutes")

# Add categorical variables for plant operations
cement_data['shift'] = ['Day' if 6 <= h <= 14 else 'Evening' if 14 < h <= 22 else 'Night' 
                       for h in cement_data['hour_of_day']]
cement_data['kiln_campaign_phase'] = np.random.choice(['Early', 'Mid', 'Late'], size=total_samples, p=[0.3, 0.5, 0.2])

print(f"\n✅ COMPREHENSIVE CEMENT PLANT DATASET GENERATED")
print(f"Ready for AI training with realistic industrial correlations")
cement_df = cement_data.copy()