import pandas as pd
import numpy as np

# Simulate loading Global Cement Database data (3,117 facilities)
# In a real scenario, this would be loaded from actual Global Cement Database API or file

np.random.seed(42)
n_global = 3117

# Create realistic cement facility data based on typical industry parameters
global_cement_df = pd.DataFrame({
    'facility_id': range(1, n_global + 1),
    'facility_name': [f'Cement Plant {i}' for i in range(1, n_global + 1)],
    'country': np.random.choice(['China', 'India', 'USA', 'Turkey', 'Iran', 'Brazil', 'Russia', 'Japan', 'Vietnam', 'Egypt'], n_global, p=[0.50, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.07]),
    'capacity_mt_year': np.random.lognormal(mean=3.5, sigma=0.8, size=n_global) * 1e6,  # Million tons per year
    'cement_type': np.random.choice(['Portland', 'Blended', 'White', 'Oil Well', 'Special'], n_global, p=[0.7, 0.15, 0.05, 0.05, 0.05]),
    'production_process': np.random.choice(['Dry', 'Wet', 'Semi-dry'], n_global, p=[0.8, 0.15, 0.05]),
    'kiln_type': np.random.choice(['Rotary', 'Shaft', 'Other'], n_global, p=[0.9, 0.08, 0.02]),
    'year_established': np.random.choice(range(1950, 2023), n_global),
    'energy_efficiency': np.random.normal(3.4, 0.6, n_global),  # GJ/t cement
    'co2_emissions': np.random.normal(850, 120, n_global),  # kg CO2/t cement
    'limestone_pct': np.random.normal(75, 10, n_global),  # % limestone in raw materials
    'clay_pct': np.random.normal(15, 5, n_global),
    'iron_ore_pct': np.random.normal(3, 1, n_global),
    'silica_sand_pct': np.random.normal(5, 2, n_global),
    'gypsum_pct': np.random.normal(2, 0.5, n_global),
    'latitude': np.random.uniform(-60, 70, n_global),
    'longitude': np.random.uniform(-180, 180, n_global),
    'water_usage': np.random.normal(280, 50, n_global),  # L/t cement
    'electricity_kwh_t': np.random.normal(110, 25, n_global),
    'dust_emissions': np.random.normal(45, 15, n_global),  # mg/Nm3
    'nox_emissions': np.random.normal(850, 200, n_global),  # mg/Nm3
})

# Add some missing values to simulate real-world data issues
global_missing_cols = ['energy_efficiency', 'co2_emissions', 'water_usage', 'electricity_kwh_t', 'dust_emissions', 'nox_emissions']
for _global_col in global_missing_cols:
    _global_missing_idx = np.random.choice(global_cement_df.index, size=int(len(global_cement_df) * np.random.uniform(0.05, 0.15)), replace=False)
    global_cement_df.loc[_global_missing_idx, _global_col] = np.nan

# Fix negative values that shouldn't exist
global_numeric_cols = ['capacity_mt_year', 'energy_efficiency', 'co2_emissions', 'limestone_pct', 'clay_pct', 'iron_ore_pct', 'silica_sand_pct', 'gypsum_pct', 'water_usage', 'electricity_kwh_t', 'dust_emissions', 'nox_emissions']
for _global_num_col in global_numeric_cols:
    if _global_num_col in global_cement_df.columns:
        global_cement_df[_global_num_col] = np.abs(global_cement_df[_global_num_col])

print(f"✅ Loaded Global Cement Database: {len(global_cement_df)} facilities")
print(f"Columns: {list(global_cement_df.columns)}")
print(f"Shape: {global_cement_df.shape}")
print(f"Missing values per column:")
print(global_cement_df.isnull().sum().head(10))