import pandas as pd
import numpy as np

print("🔗 INTEGRATING CEMENT DATASETS")
print("=" * 50)

# Initialize integrated dataset with unified schema
integrated_columns = list(unified_schema.keys())
integrated_df = pd.DataFrame(columns=integrated_columns)

print(f"Processing Global Cement Database ({len(global_cement_df)} records)...")

# Transform Global Cement Database to unified schema
global_transformed = pd.DataFrame()
_record_counter = 1

# Basic identification
global_transformed['record_id'] = [f'GLOBAL_{i:05d}' for i in range(1, len(global_cement_df) + 1)]
global_transformed['data_source'] = 'Global_DB'
global_transformed['record_type'] = 'Facility'

# Location and facility info
global_transformed['country'] = global_cement_df['country']
global_transformed['facility_name'] = global_cement_df['facility_name']
global_transformed['latitude'] = global_cement_df['latitude']
global_transformed['longitude'] = global_cement_df['longitude']

# Cement type (standardized)
global_transformed['cement_type_standard'] = global_cement_df['cement_type'].map(cement_type_mapping)
global_transformed['cement_type_detailed'] = global_cement_df['cement_type']

# Production information
global_transformed['capacity_mt_year'] = global_cement_df['capacity_mt_year']
global_transformed['year_established'] = global_cement_df['year_established']
global_transformed['production_process'] = global_cement_df['production_process']
global_transformed['kiln_type'] = global_cement_df['kiln_type']

# Raw material composition
global_transformed['limestone_pct'] = global_cement_df['limestone_pct']
global_transformed['clay_pct'] = global_cement_df['clay_pct']
global_transformed['iron_ore_pct'] = global_cement_df['iron_ore_pct']
global_transformed['silica_sand_pct'] = global_cement_df['silica_sand_pct']
global_transformed['gypsum_pct'] = global_cement_df['gypsum_pct']

# Environmental metrics
global_transformed['energy_efficiency_gj_t'] = global_cement_df['energy_efficiency']
global_transformed['co2_emissions_kg_t'] = global_cement_df['co2_emissions']
global_transformed['water_usage_l_t'] = global_cement_df['water_usage']
global_transformed['electricity_kwh_t'] = global_cement_df['electricity_kwh_t']
global_transformed['dust_emissions_mg_nm3'] = global_cement_df['dust_emissions']
global_transformed['nox_emissions_mg_nm3'] = global_cement_df['nox_emissions']

# Add remaining columns as NaN for Global DB
composition_cols = ['cement_content_kg_m3', 'blast_furnace_slag_kg_m3', 'fly_ash_kg_m3', 'water_kg_m3', 'superplasticizer_kg_m3', 'coarse_aggregate_kg_m3', 'fine_aggregate_kg_m3']
performance_cols = ['compressive_strength_mpa', 'w_c_ratio', 'slump_cm', 'age_days']
process_cols = ['curing_temp_c', 'curing_humidity_pct', 'aggregate_size_mm', 'admixture_type', 'mix_design', 'test_method']

for _col_name in composition_cols + performance_cols + process_cols:
    global_transformed[_col_name] = np.nan

print(f"✅ Transformed Global DB: {len(global_transformed)} records")

# Transform Kaggle dataset to unified schema
print(f"Processing Kaggle dataset ({len(kaggle_cement_df)} records)...")

kaggle_transformed = pd.DataFrame()

# Basic identification
kaggle_transformed['record_id'] = [f'KAGGLE_{i:05d}' for i in range(1, len(kaggle_cement_df) + 1)]
kaggle_transformed['data_source'] = 'Kaggle'
kaggle_transformed['record_type'] = 'Sample'

# Cement type (standardized)
kaggle_transformed['cement_type_standard'] = kaggle_cement_df['cement_type_detailed'].map(cement_type_mapping)
kaggle_transformed['cement_type_detailed'] = kaggle_cement_df['cement_type_detailed']

# Composition data (kg/m3)
kaggle_transformed['cement_content_kg_m3'] = kaggle_cement_df['cement']
kaggle_transformed['blast_furnace_slag_kg_m3'] = kaggle_cement_df['blast_furnace_slag']
kaggle_transformed['fly_ash_kg_m3'] = kaggle_cement_df['fly_ash']
kaggle_transformed['water_kg_m3'] = kaggle_cement_df['water']
kaggle_transformed['superplasticizer_kg_m3'] = kaggle_cement_df['superplasticizer']
kaggle_transformed['coarse_aggregate_kg_m3'] = kaggle_cement_df['coarse_aggregate']
kaggle_transformed['fine_aggregate_kg_m3'] = kaggle_cement_df['fine_aggregate']

# Performance metrics
kaggle_transformed['compressive_strength_mpa'] = kaggle_cement_df['compressive_strength']
kaggle_transformed['w_c_ratio'] = kaggle_cement_df['w_c_ratio']
kaggle_transformed['slump_cm'] = kaggle_cement_df['slump']
kaggle_transformed['age_days'] = kaggle_cement_df['age_days']

# Process parameters
kaggle_transformed['curing_temp_c'] = kaggle_cement_df['curing_temp']
kaggle_transformed['curing_humidity_pct'] = kaggle_cement_df['curing_humidity']
kaggle_transformed['aggregate_size_mm'] = kaggle_cement_df['aggregate_size']
kaggle_transformed['admixture_type'] = kaggle_cement_df['admixture_type']
kaggle_transformed['mix_design'] = kaggle_cement_df['mix_design']
kaggle_transformed['test_method'] = kaggle_cement_df['test_method']

# Add remaining columns as NaN for Kaggle data
location_cols = ['country', 'facility_name', 'latitude', 'longitude']
production_cols = ['capacity_mt_year', 'year_established', 'production_process', 'kiln_type']
raw_material_cols = ['limestone_pct', 'clay_pct', 'iron_ore_pct', 'silica_sand_pct', 'gypsum_pct']
env_cols = ['energy_efficiency_gj_t', 'co2_emissions_kg_t', 'water_usage_l_t', 'electricity_kwh_t', 'dust_emissions_mg_nm3', 'nox_emissions_mg_nm3']

for _col_name in location_cols + production_cols + raw_material_cols + env_cols:
    kaggle_transformed[_col_name] = np.nan

print(f"✅ Transformed Kaggle data: {len(kaggle_transformed)} records")

# Combine both datasets
print(f"Combining datasets...")
integrated_df = pd.concat([global_transformed, kaggle_transformed], ignore_index=True)

# Ensure column order matches schema
integrated_df = integrated_df[integrated_columns]

print(f"🎉 INTEGRATION COMPLETE!")
print(f"Total records: {len(integrated_df)}")
print(f"Total columns: {len(integrated_df.columns)}")
print(f"Global DB records: {sum(integrated_df['data_source'] == 'Global_DB')}")
print(f"Kaggle records: {sum(integrated_df['data_source'] == 'Kaggle')}")

# Quick data type summary
print(f"\nData completeness by source:")
for source in ['Global_DB', 'Kaggle']:
    _source_data = integrated_df[integrated_df['data_source'] == source]
    _total_cells = len(_source_data) * len(_source_data.columns)
    _non_null_cells = _source_data.count().sum()
    _completeness = (_non_null_cells / _total_cells) * 100
    print(f"  {source}: {_completeness:.1f}% complete")