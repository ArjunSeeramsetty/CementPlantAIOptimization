import pandas as pd
import numpy as np

# Design unified schema for integrated cement dataset
print("🏗️ DESIGNING UNIFIED CEMENT DATA SCHEMA")
print("=" * 50)

# Define unified schema mapping both data sources
unified_schema = {
    # Common identification fields
    'record_id': 'Unique identifier for each record',
    'data_source': 'Source of data (Global_DB or Kaggle)',
    'record_type': 'Type of record (Facility or Sample)',

    # Location and facility information
    'country': 'Country location',
    'facility_name': 'Facility name (for Global DB)',
    'latitude': 'Latitude coordinate',
    'longitude': 'Longitude coordinate',

    # Cement type information (standardized)
    'cement_type_standard': 'Standardized cement type',
    'cement_type_detailed': 'Detailed cement type',

    # Production/capacity information
    'capacity_mt_year': 'Annual production capacity (MT)',
    'year_established': 'Year facility established',
    'production_process': 'Production process type',
    'kiln_type': 'Kiln type used',

    # Composition data (from Kaggle, normalized)
    'cement_content_kg_m3': 'Cement content kg/m3',
    'blast_furnace_slag_kg_m3': 'Blast furnace slag content',
    'fly_ash_kg_m3': 'Fly ash content',
    'water_kg_m3': 'Water content kg/m3',
    'superplasticizer_kg_m3': 'Superplasticizer content',
    'coarse_aggregate_kg_m3': 'Coarse aggregate content',
    'fine_aggregate_kg_m3': 'Fine aggregate content',

    # Raw material composition (from Global DB)
    'limestone_pct': 'Limestone percentage in raw materials',
    'clay_pct': 'Clay percentage',
    'iron_ore_pct': 'Iron ore percentage',
    'silica_sand_pct': 'Silica sand percentage',
    'gypsum_pct': 'Gypsum percentage',

    # Performance metrics
    'compressive_strength_mpa': 'Compressive strength (MPa)',
    'w_c_ratio': 'Water to cement ratio',
    'slump_cm': 'Slump test result (cm)',
    'age_days': 'Age at testing (days)',

    # Environmental and efficiency metrics
    'energy_efficiency_gj_t': 'Energy efficiency (GJ/t)',
    'co2_emissions_kg_t': 'CO2 emissions (kg/t)',
    'water_usage_l_t': 'Water usage (L/t)',
    'electricity_kwh_t': 'Electricity consumption (kWh/t)',
    'dust_emissions_mg_nm3': 'Dust emissions (mg/Nm3)',
    'nox_emissions_mg_nm3': 'NOx emissions (mg/Nm3)',

    # Process parameters
    'curing_temp_c': 'Curing temperature (°C)',
    'curing_humidity_pct': 'Curing humidity (%)',
    'aggregate_size_mm': 'Aggregate size (mm)',
    'admixture_type': 'Type of admixture used',
    'mix_design': 'Mix design category',
    'test_method': 'Testing method used'
}

print(f"Unified schema includes {len(unified_schema)} fields")
print("\n📋 Schema categories:")
categories = {
    'Identification': ['record_id', 'data_source', 'record_type'],
    'Location': ['country', 'facility_name', 'latitude', 'longitude'],
    'Cement Type': ['cement_type_standard', 'cement_type_detailed'],
    'Production': ['capacity_mt_year', 'year_established', 'production_process', 'kiln_type'],
    'Composition': ['cement_content_kg_m3', 'blast_furnace_slag_kg_m3', 'fly_ash_kg_m3', 'water_kg_m3'],
    'Raw Materials': ['limestone_pct', 'clay_pct', 'iron_ore_pct', 'silica_sand_pct', 'gypsum_pct'],
    'Performance': ['compressive_strength_mpa', 'w_c_ratio', 'slump_cm', 'age_days'],
    'Environment': ['energy_efficiency_gj_t', 'co2_emissions_kg_t', 'water_usage_l_t', 'electricity_kwh_t'],
    'Process': ['curing_temp_c', 'curing_humidity_pct', 'aggregate_size_mm', 'admixture_type']
}

for category, fields in categories.items():
    print(f"  {category}: {len(fields)} fields")

# Create cement type mapping for standardization
cement_type_mapping = {
    # Global DB to Standard
    'Portland': 'Ordinary Portland Cement',
    'Blended': 'Blended Cement',
    'White': 'White Portland Cement',
    'Oil Well': 'Oil Well Cement',
    'Special': 'Specialty Cement',

    # Kaggle detailed to Standard
    'OPC 43': 'Ordinary Portland Cement',
    'OPC 53': 'Ordinary Portland Cement',
    'PPC': 'Portland Pozzolan Cement',
    'PSC': 'Portland Slag Cement',
    'SRC': 'Sulfate Resistant Cement'
}

print(f"\n🔧 Cement type standardization mapping:")
for original, standard in cement_type_mapping.items():
    print(f"  {original} → {standard}")

print(f"\n✅ Schema design complete")
print(f"Ready for data integration and transformation")