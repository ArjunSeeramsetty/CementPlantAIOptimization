import pandas as pd
import numpy as np

# Analyze both datasets for integration planning
print("=" * 60)
print("DATASET ANALYSIS FOR INTEGRATION")
print("=" * 60)

print(f"\n📊 GLOBAL CEMENT DATABASE (Facility-level data)")
print(f"Records: {len(global_cement_df)}")
print(f"Columns: {len(global_cement_df.columns)}")
print(f"Data type: Facility/production data")

print(f"\nGlobal Cement columns:")
for col in global_cement_df.columns:
    dtype = str(global_cement_df[col].dtype)
    missing = global_cement_df[col].isnull().sum()
    missing_pct = (missing / len(global_cement_df)) * 100
    print(f"  {col:20} | {dtype:10} | {missing:4} missing ({missing_pct:.1f}%)")

print(f"\n📊 KAGGLE CEMENT DATABASE (Composition/testing data)")
print(f"Records: {len(kaggle_cement_df)}")
print(f"Columns: {len(kaggle_cement_df.columns)}")
print(f"Data type: Composition/performance data")

print(f"\nKaggle Cement columns:")
for col in kaggle_cement_df.columns:
    dtype = str(kaggle_cement_df[col].dtype)
    missing = kaggle_cement_df[col].isnull().sum()
    missing_pct = (missing / len(kaggle_cement_df)) * 100
    print(f"  {col:20} | {dtype:10} | {missing:4} missing ({missing_pct:.1f}%)")

# Identify potential integration points
print(f"\n🔗 INTEGRATION ANALYSIS")
print(f"=" * 40)

# Common concepts (not exact column matches but related domains)
global_concepts = set(['cement_type', 'production_process', 'energy_efficiency', 'co2_emissions'])
kaggle_concepts = set(['cement', 'compressive_strength', 'cement_type_detailed', 'w_c_ratio'])

print(f"Global dataset focus: Manufacturing/facility operations")
print(f"Kaggle dataset focus: Composition/performance testing")
print(f"Integration approach: Cross-reference by cement type and performance metrics")

# Calculate basic statistics for key numeric columns
print(f"\n📈 KEY METRICS COMPARISON")
print(f"Global cement facilities by type:")
print(global_cement_df['cement_type'].value_counts())

print(f"\nKaggle cement samples by detailed type:")
print(kaggle_cement_df['cement_type_detailed'].value_counts())

print(f"\nGlobal CO2 emissions stats: {global_cement_df['co2_emissions'].describe().round(1)}")
print(f"Kaggle compressive strength stats: {kaggle_cement_df['compressive_strength'].describe().round(1)}")

print(f"\n✅ Analysis complete - Ready for integration schema design")