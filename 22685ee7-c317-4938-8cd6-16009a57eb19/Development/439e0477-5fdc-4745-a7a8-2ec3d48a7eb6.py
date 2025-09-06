import pandas as pd

# Data structure exploration
print("=== DATASET OVERVIEW ===")
print(f"Shape: {synthetic_df.shape}")
print(f"Memory usage: {synthetic_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print()

print("=== DATA TYPES ===")
print(synthetic_df.dtypes)
print()

print("=== FIRST 5 ROWS ===")
print(synthetic_df.head())
print()

print("=== BASIC STATISTICS ===")
exploration_stats = synthetic_df.describe()
print(exploration_stats)
print()

print("=== CATEGORICAL FEATURES ===")
for col in ['category_A', 'category_B']:
    print(f"{col}: {synthetic_df[col].value_counts().to_dict()}")
print()

print("=== MISSING VALUES ===")
missing_summary = synthetic_df.isnull().sum()
print(missing_summary[missing_summary > 0] if missing_summary.sum() > 0 else "No missing values found")