import pandas as pd
import numpy as np

# Create a copy for missing value handling demonstration
missing_df = synthetic_df.copy()

# Inject some missing values randomly for demonstration
np.random.seed(42)
n_rows = len(missing_df)

# Inject missing values in numerical features (5% missing rate)
for col in ['feature_1', 'feature_3', 'feature_5']:
    missing_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
    missing_df.loc[missing_indices, col] = np.nan

# Inject missing values in categorical features (3% missing rate)
for col in ['category_A']:
    missing_indices = np.random.choice(n_rows, size=int(0.03 * n_rows), replace=False)
    missing_df.loc[missing_indices, col] = np.nan

print("=== MISSING VALUES INTRODUCED ===")
missing_counts = missing_df.isnull().sum()
print(missing_counts[missing_counts > 0])
print(f"Total missing values: {missing_df.isnull().sum().sum()}")
print(f"Missing percentage: {(missing_df.isnull().sum().sum() / (len(missing_df) * len(missing_df.columns))) * 100:.2f}%")