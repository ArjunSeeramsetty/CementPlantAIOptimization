import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Create a copy for processing
cleaned_df = missing_df.copy()

print("=== HANDLING MISSING VALUES ===")
print("Before cleaning:")
before_missing = cleaned_df.isnull().sum()
print(before_missing[before_missing > 0])

# Handle numerical missing values with median imputation
numerical_cols = ['feature_1', 'feature_3', 'feature_5']
for col in numerical_cols:
    if cleaned_df[col].isnull().sum() > 0:
        median_val = cleaned_df[col].median()
        cleaned_df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} missing values with median: {median_val:.2f}")

# Handle categorical missing values with mode imputation
categorical_cols = ['category_A']
for col in categorical_cols:
    if cleaned_df[col].isnull().sum() > 0:
        mode_val = cleaned_df[col].mode()[0]
        cleaned_df[col].fillna(mode_val, inplace=True)
        print(f"Filled {col} missing values with mode: {mode_val}")

print("\nAfter cleaning:")
after_missing = cleaned_df.isnull().sum()
remaining_missing = after_missing[after_missing > 0]
if len(remaining_missing) == 0:
    print("All missing values successfully handled!")
else:
    print(remaining_missing)

print(f"Dataset shape after cleaning: {cleaned_df.shape}")