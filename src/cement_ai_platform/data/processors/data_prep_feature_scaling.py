import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create copy for scaling
scaled_df = cleaned_df.copy()

print("=== FEATURE SCALING ===")

# Separate numerical and categorical features
numerical_features = [col for col in scaled_df.columns if col.startswith('feature_')]
categorical_features = ['category_A', 'category_B']
target_col = 'target'

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Scale numerical features using StandardScaler
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(scaled_df[numerical_features])
scaled_df[numerical_features] = scaled_numerical

print("\nNumerical features scaled using StandardScaler")
print("Before scaling stats (first 3 features):")
print(cleaned_df[numerical_features[:3]].describe())
print("\nAfter scaling stats (first 3 features):")
print(scaled_df[numerical_features[:3]].describe())

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    scaled_df[f'{col}_encoded'] = le.fit_transform(scaled_df[col])
    label_encoders[col] = le
    print(f"\nEncoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nFinal dataset shape: {scaled_df.shape}")
print("Columns:", list(scaled_df.columns))