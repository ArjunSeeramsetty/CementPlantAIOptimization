import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("=== TRAIN/TEST SPLIT ===")

# Prepare features and target
# Use scaled numerical features and encoded categorical features for modeling
model_features = numerical_features + ['category_A_encoded', 'category_B_encoded']
X_final = scaled_df[model_features]
y_final = scaled_df['target']

print(f"Features for modeling: {model_features}")
print(f"Feature matrix shape: {X_final.shape}")
print(f"Target vector shape: {y_final.shape}")

# Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=None  # For regression problems
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train/test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")

# Final data summary
print("\n=== FINAL DATA SUMMARY ===")
print("✓ Dataset generated: 1000 samples with 8 numerical + 2 categorical features")
print("✓ Missing values handled: Median imputation for numerical, mode for categorical")
print("✓ Feature scaling applied: StandardScaler for numerical features")
print("✓ Categorical encoding: Label encoding applied")
print("✓ Train/test split: 80/20 split completed")
print("✓ Ready for modeling!")

print(f"\nFinal training features shape: {X_train.shape}")
print(f"Final training target shape: {y_train.shape}")
print(f"Final test features shape: {X_test.shape}")
print(f"Final test target shape: {y_test.shape}")