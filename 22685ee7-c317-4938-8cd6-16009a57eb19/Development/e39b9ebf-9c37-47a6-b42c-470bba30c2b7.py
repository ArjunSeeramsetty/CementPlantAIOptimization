import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    noise=10,
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i}' for i in range(1, 9)]

# Create DataFrame
synthetic_df = pd.DataFrame(X, columns=feature_names)
synthetic_df['target'] = y

# Add some categorical features
synthetic_df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=1000)
synthetic_df['category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=1000)

print(f"Generated synthetic dataset with {len(synthetic_df)} rows and {len(synthetic_df.columns)} columns")
print(f"Features: {list(synthetic_df.columns[:-1])}")
print(f"Target: {synthetic_df.columns[-1]}")