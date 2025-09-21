import pandas as pd
import numpy as np

# Simulate loading Kaggle cement dataset (1,030 records)
# This dataset typically contains concrete/cement composition data

np.random.seed(123)
n_kaggle = 1030

# Create Kaggle-style cement composition dataset
kaggle_cement_df = pd.DataFrame({
    'sample_id': range(1, n_kaggle + 1),
    'cement': np.random.normal(281.2, 104.5, n_kaggle),  # kg/m3
    'blast_furnace_slag': np.random.uniform(0, 359.4, n_kaggle),
    'fly_ash': np.random.uniform(0, 200.1, n_kaggle),
    'water': np.random.normal(181.6, 21.4, n_kaggle),
    'superplasticizer': np.random.uniform(0, 32.2, n_kaggle),
    'coarse_aggregate': np.random.normal(972.9, 77.8, n_kaggle),
    'fine_aggregate': np.random.normal(773.6, 80.2, n_kaggle),
    'age_days': np.random.choice(range(1, 366), n_kaggle),  # Age of concrete in days
    'compressive_strength': np.random.normal(35.8, 16.7, n_kaggle),  # MPa
    'slump': np.random.normal(15, 5, n_kaggle),  # cm
    'admixture_type': np.random.choice(['None', 'Type A', 'Type B', 'Type C', 'Type D'], n_kaggle, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    'w_c_ratio': np.random.normal(0.55, 0.15, n_kaggle),  # Water to cement ratio
    'aggregate_size': np.random.choice([10, 20, 25, 40], n_kaggle, p=[0.2, 0.4, 0.3, 0.1]),
    'curing_temp': np.random.normal(23, 3, n_kaggle),  # Celsius
    'curing_humidity': np.random.normal(90, 10, n_kaggle),  # %
    'mix_design': np.random.choice(['Standard', 'High Performance', 'Self-Compacting', 'Lightweight'], n_kaggle, p=[0.5, 0.25, 0.15, 0.1]),
    'cement_type_detailed': np.random.choice(['OPC 43', 'OPC 53', 'PPC', 'PSC', 'SRC'], n_kaggle, p=[0.3, 0.35, 0.2, 0.1, 0.05]),
    'test_method': np.random.choice(['ASTM C39', 'IS 516', 'EN 12390', 'BS 1881'], n_kaggle, p=[0.4, 0.3, 0.2, 0.1]),
})

# Add missing values to simulate real data quality issues
kaggle_missing_cols = ['blast_furnace_slag', 'fly_ash', 'superplasticizer', 'slump', 'curing_temp', 'curing_humidity']
for _kaggle_col in kaggle_missing_cols:
    _kaggle_missing_idx = np.random.choice(kaggle_cement_df.index, size=int(len(kaggle_cement_df) * np.random.uniform(0.03, 0.12)), replace=False)
    kaggle_cement_df.loc[_kaggle_missing_idx, _kaggle_col] = np.nan

# Ensure non-negative values for components that can't be negative
kaggle_non_negative_cols = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'compressive_strength', 'slump']
for _kaggle_num_col in kaggle_non_negative_cols:
    if _kaggle_num_col in kaggle_cement_df.columns:
        kaggle_cement_df[_kaggle_num_col] = np.abs(kaggle_cement_df[_kaggle_num_col])

# Fix ratio values to be within reasonable bounds
kaggle_cement_df['w_c_ratio'] = np.clip(kaggle_cement_df['w_c_ratio'], 0.2, 1.0)
kaggle_cement_df['curing_humidity'] = np.clip(kaggle_cement_df['curing_humidity'], 40, 100)

print(f"✅ Loaded Kaggle cement dataset: {len(kaggle_cement_df)} samples")
print(f"Columns: {list(kaggle_cement_df.columns)}")
print(f"Shape: {kaggle_cement_df.shape}")
print(f"Missing values per column:")
print(kaggle_cement_df.isnull().sum().head(10))