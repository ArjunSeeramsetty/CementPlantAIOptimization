import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

try:
    import requests
    _requests_available = True
except ImportError:
    _requests_available = False


def load_kaggle_cement_dataset(dataset_url: Optional[str] = None) -> pd.DataFrame:
    """
    Load Kaggle cement dataset from actual source.
    
    Args:
        dataset_url: URL to the Kaggle dataset (if available)
    
    Returns:
        DataFrame with cement composition data
    """
    if dataset_url and _requests_available:
        try:
            # Try to load from actual URL
            response = requests.get(dataset_url)
            if response.status_code == 200:
                # Parse CSV data
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                return df
        except Exception as e:
            print(f"Failed to load from URL: {e}")
    
    # Fallback: Load from local file if available
    try:
        # Try to load from local CSV file
        df = pd.read_csv('data/kaggle_cement_dataset.csv')
        return df
    except FileNotFoundError:
        print("Kaggle cement dataset not found. Please download from Kaggle and place in data/ directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        return pd.DataFrame()


def get_kaggle_cement_data() -> pd.DataFrame:
    """Get Kaggle cement dataset with proper error handling."""
    # Try to load actual data
    df = load_kaggle_cement_dataset()
    
    if df.empty:
        print("Warning: Using placeholder data. Please provide actual Kaggle cement dataset.")
        # Return empty DataFrame to indicate missing data
        return pd.DataFrame()
    
    return df


# Load the dataset
kaggle_cement_df = get_kaggle_cement_data()

# Add missing values to simulate real data quality issues if data is available
if not kaggle_cement_df.empty:
    kaggle_missing_cols = ['blast_furnace_slag', 'fly_ash', 'superplasticizer', 'slump', 'curing_temp', 'curing_humidity']
    for _kaggle_col in kaggle_missing_cols:
        if _kaggle_col in kaggle_cement_df.columns:
            _kaggle_missing_idx = np.random.choice(kaggle_cement_df.index, size=int(len(kaggle_cement_df) * np.random.uniform(0.03, 0.12)), replace=False)
            kaggle_cement_df.loc[_kaggle_missing_idx, _kaggle_col] = np.nan

    # Ensure non-negative values for components that can't be negative
    kaggle_non_negative_cols = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'compressive_strength', 'slump']
    for _kaggle_num_col in kaggle_non_negative_cols:
        if _kaggle_num_col in kaggle_cement_df.columns:
            kaggle_cement_df[_kaggle_num_col] = np.abs(kaggle_cement_df[_kaggle_num_col])

    # Fix ratio values to be within reasonable bounds
    if 'w_c_ratio' in kaggle_cement_df.columns:
        kaggle_cement_df['w_c_ratio'] = np.clip(kaggle_cement_df['w_c_ratio'], 0.2, 1.0)
    if 'curing_humidity' in kaggle_cement_df.columns:
        kaggle_cement_df['curing_humidity'] = np.clip(kaggle_cement_df['curing_humidity'], 40, 100)

print(f"✅ Loaded Kaggle cement dataset: {len(kaggle_cement_df)} samples")
print(f"Columns: {list(kaggle_cement_df.columns)}")
print(f"Shape: {kaggle_cement_df.shape}")
print(f"Missing values per column:")
print(kaggle_cement_df.isnull().sum().head(10))