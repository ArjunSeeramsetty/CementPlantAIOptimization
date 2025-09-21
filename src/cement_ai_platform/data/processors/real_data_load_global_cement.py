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


def load_global_cement_database(api_url: Optional[str] = None) -> pd.DataFrame:
    """
    Load Global Cement Database data from actual source.
    
    Args:
        api_url: URL to the Global Cement Database API (if available)
    
    Returns:
        DataFrame with global cement facility data
    """
    if api_url and _requests_available:
        try:
            # Try to load from actual API
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                return df
        except Exception as e:
            print(f"Failed to load from API: {e}")
    
    # Fallback: Load from local file if available
    try:
        # Try to load from local CSV file
        df = pd.read_csv('data/global_cement_database.csv')
        return df
    except FileNotFoundError:
        print("Global Cement Database not found. Please download and place in data/ directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading local database: {e}")
        return pd.DataFrame()


def get_global_cement_data() -> pd.DataFrame:
    """Get Global Cement Database with proper error handling."""
    # Try to load actual data
    df = load_global_cement_database()
    
    if df.empty:
        print("Warning: Using placeholder data. Please provide actual Global Cement Database.")
        # Return empty DataFrame to indicate missing data
        return pd.DataFrame()
    
    return df


# Load the dataset
global_cement_df = get_global_cement_data()

# Add some missing values to simulate real-world data issues if data is available
if not global_cement_df.empty:
    global_missing_cols = ['energy_efficiency', 'co2_emissions', 'water_usage', 'electricity_kwh_t', 'dust_emissions', 'nox_emissions']
    for _global_col in global_missing_cols:
        if _global_col in global_cement_df.columns:
            _global_missing_idx = np.random.choice(global_cement_df.index, size=int(len(global_cement_df) * np.random.uniform(0.05, 0.15)), replace=False)
            global_cement_df.loc[_global_missing_idx, _global_col] = np.nan

    # Fix negative values that shouldn't exist
    global_numeric_cols = ['capacity_mt_year', 'energy_efficiency', 'co2_emissions', 'limestone_pct', 'clay_pct', 'iron_ore_pct', 'silica_sand_pct', 'gypsum_pct', 'water_usage', 'electricity_kwh_t', 'dust_emissions', 'nox_emissions']
    for _global_num_col in global_numeric_cols:
        if _global_num_col in global_cement_df.columns:
            global_cement_df[_global_num_col] = np.abs(global_cement_df[_global_num_col])

if not global_cement_df.empty:
    print(f"✅ Loaded Global Cement Database: {len(global_cement_df)} facilities")
    print(f"Columns: {list(global_cement_df.columns)}")
    print(f"Shape: {global_cement_df.shape}")
    print(f"Missing values per column:")
    print(global_cement_df.isnull().sum().head(10))
else:
    print("⚠️ Global Cement Database is empty - no data loaded")