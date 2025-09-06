import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create time-indexed dataset for temporal validation
np.random.seed(42)
n_samples = len(X_final)

# Generate synthetic timestamps spanning 2 years
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)

# Add temporal patterns to make the data more realistic
time_features_df = pd.DataFrame({
    'timestamp': date_range,
    'month': date_range.month,
    'quarter': date_range.quarter,
    'day_of_year': date_range.dayofyear,
    'is_holiday': np.random.binomial(1, 0.05, n_samples),  # 5% holiday effect
    'seasonal_trend': np.sin(2 * np.pi * date_range.dayofyear / 365.25),
    'long_term_trend': np.linspace(0, 0.1, n_samples)  # Gradual trend
})

# Combine with existing features
temporal_df = pd.concat([
    X_final.reset_index(drop=True),
    time_features_df,
    y_final.reset_index(drop=True)
], axis=1)

# Sort by timestamp for temporal validation
temporal_df = temporal_df.sort_values('timestamp').reset_index(drop=True)

print("Created temporal dataset:")
print(f"Date range: {temporal_df['timestamp'].min()} to {temporal_df['timestamp'].max()}")
print(f"Total samples: {len(temporal_df)}")
print(f"Features with temporal components: {temporal_df.shape[1]-1}")