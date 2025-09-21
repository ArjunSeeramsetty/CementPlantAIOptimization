import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Temporal Consistency Diagnostics
def temporal_consistency_analysis(df):
    """Comprehensive temporal consistency analysis for process data"""

    temporal_diagnostics = {
        'temporal_anomalies': [],
        'trend_violations': [],
        'seasonal_inconsistencies': [],
        'rate_of_change_violations': [],
        'temporal_gaps': [],
        'consistency_metrics': {}
    }

    print(f"Analyzing temporal consistency for {len(df)} records...")

    # Create synthetic time index for analysis
    df_temp = df.copy()
    df_temp['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

    # Get numerical columns for analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) >= 3:
        # Analyze temporal patterns for each numerical column
        for col in numerical_cols[:3]:  # Analyze first 3 columns to keep output manageable
            values = df_temp[col].values

            # Rate of change analysis
            rate_of_change = np.diff(values)
            rate_threshold = np.std(rate_of_change) * 3  # 3-sigma rule

            rate_violations = np.abs(rate_of_change) > rate_threshold
            violation_indices = np.where(rate_violations)[0]

            temporal_diagnostics['rate_of_change_violations'].extend([
                {
                    'column': col,
                    'index': int(idx + 1),  # +1 because diff reduces length by 1
                    'rate_change': float(rate_of_change[idx]),
                    'threshold': float(rate_threshold)
                } for idx in violation_indices
            ])

            # Trend consistency analysis
            # Check for sudden trend reversals
            rolling_mean = pd.Series(values).rolling(window=min(10, len(values)//4), min_periods=1).mean()
            trend_changes = np.diff(np.sign(np.diff(rolling_mean.values)))

            # Identify significant trend reversals
            reversal_indices = np.where(np.abs(trend_changes) > 1)[0]

            temporal_diagnostics['trend_violations'].extend([
                {
                    'column': col,
                    'index': int(idx + 2),  # +2 because of double diff
                    'trend_change': float(trend_changes[idx])
                } for idx in reversal_indices
            ])

            print(f"{col}: {len(violation_indices)} rate violations, {len(reversal_indices)} trend reversals")

    # Overall consistency metrics
    if len(numerical_cols) > 0:
        # Calculate overall temporal stability score
        all_values = df[numerical_cols].values.flatten()
        stability_score = 1.0 / (1.0 + np.std(all_values))

        # Temporal correlation analysis
        temporal_correlations = {}
        for i, col in enumerate(numerical_cols[:5]):  # First 5 columns
            # Create lagged version
            lagged_values = np.roll(df[col].values, 1)
            lagged_values[0] = lagged_values[1]  # Fill first value

            correlation = np.corrcoef(df[col].values, lagged_values)[0, 1]
            temporal_correlations[col] = float(correlation)

        temporal_diagnostics['consistency_metrics'] = {
            'stability_score': float(stability_score),
            'total_rate_violations': len(temporal_diagnostics['rate_of_change_violations']),
            'total_trend_violations': len(temporal_diagnostics['trend_violations']),
            'temporal_correlations': temporal_correlations,
            'analysis_window': '1 hour intervals',
            'data_quality_score': float(1.0 - (len(temporal_diagnostics['rate_of_change_violations']) / len(df)))
        }

    return temporal_diagnostics

# Apply temporal analysis to available data
print("=== Starting Temporal Consistency Analysis ===")

if 'scaled_df' in globals():
    temporal_results = temporal_consistency_analysis(scaled_df)

    print(f"\n=== Temporal Analysis Results ===")
    print(f"Total rate violations: {temporal_results['consistency_metrics']['total_rate_violations']}")
    print(f"Total trend violations: {temporal_results['consistency_metrics']['total_trend_violations']}")
    print(f"Data quality score: {temporal_results['consistency_metrics']['data_quality_score']:.4f}")
    print(f"Stability score: {temporal_results['consistency_metrics']['stability_score']:.4f}")

else:
    print("No scaled_df available for temporal analysis")
    temporal_results = {'status': 'no_data_available'}