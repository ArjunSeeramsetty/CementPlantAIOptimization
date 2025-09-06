import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings

def validate_temporal_data_integrity(df: pd.DataFrame,
                                   timestamp_column: str = None,
                                   value_columns: List[str] = None,
                                   expected_interval: str = 'auto',
                                   max_gap_tolerance: float = 2.0) -> Dict[str, Any]:
    """
    Validate temporal data integrity for time-series industrial data
    
    Args:
        df: Input dataframe
        timestamp_column: Column containing timestamps
        value_columns: List of value columns to validate for temporal consistency
        expected_interval: Expected time interval ('auto', 'minutes', 'hours', 'days')
        max_gap_tolerance: Maximum acceptable gap as multiple of expected interval
    
    Returns:
        Dictionary with temporal validation results
    """
    # Auto-detect timestamp column if not provided
    if timestamp_column is None:
        timestamp_candidates = [col for col in df.columns if any(
            keyword in col.lower() for keyword in [
                'time', 'date', 'timestamp', 'created', 'updated', 'recorded',
                'sample_time', 'measurement_time', 'batch_time'
            ]
        )]
        
        # Also check for datetime columns
        datetime_columns = [col for col in df.columns if 
                          pd.api.types.is_datetime64_any_dtype(df[col])]
        
        timestamp_candidates.extend(datetime_columns)
        
        if timestamp_candidates:
            timestamp_column = timestamp_candidates[0]
    
    # Auto-detect value columns if not provided
    if value_columns is None:
        value_columns = [col for col in df.columns if 
                        df[col].dtype in ['float64', 'int64'] and 
                        col != timestamp_column]
    
    validation_results = {
        'total_samples': len(df),
        'timestamp_column': timestamp_column,
        'value_columns_checked': value_columns,
        'temporal_gaps': 0,
        'duplicate_timestamps': 0,
        'out_of_sequence': 0,
        'data_continuity_score': 0.0,
        'temporal_statistics': {},
        'gap_analysis': {},
        'validation_passed': True,
        'issues_found': [],
        'recommendations': []
    }
    
    if not timestamp_column or timestamp_column not in df.columns:
        # Create synthetic timestamp data for validation
        validation_results['timestamp_column'] = 'synthetic_time'
        df_temp = df.copy()
        
        # Create synthetic timestamps (hourly intervals)
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        df_temp['synthetic_time'] = [base_time + timedelta(hours=i) for i in range(len(df_temp))]
        
        # Add some gaps and duplicates for testing
        gap_indices = np.random.choice(len(df_temp), size=min(10, len(df_temp)//10), replace=False)
        for idx in gap_indices:
            if idx < len(df_temp) - 1:
                # Convert numpy.int64 to int
                gap_hours = int(np.random.choice([2, 3, 5]))
                df_temp.loc[idx, 'synthetic_time'] = df_temp.loc[idx, 'synthetic_time'] + timedelta(hours=gap_hours)
        
        # Add some duplicates
        dup_indices = np.random.choice(len(df_temp), size=min(5, len(df_temp)//20), replace=False)
        for idx in dup_indices:
            if idx > 0:
                df_temp.loc[idx, 'synthetic_time'] = df_temp.loc[idx-1, 'synthetic_time']
        
        timestamp_column = 'synthetic_time'
        
        validation_results['issues_found'].append("No timestamp column found - using synthetic timestamps for validation")
    else:
        df_temp = df.copy()
        
        # Convert timestamp column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_temp[timestamp_column]):
            try:
                df_temp[timestamp_column] = pd.to_datetime(df_temp[timestamp_column])
            except:
                validation_results['validation_passed'] = False
                validation_results['issues_found'].append(f"Cannot parse timestamp column '{timestamp_column}' as datetime")
                return validation_results
    
    # Sort by timestamp for analysis
    df_temp = df_temp.sort_values(timestamp_column)
    timestamps = df_temp[timestamp_column]
    
    if len(timestamps) < 2:
        validation_results['issues_found'].append("Insufficient temporal data for analysis")
        return validation_results
    
    # Basic temporal statistics
    time_diffs = timestamps.diff().dropna()
    total_duration = timestamps.max() - timestamps.min()
    
    temporal_stats = {
        'total_duration_hours': float(total_duration.total_seconds() / 3600),
        'median_interval_minutes': float(time_diffs.median().total_seconds() / 60),
        'mean_interval_minutes': float(time_diffs.mean().total_seconds() / 60),
        'min_interval_seconds': float(time_diffs.min().total_seconds()),
        'max_interval_hours': float(time_diffs.max().total_seconds() / 3600),
        'unique_timestamps': len(timestamps.unique()),
        'total_timestamps': len(timestamps)
    }
    
    # Detect expected interval
    if expected_interval == 'auto':
        median_minutes = temporal_stats['median_interval_minutes']
        if median_minutes < 2:
            expected_interval = 'minutes'
            expected_seconds = 60
        elif median_minutes < 120:
            expected_interval = 'hours'
            expected_seconds = 3600
        else:
            expected_interval = 'days'
            expected_seconds = 86400
    else:
        interval_map = {'minutes': 60, 'hours': 3600, 'days': 86400}
        expected_seconds = interval_map.get(expected_interval, 3600)
    
    temporal_stats['detected_interval'] = expected_interval
    temporal_stats['expected_interval_seconds'] = expected_seconds
    
    # Gap analysis
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=expected_seconds * max_gap_tolerance)]
    validation_results['temporal_gaps'] = len(large_gaps)
    
    gap_analysis = {
        'large_gaps_count': len(large_gaps),
        'max_gap_hours': float(large_gaps.max().total_seconds() / 3600) if len(large_gaps) > 0 else 0.0,
        'total_gap_time_hours': float(large_gaps.sum().total_seconds() / 3600) if len(large_gaps) > 0 else 0.0,
        'gap_locations': large_gaps.index.tolist()[:10]  # First 10 gap locations
    }
    
    # Duplicate timestamps
    duplicate_count = len(timestamps) - len(timestamps.unique())
    validation_results['duplicate_timestamps'] = duplicate_count
    
    # Out of sequence detection (after sorting, this should be minimal)
    original_order = df[timestamp_column] if timestamp_column in df.columns else df_temp[timestamp_column]
    out_of_sequence_count = 0
    if len(original_order) > 1:
        for i in range(1, len(original_order)):
            if original_order.iloc[i] < original_order.iloc[i-1]:
                out_of_sequence_count += 1
    
    validation_results['out_of_sequence'] = out_of_sequence_count
    
    # Data continuity analysis for value columns
    continuity_issues = {}
    if value_columns:
        for col in value_columns:
            if col not in df_temp.columns:
                continue
                
            col_data = df_temp[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for sudden changes (potential sensor issues)
            col_diffs = col_data.diff().abs()
            col_std = col_data.std()
            
            if col_std > 0:
                # Values that change more than 3 standard deviations
                sudden_changes = col_diffs > (3 * col_std)
                sudden_change_count = sudden_changes.sum()
                
                continuity_issues[col] = {
                    'sudden_changes': int(sudden_change_count),
                    'max_change': float(col_diffs.max()),
                    'std_dev': float(col_std),
                    'data_points': len(col_data)
                }
    
    # Calculate data continuity score
    total_issues = (validation_results['temporal_gaps'] + 
                   validation_results['duplicate_timestamps'] + 
                   validation_results['out_of_sequence'])
    
    if len(timestamps) == 0:
        validation_results['data_continuity_score'] = 0.0
    else:
        issue_rate = total_issues / len(timestamps)
        validation_results['data_continuity_score'] = max(0.0, 1.0 - issue_rate)
    
    # Store results
    validation_results['temporal_statistics'] = temporal_stats
    validation_results['gap_analysis'] = gap_analysis
    validation_results['continuity_issues'] = continuity_issues
    
    # Validation checks
    if validation_results['temporal_gaps'] > len(timestamps) * 0.05:  # More than 5% gaps
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"High temporal gap rate: {validation_results['temporal_gaps']} gaps"
        )
    
    if validation_results['duplicate_timestamps'] > len(timestamps) * 0.02:  # More than 2% duplicates
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"High duplicate timestamp rate: {validation_results['duplicate_timestamps']} duplicates"
        )
    
    if validation_results['out_of_sequence'] > 0:
        validation_results['issues_found'].append(
            f"Data out of sequence: {validation_results['out_of_sequence']} instances"
        )
    
    # Generate recommendations
    if validation_results['temporal_gaps'] > 0:
        validation_results['recommendations'].append("Review data collection system for timing gaps")
        validation_results['recommendations'].append("Implement gap-filling strategies for missing time periods")
    
    if validation_results['duplicate_timestamps'] > 0:
        validation_results['recommendations'].append("Check for duplicate data entries in source systems")
        validation_results['recommendations'].append("Implement timestamp deduplication in data pipeline")
    
    if validation_results['data_continuity_score'] < 0.8:
        validation_results['recommendations'].append("Improve data collection reliability and timing")
    
    return validation_results

# Test with synthetic temporal data
# Create synthetic time-series data for validation
synthetic_temporal_df = synthetic_df.copy()

# Add synthetic timestamp column
base_time = datetime(2024, 1, 1, 0, 0, 0)
synthetic_temporal_df['measurement_time'] = [base_time + timedelta(hours=i) for i in range(len(synthetic_df))]

# Add some temporal issues for testing
# 1. Create gaps
gap_indices = np.random.choice(len(synthetic_temporal_df), size=15, replace=False)
for idx in gap_indices:
    if idx < len(synthetic_temporal_df) - 1:
        # Convert numpy.int64 to int to fix the timedelta issue
        gap_size = int(np.random.choice([2, 4, 8, 12]))  # hours
        synthetic_temporal_df.loc[idx, 'measurement_time'] = (
            synthetic_temporal_df.loc[idx, 'measurement_time'] + timedelta(hours=gap_size)
        )

# 2. Create duplicates
dup_indices = np.random.choice(len(synthetic_temporal_df), size=8, replace=False)
for idx in dup_indices:
    if idx > 0:
        synthetic_temporal_df.loc[idx, 'measurement_time'] = synthetic_temporal_df.loc[idx-1, 'measurement_time']

# 3. Shuffle some timestamps to create out-of-sequence issues
shuffle_indices = np.random.choice(len(synthetic_temporal_df), size=5, replace=False)
for i in range(0, len(shuffle_indices), 2):
    if i+1 < len(shuffle_indices):
        idx1, idx2 = shuffle_indices[i], shuffle_indices[i+1]
        # Swap timestamps
        temp_time = synthetic_temporal_df.loc[idx1, 'measurement_time']
        synthetic_temporal_df.loc[idx1, 'measurement_time'] = synthetic_temporal_df.loc[idx2, 'measurement_time']
        synthetic_temporal_df.loc[idx2, 'measurement_time'] = temp_time

temporal_validation = validate_temporal_data_integrity(
    synthetic_temporal_df,
    timestamp_column='measurement_time',
    value_columns=['target', 'feature_1', 'feature_2'],
    expected_interval='hours',
    max_gap_tolerance=2.0
)

print("📅 TEMPORAL DATA INTEGRITY VALIDATION RESULTS")
print("=" * 50)
print(f"Samples Validated: {temporal_validation['total_samples']:,}")
print(f"Timestamp Column: {temporal_validation['timestamp_column']}")
print(f"Temporal Gaps: {temporal_validation['temporal_gaps']}")
print(f"Duplicate Timestamps: {temporal_validation['duplicate_timestamps']}")
print(f"Out of Sequence: {temporal_validation['out_of_sequence']}")
print(f"Data Continuity Score: {temporal_validation['data_continuity_score']:.3f}")
print(f"Validation Passed: {'✅' if temporal_validation['validation_passed'] else '❌'}")

if temporal_validation['temporal_statistics']:
    stats = temporal_validation['temporal_statistics']
    print(f"\nTemporal Statistics:")
    print(f"  Total Duration: {stats['total_duration_hours']:.1f} hours")
    print(f"  Median Interval: {stats['median_interval_minutes']:.1f} minutes")
    print(f"  Detected Interval: {stats['detected_interval']}")
    print(f"  Unique Timestamps: {stats['unique_timestamps']:,}/{stats['total_timestamps']:,}")

if temporal_validation['gap_analysis']:
    gap_stats = temporal_validation['gap_analysis']
    print(f"\nGap Analysis:")
    print(f"  Large Gaps: {gap_stats['large_gaps_count']}")
    if gap_stats['max_gap_hours'] > 0:
        print(f"  Max Gap: {gap_stats['max_gap_hours']:.1f} hours")
        print(f"  Total Gap Time: {gap_stats['total_gap_time_hours']:.1f} hours")

if temporal_validation['issues_found']:
    print(f"\n⚠️  Issues Found ({len(temporal_validation['issues_found'])}):")
    for issue in temporal_validation['issues_found'][:3]:
        print(f"  • {issue}")

temporal_integrity_results = temporal_validation