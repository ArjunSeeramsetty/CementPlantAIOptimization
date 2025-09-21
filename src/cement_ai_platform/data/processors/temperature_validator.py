import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
import warnings

def validate_temperature_profile_consistency(df: pd.DataFrame,
                                           temp_columns: List[str] = None,
                                           time_column: str = None,
                                           temp_tolerance: float = 5.0,
                                           gradient_tolerance: float = 10.0) -> Dict[str, Any]:
    """
    Validate temperature profile consistency for industrial process data

    Args:
        df: Input dataframe
        temp_columns: List of temperature measurement columns
        time_column: Column containing time/sequence information
        temp_tolerance: Allowable temperature deviation (°C)
        gradient_tolerance: Allowable temperature gradient (°C/unit time)

    Returns:
        Dictionary with temperature validation results
    """
    # Auto-detect temperature columns if not provided
    if temp_columns is None:
        temp_keywords = ['temp', 'temperature', 'celsius', 'fahrenheit', 'kelvin', 'curing_temp', 'heat']
        temp_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in temp_keywords
        ) and df[col].dtype in ['float64', 'int64']]

    validation_results = {
        'total_samples': len(df),
        'temperature_columns_checked': temp_columns,
        'profile_violations': 0,
        'gradient_violations': 0,
        'consistency_score': 0.0,
        'temperature_statistics': {},
        'outlier_detection': {},
        'validation_passed': True,
        'issues_found': [],
        'recommendations': []
    }

    if not temp_columns:
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append("No temperature columns found for validation")
        return validation_results

    # Temperature consistency analysis
    temp_data = df[temp_columns].copy()
    temp_data = temp_data.dropna()

    if len(temp_data) == 0:
        validation_results['issues_found'].append("No complete temperature data available")
        return validation_results

    # Statistical analysis for each temperature column
    for temp_col in temp_columns:
        if temp_col not in temp_data.columns:
            continue

        temp_values = temp_data[temp_col]

        # Basic statistics
        temp_stats = {
            'mean': float(temp_values.mean()),
            'std': float(temp_values.std()),
            'min': float(temp_values.min()),
            'max': float(temp_values.max()),
            'range': float(temp_values.max() - temp_values.min()),
            'cv': float(temp_values.std() / temp_values.mean()) if temp_values.mean() != 0 else 0,
            'median': float(temp_values.median())
        }

        # Outlier detection using IQR method
        Q1 = temp_values.quantile(0.25)
        Q3 = temp_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = temp_values[(temp_values < lower_bound) | (temp_values > upper_bound)]
        temp_stats['outlier_count'] = len(outliers)
        temp_stats['outlier_percentage'] = float(len(outliers) / len(temp_values) * 100)

        # Temperature range validation
        if temp_stats['range'] > temp_tolerance * 4:  # Range significantly larger than tolerance
            validation_results['profile_violations'] += 1
            validation_results['issues_found'].append(
                f"Excessive temperature range in {temp_col}: {temp_stats['range']:.1f}°C"
            )

        # Coefficient of variation check
        if temp_stats['cv'] > 0.2:  # High variability
            validation_results['profile_violations'] += 1
            validation_results['issues_found'].append(
                f"High temperature variability in {temp_col}: CV={temp_stats['cv']:.3f}"
            )

        validation_results['temperature_statistics'][temp_col] = temp_stats
        validation_results['outlier_detection'][temp_col] = {
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_indices': outliers.index.tolist()[:10]  # First 10 outlier indices
        }

    # Cross-column temperature consistency (if multiple temperature columns)
    if len(temp_columns) > 1:
        temp_correlations = temp_data[temp_columns].corr()

        # Check correlations between temperature measurements
        correlation_issues = []
        for i, col1 in enumerate(temp_columns):
            for j, col2 in enumerate(temp_columns):
                if i < j and col1 in temp_correlations.columns and col2 in temp_correlations.columns:
                    corr_val = temp_correlations.loc[col1, col2]
                    if not np.isnan(corr_val) and abs(corr_val) < 0.5:  # Low correlation
                        correlation_issues.append(f"{col1} vs {col2}: r={corr_val:.3f}")

        if correlation_issues:
            validation_results['issues_found'].extend([
                f"Low temperature correlation: {issue}" for issue in correlation_issues[:3]
            ])

    # Temporal gradient analysis (if time column available)
    if time_column and time_column in df.columns:
        df_sorted = df.sort_values(time_column)

        for temp_col in temp_columns:
            if temp_col not in df_sorted.columns:
                continue

            temp_series = df_sorted[temp_col].dropna()
            if len(temp_series) < 2:
                continue

            # Calculate temperature gradients
            temp_diff = temp_series.diff()
            gradients = temp_diff.abs()

            # Check for excessive gradients
            excessive_gradients = gradients > gradient_tolerance
            gradient_violations = excessive_gradients.sum()

            if gradient_violations > 0:
                validation_results['gradient_violations'] += int(gradient_violations)
                validation_results['issues_found'].append(
                    f"Excessive temperature gradients in {temp_col}: {gradient_violations} violations"
                )

    # Calculate overall consistency score
    total_violations = validation_results['profile_violations'] + validation_results['gradient_violations']
    if len(temp_data) == 0:
        validation_results['consistency_score'] = 0.0
    else:
        violation_rate = total_violations / len(temp_data)
        validation_results['consistency_score'] = max(0.0, 1.0 - violation_rate)

    # Overall validation
    if total_violations > len(temp_data) * 0.1:  # More than 10% violations
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"High temperature inconsistency rate: {total_violations} violations"
        )

    # Generate recommendations
    if validation_results['profile_violations'] > 0:
        validation_results['recommendations'].append("Investigate temperature sensor calibration")
        validation_results['recommendations'].append("Review process control parameters")

    if validation_results['gradient_violations'] > 0:
        validation_results['recommendations'].append("Check for temperature control system issues")

    return validation_results

# Test with available data
# Check if we have actual temperature data
synthetic_df = pd.DataFrame({'time': range(100)})
temp_data_available = False
if 'kaggle_cement_df' in locals() and 'curing_temp' in kaggle_cement_df.columns:
    temp_data_available = True
    temp_validation = validate_temperature_profile_consistency(
        kaggle_cement_df,
        temp_columns=['curing_temp'],
        temp_tolerance=5.0,
        gradient_tolerance=10.0
    )
else:
    # Simulate temperature data for testing
    synthetic_temp_df = synthetic_df.copy()
    # Add simulated temperature columns
    synthetic_temp_df['process_temp'] = 85 + np.random.normal(0, 3, len(synthetic_df))  # Process temp around 85°C ± 3°C
    synthetic_temp_df['curing_temp'] = 23 + np.random.normal(0, 2, len(synthetic_df))  # Curing temp around 23°C ± 2°C
    synthetic_temp_df['kiln_temp'] = 1450 + np.random.normal(0, 50, len(synthetic_df))  # Kiln temp around 1450°C ± 50°C

    # Add some outliers
    outlier_indices = np.random.choice(len(synthetic_temp_df), size=20, replace=False)
    synthetic_temp_df.loc[outlier_indices, 'process_temp'] += np.random.choice([-25, 25], size=20)

    temp_validation = validate_temperature_profile_consistency(
        synthetic_temp_df,
        temp_columns=['process_temp', 'curing_temp', 'kiln_temp'],
        temp_tolerance=5.0,
        gradient_tolerance=10.0
    )

print("🌡️  TEMPERATURE PROFILE VALIDATION RESULTS")
print("=" * 50)
print(f"Samples Validated: {temp_validation['total_samples']:,}")
print(f"Temperature Columns: {len(temp_validation['temperature_columns_checked'])}")
print(f"Profile Violations: {temp_validation['profile_violations']}")
print(f"Gradient Violations: {temp_validation['gradient_violations']}")
print(f"Consistency Score: {temp_validation['consistency_score']:.3f}")
print(f"Validation Passed: {'✅' if temp_validation['validation_passed'] else '❌'}")

if temp_validation['temperature_statistics']:
    print(f"\nTemperature Statistics:")
    for temp_col, stats in list(temp_validation['temperature_statistics'].items())[:3]:
        print(f"  {temp_col}:")
        print(f"    Mean: {stats['mean']:.1f}°C (±{stats['std']:.1f})")
        print(f"    Range: {stats['min']:.1f}°C to {stats['max']:.1f}°C")
        print(f"    Outliers: {stats['outlier_count']} ({stats['outlier_percentage']:.1f}%)")

if temp_validation['issues_found']:
    print(f"\n⚠️  Issues Found ({len(temp_validation['issues_found'])}):")
    for issue in temp_validation['issues_found'][:3]:
        print(f"  • {issue}")

temperature_profile_results = temp_validation