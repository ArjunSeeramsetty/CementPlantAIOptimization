import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings

def validate_energy_consumption_alignment(df: pd.DataFrame,
                                        energy_columns: List[str] = None,
                                        benchmark_values: Dict[str, float] = None,
                                        tolerance_percentage: float = 15.0) -> Dict[str, Any]:
    """
    Validate energy consumption alignment with industry benchmarks

    Args:
        df: Input dataframe
        energy_columns: List of energy consumption columns
        benchmark_values: Dictionary of benchmark values for energy consumption
        tolerance_percentage: Allowable deviation from benchmarks (%)

    Returns:
        Dictionary with energy validation results
    """
    # Auto-detect energy columns if not provided
    if energy_columns is None:
        energy_keywords = ['energy', 'power', 'consumption', 'kwh', 'kw', 'watt', 'electricity',
                          'fuel', 'gas', 'thermal', 'electrical', 'heat_consumption', 'steam']
        energy_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in energy_keywords
        ) and df[col].dtype in ['float64', 'int64']]

    # Default benchmark values for industrial processes (per unit/kg/batch)
    if benchmark_values is None:
        benchmark_values = {
            'electrical_energy': 100.0,  # kWh per ton
            'thermal_energy': 800.0,     # MJ per ton
            'total_energy': 1000.0,      # kWh equivalent per ton
            'specific_energy': 3.5,      # GJ per ton cement
            'fuel_consumption': 750.0,   # kcal per kg
            'steam_consumption': 150.0   # kg steam per ton
        }

    validation_results = {
        'total_samples': len(df),
        'energy_columns_checked': energy_columns,
        'benchmark_violations': 0,
        'efficiency_score': 0.0,
        'energy_statistics': {},
        'benchmark_alignment': {},
        'consumption_patterns': {},
        'validation_passed': True,
        'issues_found': [],
        'recommendations': []
    }

    if not energy_columns:
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append("No energy consumption columns found for validation")
        return validation_results

    # Energy consumption analysis
    energy_data = df[energy_columns].copy()
    energy_data = energy_data.dropna()

    if len(energy_data) == 0:
        validation_results['issues_found'].append("No complete energy consumption data available")
        return validation_results

    total_violations = 0

    # Analyze each energy column
    for energy_col in energy_columns:
        if energy_col not in energy_data.columns:
            continue

        energy_values = energy_data[energy_col]

        # Basic statistics
        energy_stats = {
            'mean': float(energy_values.mean()),
            'std': float(energy_values.std()),
            'median': float(energy_values.median()),
            'min': float(energy_values.min()),
            'max': float(energy_values.max()),
            'percentile_25': float(energy_values.quantile(0.25)),
            'percentile_75': float(energy_values.quantile(0.75)),
            'coefficient_of_variation': float(energy_values.std() / energy_values.mean()) if energy_values.mean() != 0 else 0
        }

        # Find matching benchmark
        benchmark_key = None
        benchmark_value = None

        for key, value in benchmark_values.items():
            if key.lower() in energy_col.lower() or energy_col.lower() in key.lower():
                benchmark_key = key
                benchmark_value = value
                break

        # Use a generic benchmark if no specific match
        if benchmark_value is None:
            benchmark_value = benchmark_values.get('total_energy', 1000.0)
            benchmark_key = 'generic_energy'

        # Calculate deviations from benchmark
        deviations = np.abs(energy_values - benchmark_value) / benchmark_value * 100
        violations = deviations > tolerance_percentage
        violation_count = violations.sum()

        total_violations += violation_count

        # Efficiency analysis
        efficiency_ratio = energy_values.mean() / benchmark_value
        efficiency_category = 'excellent' if efficiency_ratio < 0.9 else \
                             'good' if efficiency_ratio < 1.1 else \
                             'acceptable' if efficiency_ratio < 1.3 else 'poor'

        energy_stats.update({
            'benchmark_value': float(benchmark_value),
            'benchmark_key': benchmark_key,
            'mean_deviation_percent': float(deviations.mean()),
            'violation_count': int(violation_count),
            'violation_rate': float(violation_count / len(energy_values)),
            'efficiency_ratio': float(efficiency_ratio),
            'efficiency_category': efficiency_category
        })

        validation_results['energy_statistics'][energy_col] = energy_stats

        # Record benchmark alignment
        validation_results['benchmark_alignment'][energy_col] = {
            'aligned': violation_count < len(energy_values) * 0.1,  # Less than 10% violations
            'benchmark_value': float(benchmark_value),
            'actual_mean': float(energy_values.mean()),
            'deviation_percent': float(abs(energy_values.mean() - benchmark_value) / benchmark_value * 100),
            'efficiency_rating': efficiency_category
        }

        # Identify issues
        if violation_count > len(energy_values) * 0.2:  # More than 20% violations
            validation_results['issues_found'].append(
                f"High energy deviation in {energy_col}: {violation_count} violations ({violation_count/len(energy_values)*100:.1f}%)"
            )

        if efficiency_ratio > 1.5:  # 50% higher than benchmark
            validation_results['issues_found'].append(
                f"Poor energy efficiency in {energy_col}: {efficiency_ratio:.2f}x benchmark"
            )

        if energy_stats['coefficient_of_variation'] > 0.3:  # High variability
            validation_results['issues_found'].append(
                f"High energy consumption variability in {energy_col}: CV={energy_stats['coefficient_of_variation']:.3f}"
            )

    validation_results['benchmark_violations'] = int(total_violations)

    # Calculate overall efficiency score
    if validation_results['energy_statistics']:
        efficiency_ratios = [stats['efficiency_ratio'] for stats in validation_results['energy_statistics'].values()]
        avg_efficiency_ratio = np.mean(efficiency_ratios)
        # Score: 1.0 = perfect alignment, decreases with deviation
        validation_results['efficiency_score'] = max(0.0, 2.0 - avg_efficiency_ratio)

    # Consumption pattern analysis
    if len(energy_columns) > 1:
        # Analyze correlations between different energy types
        energy_corr = energy_data[energy_columns].corr()

        pattern_analysis = {
            'energy_correlation_matrix': energy_corr.to_dict(),
            'total_energy_estimate': float(energy_data[energy_columns].sum(axis=1).mean())
        }

        # Check for unusual patterns
        if len(energy_columns) >= 2:
            main_correlations = []
            for i, col1 in enumerate(energy_columns):
                for j, col2 in enumerate(energy_columns):
                    if i < j:
                        corr_val = energy_corr.loc[col1, col2]
                        if not np.isnan(corr_val):
                            main_correlations.append(abs(corr_val))

            if main_correlations:
                avg_correlation = np.mean(main_correlations)
                pattern_analysis['average_energy_correlation'] = float(avg_correlation)

                if avg_correlation < 0.3:
                    validation_results['issues_found'].append(
                        f"Low correlation between energy types: avg={avg_correlation:.3f}"
                    )

        validation_results['consumption_patterns'] = pattern_analysis

    # Overall validation
    violation_rate = total_violations / (len(energy_data) * len(energy_columns)) if energy_columns else 0
    if violation_rate > 0.15:  # More than 15% overall violations
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"High overall energy benchmark violation rate: {violation_rate:.2%}"
        )

    # Generate recommendations
    if validation_results['efficiency_score'] < 0.7:
        validation_results['recommendations'].append("Investigate energy efficiency improvement opportunities")
        validation_results['recommendations'].append("Consider equipment optimization or replacement")

    if total_violations > len(energy_data) * 0.1:
        validation_results['recommendations'].append("Review energy measurement accuracy and calibration")
        validation_results['recommendations'].append("Benchmark against similar industrial processes")

    return validation_results

# Test with simulated energy consumption data
# Create synthetic energy data that represents industrial energy consumption
synthetic_energy_df = synthetic_df.copy()

# Add energy consumption columns with realistic industrial values
np.random.seed(456)  # Different seed for energy data
synthetic_energy_df['electrical_energy'] = np.random.normal(95, 15, len(synthetic_df))  # kWh per ton
synthetic_energy_df['thermal_energy'] = np.random.normal(820, 80, len(synthetic_df))   # MJ per ton
synthetic_energy_df['fuel_consumption'] = np.random.normal(760, 60, len(synthetic_df)) # kcal per kg

# Add some outliers and inefficient operations
outlier_indices_energy = np.random.choice(len(synthetic_energy_df), size=30, replace=False)
synthetic_energy_df.loc[outlier_indices_energy, 'electrical_energy'] *= np.random.uniform(1.3, 1.8, size=30)
synthetic_energy_df.loc[outlier_indices_energy[:15], 'thermal_energy'] *= np.random.uniform(0.6, 0.8, size=15)  # Some very efficient
synthetic_energy_df.loc[outlier_indices_energy[15:], 'thermal_energy'] *= np.random.uniform(1.4, 1.7, size=15)  # Some very inefficient

# Ensure no negative values
for col in ['electrical_energy', 'thermal_energy', 'fuel_consumption']:
    synthetic_energy_df[col] = np.abs(synthetic_energy_df[col])

energy_validation = validate_energy_consumption_alignment(
    synthetic_energy_df,
    energy_columns=['electrical_energy', 'thermal_energy', 'fuel_consumption'],
    tolerance_percentage=15.0
)

print("⚡ ENERGY CONSUMPTION VALIDATION RESULTS")
print("=" * 50)
print(f"Samples Validated: {energy_validation['total_samples']:,}")
print(f"Energy Columns: {len(energy_validation['energy_columns_checked'])}")
print(f"Benchmark Violations: {energy_validation['benchmark_violations']:,}")
print(f"Efficiency Score: {energy_validation['efficiency_score']:.3f}")
print(f"Validation Passed: {'✅' if energy_validation['validation_passed'] else '❌'}")

if energy_validation['energy_statistics']:
    print(f"\nEnergy Statistics:")
    for energy_col, stats in list(energy_validation['energy_statistics'].items())[:3]:
        print(f"  {energy_col}:")
        print(f"    Mean: {stats['mean']:.1f} (Benchmark: {stats['benchmark_value']:.1f})")
        print(f"    Efficiency: {stats['efficiency_category']} ({stats['efficiency_ratio']:.2f}x)")
        print(f"    Violations: {stats['violation_count']} ({stats['violation_rate']:.1%})")

if energy_validation['issues_found']:
    print(f"\n⚠️  Issues Found ({len(energy_validation['issues_found'])}):")
    for issue in energy_validation['issues_found'][:3]:
        print(f"  • {issue}")

energy_consumption_results = energy_validation