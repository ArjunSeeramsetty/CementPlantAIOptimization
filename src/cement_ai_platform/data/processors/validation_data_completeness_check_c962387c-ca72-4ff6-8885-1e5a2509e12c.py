import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings

# Data completeness validation framework
def validate_data_completeness(df: pd.DataFrame,
                             required_columns: List[str] = None,
                             completeness_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Comprehensive data completeness validation

    Args:
        df: Input dataframe
        required_columns: List of columns that must be present
        completeness_threshold: Minimum required completeness ratio

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'overall_completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
        'column_completeness': {},
        'missing_patterns': {},
        'data_quality_score': 0.0,
        'validation_passed': True,
        'issues_found': [],
        'recommendations': []
    }

    # Column-wise completeness analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        completeness_ratio = (len(df) - missing_count) / len(df)

        validation_results['column_completeness'][col] = {
            'missing_count': int(missing_count),
            'completeness_ratio': completeness_ratio,
            'data_type': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'passed_threshold': completeness_ratio >= completeness_threshold
        }

        if completeness_ratio < completeness_threshold:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append(
                f"Column '{col}' has {completeness_ratio:.2%} completeness (below {completeness_threshold:.2%} threshold)"
            )

    # Pattern analysis for missing data
    missing_pattern = df.isnull()
    if missing_pattern.any().any():
        # Find rows with multiple missing values
        rows_with_missing = missing_pattern.sum(axis=1)
        validation_results['missing_patterns'] = {
            'rows_with_no_missing': int((rows_with_missing == 0).sum()),
            'rows_with_1_missing': int((rows_with_missing == 1).sum()),
            'rows_with_2_missing': int((rows_with_missing == 2).sum()),
            'rows_with_3_plus_missing': int((rows_with_missing >= 3).sum()),
            'max_missing_per_row': int(rows_with_missing.max()),
            'avg_missing_per_row': float(rows_with_missing.mean())
        }

    # Required columns check
    if required_columns:
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            validation_results['validation_passed'] = False
            validation_results['issues_found'].append(
                f"Missing required columns: {missing_required}"
            )

    # Calculate overall data quality score
    completeness_scores = [info['completeness_ratio'] for info in validation_results['column_completeness'].values()]
    validation_results['data_quality_score'] = np.mean(completeness_scores)

    # Generate recommendations
    if validation_results['data_quality_score'] < completeness_threshold:
        validation_results['recommendations'].append("Consider data imputation for missing values")
        validation_results['recommendations'].append("Investigate root causes of missing data patterns")

    if validation_results['missing_patterns'].get('max_missing_per_row', 0) > 3:
        validation_results['recommendations'].append("Review rows with excessive missing values for potential removal")

    return validation_results

# Test with synthetic data
synthetic_completeness = validate_data_completeness(
    synthetic_df,
    required_columns=['target', 'feature_1', 'feature_2'],
    completeness_threshold=0.95
)

print("🔍 DATA COMPLETENESS VALIDATION RESULTS")
print("=" * 50)
print(f"Overall Completeness: {synthetic_completeness['overall_completeness']:.2%}")
print(f"Data Quality Score: {synthetic_completeness['data_quality_score']:.2%}")
print(f"Validation Passed: {'✅' if synthetic_completeness['validation_passed'] else '❌'}")
print(f"\nTotal Rows: {synthetic_completeness['total_rows']:,}")
print(f"Total Columns: {synthetic_completeness['total_columns']}")

if synthetic_completeness['issues_found']:
    print(f"\n⚠️  Issues Found ({len(synthetic_completeness['issues_found'])}):")
    for issue in synthetic_completeness['issues_found'][:3]:
        print(f"  • {issue}")

print(f"\nColumn Completeness Summary:")
for col, info in list(synthetic_completeness['column_completeness'].items())[:5]:
    status = "✅" if info['passed_threshold'] else "❌"
    print(f"  {status} {col}: {info['completeness_ratio']:.2%} ({info['missing_count']} missing)")

completeness_validation = synthetic_completeness