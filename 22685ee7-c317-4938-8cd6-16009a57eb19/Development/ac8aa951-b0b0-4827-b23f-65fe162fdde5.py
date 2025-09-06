import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings

def validate_mass_balance_constraints(df: pd.DataFrame, 
                                    material_columns: List[str] = None,
                                    balance_tolerance: float = 0.05,
                                    total_mass_column: str = None) -> Dict[str, Any]:
    """
    Validate mass balance constraints for industrial process data
    
    Args:
        df: Input dataframe
        material_columns: List of columns representing material quantities
        balance_tolerance: Allowable tolerance for mass balance (default 5%)
        total_mass_column: Column containing total mass (if available)
    
    Returns:
        Dictionary with mass balance validation results
    """
    # Auto-detect material columns if not provided
    if material_columns is None:
        # Look for cement-related columns
        potential_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in [
                'cement', 'concrete', 'aggregate', 'water', 'fly_ash', 
                'slag', 'superplasticizer', 'admixture', 'sand', 'stone',
                'composition', 'content', 'kg', 'mass', 'weight'
            ]
        )]
        material_columns = [col for col in potential_columns 
                          if df[col].dtype in ['float64', 'int64']]
    
    validation_results = {
        'total_samples': len(df),
        'material_columns_checked': material_columns,
        'balance_violations': 0,
        'balance_violation_rate': 0.0,
        'mass_conservation_score': 0.0,
        'violations_by_severity': {'minor': 0, 'moderate': 0, 'severe': 0},
        'balance_statistics': {},
        'validation_passed': True,
        'issues_found': [],
        'recommendations': []
    }
    
    if not material_columns or len(material_columns) < 2:
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append("Insufficient material columns for mass balance validation")
        return validation_results
    
    # Calculate sum of components
    material_data = df[material_columns].copy()
    material_data = material_data.dropna()  # Remove rows with missing data
    
    if len(material_data) == 0:
        validation_results['issues_found'].append("No complete material composition data available")
        return validation_results
    
    # Calculate component sums
    component_sum = material_data.sum(axis=1)
    
    # If total mass column provided, compare with it
    if total_mass_column and total_mass_column in df.columns:
        total_masses = df.loc[material_data.index, total_mass_column]
        balance_ratios = component_sum / total_masses
        balance_errors = np.abs(balance_ratios - 1.0)
        
        validation_results['balance_statistics']['mean_balance_ratio'] = float(balance_ratios.mean())
        validation_results['balance_statistics']['std_balance_ratio'] = float(balance_ratios.std())
        validation_results['balance_statistics']['median_balance_error'] = float(np.median(balance_errors))
    else:
        # Use relative variation in component sums as balance indicator
        balance_errors = np.abs(component_sum - component_sum.median()) / component_sum.median()
        
        validation_results['balance_statistics']['mean_total_mass'] = float(component_sum.mean())
        validation_results['balance_statistics']['std_total_mass'] = float(component_sum.std())
        validation_results['balance_statistics']['cv_total_mass'] = float(component_sum.std() / component_sum.mean())
    
    # Count violations by severity
    minor_violations = (balance_errors > balance_tolerance) & (balance_errors <= balance_tolerance * 2)
    moderate_violations = (balance_errors > balance_tolerance * 2) & (balance_errors <= balance_tolerance * 5)
    severe_violations = balance_errors > balance_tolerance * 5
    
    validation_results['violations_by_severity']['minor'] = int(minor_violations.sum())
    validation_results['violations_by_severity']['moderate'] = int(moderate_violations.sum())
    validation_results['violations_by_severity']['severe'] = int(severe_violations.sum())
    
    total_violations = minor_violations.sum() + moderate_violations.sum() + severe_violations.sum()
    validation_results['balance_violations'] = int(total_violations)
    validation_results['balance_violation_rate'] = float(total_violations / len(material_data))
    
    # Calculate mass conservation score (1.0 = perfect, 0.0 = worst)
    max_error = balance_errors.max()
    if max_error == 0:
        validation_results['mass_conservation_score'] = 1.0
    else:
        validation_results['mass_conservation_score'] = max(0.0, 1.0 - (max_error / balance_tolerance))
    
    # Validation checks
    if validation_results['balance_violation_rate'] > 0.1:  # More than 10% violations
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"High mass balance violation rate: {validation_results['balance_violation_rate']:.2%}"
        )
    
    if severe_violations.any():
        validation_results['validation_passed'] = False
        validation_results['issues_found'].append(
            f"Found {severe_violations.sum()} severe mass balance violations"
        )
    
    # Generate recommendations
    if validation_results['balance_violation_rate'] > 0.05:
        validation_results['recommendations'].append("Review data collection procedures for material quantities")
        validation_results['recommendations'].append("Implement data validation at source systems")
    
    if validation_results['violations_by_severity']['severe'] > 0:
        validation_results['recommendations'].append("Investigate and correct severe mass balance violations")
    
    # Additional statistics
    validation_results['balance_statistics']['max_error'] = float(balance_errors.max())
    validation_results['balance_statistics']['samples_validated'] = len(material_data)
    
    return validation_results

# Test with cement dataset - simulate material composition data
if 'kaggle_cement_df' in locals():
    # Use cement dataset
    cement_material_columns = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 
                              'superplasticizer', 'coarse_aggregate', 'fine_aggregate']
    
    mass_balance_validation = validate_mass_balance_constraints(
        kaggle_cement_df,
        material_columns=cement_material_columns,
        balance_tolerance=0.05
    )
else:
    # Use synthetic data - simulate material composition
    synthetic_materials = synthetic_df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].copy()
    synthetic_materials.columns = ['cement', 'water', 'aggregate', 'admixture']
    synthetic_materials = np.abs(synthetic_materials * 100 + 500)  # Convert to positive material quantities
    
    mass_balance_validation = validate_mass_balance_constraints(
        synthetic_materials,
        material_columns=['cement', 'water', 'aggregate', 'admixture'],
        balance_tolerance=0.10
    )

print("⚖️  MASS BALANCE VALIDATION RESULTS")
print("=" * 50)
print(f"Samples Validated: {mass_balance_validation['balance_statistics'].get('samples_validated', 0):,}")
print(f"Material Columns: {len(mass_balance_validation['material_columns_checked'])}")
print(f"Balance Violations: {mass_balance_validation['balance_violations']:,} ({mass_balance_validation['balance_violation_rate']:.2%})")
print(f"Mass Conservation Score: {mass_balance_validation['mass_conservation_score']:.3f}")
print(f"Validation Passed: {'✅' if mass_balance_validation['validation_passed'] else '❌'}")

print(f"\nViolations by Severity:")
for severity, count in mass_balance_validation['violations_by_severity'].items():
    print(f"  {severity.capitalize()}: {count:,}")

if mass_balance_validation['balance_statistics']:
    print(f"\nBalance Statistics:")
    for stat, value in list(mass_balance_validation['balance_statistics'].items())[:4]:
        if isinstance(value, float):
            print(f"  {stat.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {stat.replace('_', ' ').title()}: {value}")

if mass_balance_validation['issues_found']:
    print(f"\n⚠️  Issues Found ({len(mass_balance_validation['issues_found'])}):")
    for issue in mass_balance_validation['issues_found']:
        print(f"  • {issue}")

mass_balance_results = mass_balance_validation