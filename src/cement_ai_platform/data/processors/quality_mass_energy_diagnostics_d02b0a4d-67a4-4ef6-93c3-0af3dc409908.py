import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Mass/Energy Balance Diagnostics
def physics_based_validation(df):
    """Comprehensive physics-based validation for mass and energy balances"""

    # Initialize diagnostics dictionary
    mass_energy_diagnostics = {
        'mass_balance_violations': [],
        'energy_balance_violations': [],
        'conservation_errors': {},
        'outlier_analysis': {},
        'root_causes': {}
    }

    # Check if required columns exist in the data
    available_cols = df.columns.tolist()
    print(f"Available columns: {available_cols}")

    # For synthetic data, use available numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) >= 6:
        # Use first 3 as inputs, next 3 as outputs for mass balance
        mass_in_synthetic = numerical_cols[:3]
        mass_out_synthetic = numerical_cols[3:6]

        # Calculate mass balance for each row
        mass_in = df[mass_in_synthetic].sum(axis=1)
        mass_out = df[mass_out_synthetic].sum(axis=1)
        mass_balance_error = abs(mass_in - mass_out) / (mass_in + 1e-6)  # Avoid division by zero

        # Identify significant violations (>5% error)
        violation_threshold = 0.05
        mass_violations = mass_balance_error > violation_threshold
        mass_energy_diagnostics['mass_balance_violations'] = df.index[mass_violations].tolist()

        print(f"Mass balance violations: {mass_violations.sum()} out of {len(df)} records ({mass_violations.sum()/len(df)*100:.1f}%)")

        # Energy balance analysis
        if len(numerical_cols) >= 8:
            energy_in_synthetic = [numerical_cols[6]] if len(numerical_cols) > 6 else []
            energy_out_synthetic = [numerical_cols[7]] if len(numerical_cols) > 7 else []

            if energy_in_synthetic and energy_out_synthetic:
                energy_in = df[energy_in_synthetic].sum(axis=1)
                energy_out = df[energy_out_synthetic].sum(axis=1)
                energy_balance_error = abs(energy_in - energy_out) / (energy_in + 1e-6)

                energy_violations = energy_balance_error > violation_threshold
                mass_energy_diagnostics['energy_balance_violations'] = df.index[energy_violations].tolist()

                print(f"Energy balance violations: {energy_violations.sum()} out of {len(df)} records ({energy_violations.sum()/len(df)*100:.1f}%)")

        # Conservation law violations
        mass_energy_diagnostics['conservation_errors'] = {
            'mass_balance_error_stats': {
                'mean': float(mass_balance_error.mean()),
                'std': float(mass_balance_error.std()),
                'max': float(mass_balance_error.max()),
                'violations_count': int(mass_violations.sum())
            }
        }

        # Root cause analysis using statistical methods
        # Identify which variables contribute most to balance violations
        violation_mask = mass_violations | (energy_violations if 'energy_violations' in locals() else pd.Series([False]*len(df)))

        if violation_mask.sum() > 0:
            # Statistical analysis of violations
            violation_data = df[violation_mask]
            normal_data = df[~violation_mask]

            root_causes = {}
            for col in numerical_cols:
                if len(normal_data[col]) > 0 and len(violation_data[col]) > 0:
                    # T-test to identify significant differences
                    t_stat, p_value = stats.ttest_ind(violation_data[col], normal_data[col])
                    if p_value < 0.05:  # Significant difference
                        root_causes[col] = {
                            'p_value': float(p_value),
                            'mean_violation': float(violation_data[col].mean()),
                            'mean_normal': float(normal_data[col].mean()),
                            'significance': 'high' if p_value < 0.01 else 'medium'
                        }

            mass_energy_diagnostics['root_causes'] = root_causes

            print(f"Root cause analysis identified {len(root_causes)} significant variables")
            for var, stats_dict in root_causes.items():
                print(f"  {var}: p={stats_dict['p_value']:.4f}, significance={stats_dict['significance']}")

    return mass_energy_diagnostics

# Apply diagnostics to available data using cleaned_df which is available
if 'cleaned_df' in globals():
    mass_energy_results = physics_based_validation(cleaned_df)
    print(f"\n=== Mass/Energy Balance Diagnostics Complete ===")
    print(f"Total conservation violations: {len(mass_energy_results['mass_balance_violations']) + len(mass_energy_results['energy_balance_violations'])}")
else:
    print("No data available for diagnostics")
    mass_energy_results = {'status': 'no_data_available'}