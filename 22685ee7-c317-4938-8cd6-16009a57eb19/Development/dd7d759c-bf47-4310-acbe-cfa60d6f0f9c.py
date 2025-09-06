import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Adaptive Missing Value Imputation with Physics-Informed Methods
class PhysicsInformedImputation:
    def __init__(self):
        self.imputation_methods = {}
        self.physics_constraints = {}
        self.performance_tracker = {}
        
    def add_physics_constraint(self, constraint_name, constraint_func):
        """Add physics-based constraint for validation"""
        self.physics_constraints[constraint_name] = constraint_func
    
    def validate_physics_constraints(self, df, imputed_cols):
        """Validate that imputed values satisfy physics constraints"""
        violations = {}
        
        for constraint_name, constraint_func in self.physics_constraints.items():
            try:
                violations[constraint_name] = constraint_func(df, imputed_cols)
            except Exception as e:
                violations[constraint_name] = f"Error: {str(e)}"
        
        return violations
    
    def adaptive_impute(self, df, target_col=None, missing_threshold=0.1):
        """Perform adaptive imputation using multiple methods"""
        
        imputation_results = {
            'original_missing_count': df.isnull().sum().sum(),
            'methods_used': {},
            'quality_metrics': {},
            'imputed_data': df.copy(),
            'physics_validation': {}
        }
        
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            print("No missing values found in the dataset")
            return imputation_results
        
        print(f"Found missing values in {len(missing_cols)} columns: {missing_cols}")
        
        # For each column with missing values, select best imputation method
        for col in missing_cols:
            missing_ratio = df[col].isnull().sum() / len(df)
            
            print(f"\nProcessing column '{col}' - Missing ratio: {missing_ratio:.3f}")
            
            if missing_ratio > missing_threshold:
                # High missing ratio - use simple imputation
                if df[col].dtype in ['float64', 'int64']:
                    imputation_results['imputed_data'][col].fillna(
                        df[col].median(), inplace=True
                    )
                    imputation_results['methods_used'][col] = 'median_imputation'
                    print(f"  Used median imputation for {col}")
                else:
                    imputation_results['imputed_data'][col].fillna(
                        df[col].mode()[0] if not df[col].mode().empty else 'unknown', 
                        inplace=True
                    )
                    imputation_results['methods_used'][col] = 'mode_imputation'
                    print(f"  Used mode imputation for {col}")
            else:
                # Low missing ratio - use advanced ML-based imputation
                method_used = self._ml_imputation(
                    imputation_results['imputed_data'], col, target_col
                )
                imputation_results['methods_used'][col] = method_used
                print(f"  Used {method_used} for {col}")
        
        # Physics-based validation
        imputed_cols = list(imputation_results['methods_used'].keys())
        imputation_results['physics_validation'] = self.validate_physics_constraints(
            imputation_results['imputed_data'], imputed_cols
        )
        
        # Calculate quality metrics
        final_missing = imputation_results['imputed_data'].isnull().sum().sum()
        imputation_results['quality_metrics'] = {
            'original_missing': imputation_results['original_missing_count'],
            'final_missing': final_missing,
            'imputation_success_rate': 1.0 - (final_missing / max(1, imputation_results['original_missing_count'])),
            'columns_imputed': len(imputed_cols),
            'total_values_imputed': imputation_results['original_missing_count'] - final_missing
        }
        
        return imputation_results
    
    def _ml_imputation(self, df, target_col, reference_col=None):
        """ML-based imputation for a specific column"""
        
        # Get available features (exclude target and columns with too many missing values)
        feature_cols = []
        for col in df.columns:
            if col != target_col and df[col].isnull().sum() / len(df) < 0.5:
                if df[col].dtype in ['float64', 'int64', 'bool']:
                    feature_cols.append(col)
        
        if len(feature_cols) < 2:
            # Fallback to simple imputation
            if df[target_col].dtype in ['float64', 'int64']:
                df[target_col].fillna(df[target_col].median(), inplace=True)
                return 'median_fallback'
            else:
                mode_val = df[target_col].mode()[0] if not df[target_col].mode().empty else 'unknown'
                df[target_col].fillna(mode_val, inplace=True)
                return 'mode_fallback'
        
        # Prepare training data (rows with non-null target values)
        train_mask = df[target_col].notna()
        X_train = df.loc[train_mask, feature_cols].fillna(df[feature_cols].median())
        y_train = df.loc[train_mask, target_col]
        
        # Prepare prediction data (rows with null target values)
        predict_mask = df[target_col].isna()
        X_predict = df.loc[predict_mask, feature_cols].fillna(df[feature_cols].median())
        
        if len(X_train) < 10 or len(X_predict) == 0:
            # Not enough data for ML - use simple imputation
            df[target_col].fillna(df[target_col].median(), inplace=True)
            return 'insufficient_data_fallback'
        
        # Try different models and select best performing
        models = {
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'linear_regression': LinearRegression(),
            'knn': KNeighborsRegressor(n_neighbors=min(5, len(X_train)//2))
        }
        
        best_model = None
        best_score = -np.inf
        best_method = 'median_fallback'
        
        # Cross-validation to select best model
        for method_name, model in models.items():
            try:
                # Scale features for KNN
                if method_name == 'knn':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    scores = cross_val_score(model, X_train_scaled, y_train, cv=min(3, len(X_train)//3), scoring='r2')
                else:
                    scores = cross_val_score(model, X_train, y_train, cv=min(3, len(X_train)//3), scoring='r2')
                
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_method = method_name
                    
            except Exception as e:
                print(f"    Error with {method_name}: {str(e)}")
                continue
        
        # Use best model for imputation
        if best_model is not None and best_score > 0.1:  # Only use if reasonable performance
            try:
                if best_method == 'knn':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_predict_scaled = scaler.transform(X_predict)
                    best_model.fit(X_train_scaled, y_train)
                    predictions = best_model.predict(X_predict_scaled)
                else:
                    best_model.fit(X_train, y_train)
                    predictions = best_model.predict(X_predict)
                
                # Apply physics constraints if available
                predictions = self._apply_physics_constraints(predictions, target_col)
                
                df.loc[predict_mask, target_col] = predictions
                return f'{best_method}_ml_imputation'
                
            except Exception as e:
                print(f"    ML imputation failed: {str(e)}")
                df[target_col].fillna(df[target_col].median(), inplace=True)
                return 'ml_failed_fallback'
        else:
            # ML performance too poor, use simple imputation
            df[target_col].fillna(df[target_col].median(), inplace=True)
            return 'poor_performance_fallback'
    
    def _apply_physics_constraints(self, predictions, col_name):
        """Apply physics-based constraints to predictions"""
        
        # Example constraints (would be customized based on domain)
        if 'temperature' in col_name.lower():
            # Temperature constraints (assuming Celsius)
            predictions = np.clip(predictions, -50, 2000)
        elif 'pressure' in col_name.lower():
            # Pressure must be positive
            predictions = np.maximum(predictions, 0.001)
        elif 'flow' in col_name.lower():
            # Flow rate must be non-negative
            predictions = np.maximum(predictions, 0)
        
        return predictions

# Create physics-informed imputation system
def create_imputation_system(df):
    """Create and configure physics-informed imputation system"""
    
    imputer = PhysicsInformedImputation()
    
    # Add physics constraints
    def mass_balance_constraint(df_check, imputed_cols):
        """Check mass balance after imputation"""
        # Example: sum of inputs should roughly equal outputs
        numerical_cols = df_check.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) >= 4:
            inputs = df_check[numerical_cols[:2]].sum(axis=1)
            outputs = df_check[numerical_cols[2:4]].sum(axis=1)
            balance_error = abs(inputs - outputs) / (inputs + 0.001)
            violations = balance_error > 0.2  # 20% tolerance
            return {
                'violation_count': violations.sum(),
                'violation_percentage': float(violations.mean() * 100),
                'max_error': float(balance_error.max())
            }
        return {'status': 'insufficient_data'}
    
    def range_constraint(df_check, imputed_cols):
        """Check if values are within reasonable ranges"""
        violations = {}
        for col in imputed_cols:
            if col in df_check.columns:
                col_values = df_check[col]
                # Check for extreme outliers (beyond 5 sigma)
                z_scores = np.abs((col_values - col_values.mean()) / (col_values.std() + 1e-6))
                extreme_outliers = z_scores > 5
                violations[col] = {
                    'extreme_outliers': int(extreme_outliers.sum()),
                    'max_z_score': float(z_scores.max())
                }
        return violations
    
    # Add constraints to imputer
    imputer.add_physics_constraint('mass_balance', mass_balance_constraint)
    imputer.add_physics_constraint('range_validation', range_constraint)
    
    return imputer

# Apply to available data
print("=== Creating Physics-Informed Imputation System ===")

imputation_system_results = {}

if 'missing_df' in globals():
    # Create imputation system
    imputer = create_imputation_system(missing_df)
    
    # Perform adaptive imputation
    imputation_results = imputer.adaptive_impute(missing_df, missing_threshold=0.05)
    
    print(f"\n=== Imputation Results ===")
    print(f"Original missing values: {imputation_results['quality_metrics']['original_missing']}")
    print(f"Final missing values: {imputation_results['quality_metrics']['final_missing']}")
    print(f"Success rate: {imputation_results['quality_metrics']['imputation_success_rate']:.4f}")
    print(f"Columns imputed: {imputation_results['quality_metrics']['columns_imputed']}")
    
    # Store results
    imputation_system_results = {
        'imputer': imputer,
        'results': imputation_results,
        'improved_data': imputation_results['imputed_data']
    }
    
    # Calculate improvement in data quality
    original_completeness = 1 - (imputation_results['original_missing_count'] / missing_df.size)
    final_completeness = 1 - (imputation_results['quality_metrics']['final_missing'] / missing_df.size)
    improvement = final_completeness - original_completeness
    
    print(f"Data completeness improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
else:
    print("No missing_df available for imputation system")
    imputation_system_results = {'status': 'no_data_available'}