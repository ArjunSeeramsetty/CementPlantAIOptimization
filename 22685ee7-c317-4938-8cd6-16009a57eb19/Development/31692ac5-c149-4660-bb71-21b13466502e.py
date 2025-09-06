import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Walk-forward validation implementation
def walk_forward_validation(temporal_data, n_splits=5, initial_train_size=0.6):
    """
    Implement walk-forward validation for time series data
    """
    n_samples = len(temporal_data)
    initial_size = int(n_samples * initial_train_size)
    step_size = (n_samples - initial_size) // n_splits
    
    # Feature columns (exclude target and timestamp)
    feature_cols = [col for col in temporal_data.columns if col not in ['target', 'timestamp']]
    
    results = {
        'LinearRegression': {'r2': [], 'rmse': [], 'mae': []},
        'RandomForest': {'r2': [], 'rmse': [], 'mae': []},
        'NeuralNetwork': {'r2': [], 'rmse': [], 'mae': []}
    }
    
    fold_info = []
    
    for fold in range(n_splits):
        # Define training and test windows
        train_end = initial_size + fold * step_size
        test_start = train_end
        test_end = test_start + step_size
        
        # Ensure we don't exceed data bounds
        if test_end > n_samples:
            test_end = n_samples
        
        # Split data
        X_train_fold = temporal_data.iloc[:train_end][feature_cols]
        y_train_fold = temporal_data.iloc[:train_end]['target']
        X_test_fold = temporal_data.iloc[test_start:test_end][feature_cols]
        y_test_fold = temporal_data.iloc[test_start:test_end]['target']
        
        fold_info.append({
            'fold': fold + 1,
            'train_size': len(X_train_fold),
            'test_size': len(X_test_fold),
            'train_period': f"{temporal_data.iloc[0]['timestamp'].strftime('%Y-%m-%d')} to {temporal_data.iloc[train_end-1]['timestamp'].strftime('%Y-%m-%d')}",
            'test_period': f"{temporal_data.iloc[test_start]['timestamp'].strftime('%Y-%m-%d')} to {temporal_data.iloc[test_end-1]['timestamp'].strftime('%Y-%m-%d')}"
        })
        
        # Test models
        models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
        }
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred = model.predict(X_test_fold)
            
            # Calculate metrics
            r2 = r2_score(y_test_fold, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            mae = mean_absolute_error(y_test_fold, y_pred)
            
            # Store results
            results[model_name]['r2'].append(r2)
            results[model_name]['rmse'].append(rmse)
            results[model_name]['mae'].append(mae)
    
    return results, fold_info

# Run walk-forward validation
wf_results, wf_fold_info = walk_forward_validation(temporal_df, n_splits=5)

print("Walk-Forward Validation Results:")
print("=" * 50)

for model_name, metrics in wf_results.items():
    print(f"\n{model_name}:")
    print(f"  Average R²: {np.mean(metrics['r2']):.4f} ± {np.std(metrics['r2']):.4f}")
    print(f"  Average RMSE: {np.mean(metrics['rmse']):.4f} ± {np.std(metrics['rmse']):.4f}")
    print(f"  Average MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")

print(f"\nValidation performed across {len(wf_fold_info)} time windows")