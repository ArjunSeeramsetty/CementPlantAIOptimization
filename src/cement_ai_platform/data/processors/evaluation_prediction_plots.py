import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Compile all predictions and actual values
predictions_data = {
    'Linear Regression': {'train': lr_train_pred, 'test': lr_test_pred},
    'Decision Tree': {'train': dt_train_pred, 'test': dt_test_pred},
    'Random Forest': {'train': rf_train_pred, 'test': rf_test_pred},
    'Gradient Boosting': {'train': gbr_train_pred, 'test': gbr_test_pred},
    'Extra Trees': {'train': et_train_pred, 'test': et_test_pred},
    'Neural Network': {'train': nn_train_pred, 'test': nn_test_pred}
}

# Create prediction vs actual plots
plt.figure(figsize=(18, 12))

for i, (model_name, preds) in enumerate(predictions_data.items()):
    # Training predictions
    plt.subplot(2, 6, i + 1)
    plt.scatter(y_train, preds['train'], alpha=0.6, s=30)
    _min_val = min(y_train.min(), preds['train'].min())
    _max_val = max(y_train.max(), preds['train'].max())
    plt.plot([_min_val, _max_val], [_min_val, _max_val], 'r--', lw=2, alpha=0.8)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}\nTrain R²: {r2_score(y_train, preds["train"]):.3f}')
    plt.grid(True, alpha=0.3)

    # Test predictions
    plt.subplot(2, 6, i + 7)
    plt.scatter(y_test, preds['test'], alpha=0.6, s=30, color='orange')
    _min_val_test = min(y_test.min(), preds['test'].min())
    _max_val_test = max(y_test.max(), preds['test'].max())
    plt.plot([_min_val_test, _max_val_test], [_min_val_test, _max_val_test], 'r--', lw=2, alpha=0.8)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Test R²: {r2_score(y_test, preds["test"]):.3f}')
    plt.grid(True, alpha=0.3)

plt.suptitle('Prediction vs Actual Plots - Train (Top) and Test (Bottom)', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()

# Create residual plots
plt.figure(figsize=(18, 8))

for i, (model_name, preds) in enumerate(predictions_data.items()):
    plt.subplot(2, 6, i + 1)
    _residuals_test = y_test - preds['test']
    plt.scatter(preds['test'], _residuals_test, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}')
    plt.grid(True, alpha=0.3)

    # Residual distribution
    plt.subplot(2, 6, i + 7)
    plt.hist(_residuals_test, bins=20, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'Residual Distribution')
    plt.grid(True, alpha=0.3)

plt.suptitle('Residual Analysis - Scatter (Top) and Distribution (Bottom)', fontsize=16, y=0.95)
plt.tight_layout()
plt.show()

# Calculate prediction quality metrics
prediction_quality = {}
for model_name, preds in predictions_data.items():
    _test_residuals = y_test - preds['test']
    _train_residuals = y_train - preds['train']

    prediction_quality[model_name] = {
        'test_residual_std': _test_residuals.std(),
        'test_residual_mean': _test_residuals.mean(),
        'train_residual_std': _train_residuals.std(),
        'train_residual_mean': _train_residuals.mean(),
        'test_abs_residual_mean': np.abs(_test_residuals).mean(),
        'homoscedasticity_score': np.corrcoef(preds['test'], np.abs(_test_residuals))[0,1]
    }

quality_df = pd.DataFrame(prediction_quality).T

print("📊 PREDICTION QUALITY ANALYSIS")
print("="*50)
print(quality_df.round(4))

# Best and worst prediction examples
print(f"\n🎯 PREDICTION INSIGHTS:")
print(f"   Most consistent residuals: {quality_df['test_residual_std'].idxmin()}")
print(f"   Least biased predictions: {quality_df['test_residual_mean'].abs().idxmin()}")
print(f"   Best absolute accuracy: {quality_df['test_abs_residual_mean'].idxmin()}")

prediction_analysis = {
    'predictions_data': predictions_data,
    'quality_metrics': quality_df,
    'best_consistency': quality_df['test_residual_std'].idxmin(),
    'best_unbiased': quality_df['test_residual_mean'].abs().idxmin()
}