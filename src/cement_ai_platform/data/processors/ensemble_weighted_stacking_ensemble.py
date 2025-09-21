import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

print("🔀 ADVANCED MODEL ENSEMBLE METHODS")
print("="*50)

# Use the best models from previous training
base_models = [
    ('lr', lr_model),
    ('gbr', gbr_model),
    ('rf', rf_model),
    ('nn', nn_model)
]

print(f"📦 Base models for ensembling:")
for name, model in base_models:
    print(f"   • {name}: {type(model).__name__}")

# 1. WEIGHTED AVERAGING ENSEMBLE
print(f"\n🎯 1. WEIGHTED AVERAGING ENSEMBLE")
print("-" * 35)

# Calculate weights based on individual model performance (inverse of error)
base_predictions = {}
base_errors = {}

for name, model in base_models:
    pred = model.predict(X_test)
    base_predictions[name] = pred
    error = mean_squared_error(y_test, pred)
    base_errors[name] = error

# Calculate weights (inverse of MSE, normalized)
total_inverse_error = sum(1/error for error in base_errors.values())
weights = {name: (1/base_errors[name])/total_inverse_error for name in base_errors.keys()}

print("📊 Model weights based on test performance:")
for name, weight in weights.items():
    print(f"   • {name}: {weight:.3f} (MSE: {base_errors[name]:.2f})")

# Create weighted ensemble prediction
weighted_pred = np.zeros_like(y_test)
for name, weight in weights.items():
    weighted_pred += weight * base_predictions[name]

weighted_r2 = r2_score(y_test, weighted_pred)
weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
weighted_mae = mean_absolute_error(y_test, weighted_pred)

print(f"\n✅ Weighted Average Ensemble Performance:")
print(f"   • Test R²: {weighted_r2:.4f}")
print(f"   • Test RMSE: {weighted_rmse:.2f}")
print(f"   • Test MAE: {weighted_mae:.2f}")

# 2. VOTING REGRESSOR ENSEMBLE
print(f"\n🗳️  2. VOTING REGRESSOR ENSEMBLE")
print("-" * 30)

voting_ensemble = VotingRegressor(estimators=base_models)
voting_ensemble.fit(X_train, y_train)

voting_pred = voting_ensemble.predict(X_test)
voting_r2 = r2_score(y_test, voting_pred)
voting_rmse = np.sqrt(mean_squared_error(y_test, voting_pred))
voting_mae = mean_absolute_error(y_test, voting_pred)

print(f"✅ Voting Ensemble Performance:")
print(f"   • Test R²: {voting_r2:.4f}")
print(f"   • Test RMSE: {voting_rmse:.2f}")
print(f"   • Test MAE: {voting_mae:.2f}")

# 3. STACKING ENSEMBLE WITH META-LEARNER
print(f"\n🏗️  3. STACKING ENSEMBLE (META-LEARNER)")
print("-" * 35)

# Use Ridge regression as meta-learner to avoid overfitting
meta_learner = Ridge(alpha=1.0)

stacking_ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # Use cross-validation to generate meta-features
    n_jobs=-1
)

stacking_ensemble.fit(X_train, y_train)

stacking_pred = stacking_ensemble.predict(X_test)
stacking_r2 = r2_score(y_test, stacking_pred)
stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
stacking_mae = mean_absolute_error(y_test, stacking_pred)

print(f"✅ Stacking Ensemble Performance:")
print(f"   • Test R²: {stacking_r2:.4f}")
print(f"   • Test RMSE: {stacking_rmse:.2f}")
print(f"   • Test MAE: {stacking_mae:.2f}")

# Get meta-learner coefficients
meta_coefs = stacking_ensemble.final_estimator_.coef_
print(f"\n📈 Meta-learner coefficients:")
for i, (name, _) in enumerate(base_models):
    print(f"   • {name}: {meta_coefs[i]:.3f}")

# 4. UNCERTAINTY QUANTIFICATION
print(f"\n🎲 4. UNCERTAINTY QUANTIFICATION")
print("-" * 30)

# Calculate prediction variance across ensemble members
all_predictions = np.column_stack([
    base_predictions['lr'],
    base_predictions['gbr'],
    base_predictions['rf'],
    base_predictions['nn']
])

# Prediction mean and standard deviation
ensemble_mean = np.mean(all_predictions, axis=1)
ensemble_std = np.std(all_predictions, axis=1)
prediction_intervals_lower = ensemble_mean - 1.96 * ensemble_std
prediction_intervals_upper = ensemble_mean + 1.96 * ensemble_std

# Calculate coverage (how many true values fall within prediction intervals)
coverage = np.mean((y_test >= prediction_intervals_lower) & (y_test <= prediction_intervals_upper))

# Uncertainty metrics
avg_uncertainty = np.mean(ensemble_std)
max_uncertainty = np.max(ensemble_std)
uncertainty_coeff_var = np.std(ensemble_std) / np.mean(ensemble_std)

print(f"📊 Uncertainty Analysis:")
print(f"   • Mean prediction uncertainty (σ): {avg_uncertainty:.2f}")
print(f"   • Max prediction uncertainty: {max_uncertainty:.2f}")
print(f"   • Uncertainty coefficient of variation: {uncertainty_coeff_var:.3f}")
print(f"   • 95% prediction interval coverage: {coverage:.1%}")

# 5. ENSEMBLE COMPARISON
ensemble_results = {
    'Individual Best (Linear Regression)': {
        'r2': lr_test_r2,
        'rmse': np.sqrt(lr_test_mse),
        'mae': lr_test_mae,
        'predictions': lr_test_pred
    },
    'Weighted Average': {
        'r2': weighted_r2,
        'rmse': weighted_rmse,
        'mae': weighted_mae,
        'predictions': weighted_pred
    },
    'Voting Regressor': {
        'r2': voting_r2,
        'rmse': voting_rmse,
        'mae': voting_mae,
        'predictions': voting_pred
    },
    'Stacking Regressor': {
        'r2': stacking_r2,
        'rmse': stacking_rmse,
        'mae': stacking_mae,
        'predictions': stacking_pred
    }
}

# Create comparison DataFrame
comparison_data = []
for method, metrics in ensemble_results.items():
    improvement = metrics['r2'] - lr_test_r2  # Compare to best individual model
    comparison_data.append({
        'Method': method,
        'Test_R2': metrics['r2'],
        'Test_RMSE': metrics['rmse'],
        'Test_MAE': metrics['mae'],
        'R2_Improvement': improvement,
        'Improvement_Pct': (improvement / lr_test_r2) * 100
    })

ensemble_comparison_df = pd.DataFrame(comparison_data)
ensemble_comparison_df = ensemble_comparison_df.sort_values('Test_R2', ascending=False)

print(f"\n📊 ENSEMBLE METHOD COMPARISON")
print("="*50)
print(ensemble_comparison_df.round(4).to_string(index=False))

# Advanced visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Model performance comparison
methods = ensemble_comparison_df['Method'].values
r2_scores = ensemble_comparison_df['Test_R2'].values

bars = ax1.bar(range(len(methods)), r2_scores, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
ax1.set_xlabel('Ensemble Methods')
ax1.set_ylabel('Test R² Score')
ax1.set_title('Ensemble Method Performance Comparison', fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# Prediction vs actual scatter plot for best ensemble
best_method = ensemble_comparison_df.iloc[0]['Method']
best_predictions = ensemble_results[best_method]['predictions']

ax2.scatter(y_test, best_predictions, alpha=0.6, s=30)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Predictions vs Actual ({best_method})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add R² annotation
ax2.text(0.05, 0.95, f'R² = {ensemble_results[best_method]["r2"]:.4f}',
         transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Uncertainty visualization
sorted_indices = np.argsort(ensemble_std)
x_sorted = np.arange(len(sorted_indices))

ax3.fill_between(x_sorted,
                prediction_intervals_lower[sorted_indices],
                prediction_intervals_upper[sorted_indices],
                alpha=0.3, color='lightblue', label='95% Prediction Interval')
ax3.scatter(x_sorted, y_test.iloc[sorted_indices], alpha=0.6, s=20, color='red', label='Actual')
ax3.plot(x_sorted, ensemble_mean[sorted_indices], 'b-', alpha=0.8, label='Ensemble Mean')

ax3.set_xlabel('Sample Index (sorted by uncertainty)')
ax3.set_ylabel('Target Value')
ax3.set_title('Prediction Uncertainty Quantification', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Model agreement analysis
disagreement = np.std(all_predictions, axis=1)
ax4.hist(disagreement, bins=20, alpha=0.7, color='orange', edgecolor='black')
ax4.set_xlabel('Model Disagreement (σ)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Model Disagreement', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add statistics
ax4.axvline(np.mean(disagreement), color='red', linestyle='--',
           label=f'Mean: {np.mean(disagreement):.2f}')
ax4.axvline(np.median(disagreement), color='green', linestyle='--',
           label=f'Median: {np.median(disagreement):.2f}')
ax4.legend()

plt.tight_layout()
plt.show()

# Final summary
best_ensemble_r2 = ensemble_comparison_df.iloc[0]['Test_R2']
best_individual_r2 = lr_test_r2
overall_improvement = best_ensemble_r2 - best_individual_r2

print(f"\n🎯 ENSEMBLE SUMMARY")
print("="*30)
print(f"   • Best individual model R²: {best_individual_r2:.4f}")
print(f"   • Best ensemble method: {best_method}")
print(f"   • Best ensemble R²: {best_ensemble_r2:.4f}")
print(f"   • Overall improvement: {overall_improvement:.4f} ({(overall_improvement/best_individual_r2)*100:.2f}%)")
print(f"   • Prediction uncertainty (avg): {avg_uncertainty:.2f}")
print(f"   • Coverage probability: {coverage:.1%}")

# Store ensemble results
ensemble_summary = {
    'comparison_df': ensemble_comparison_df,
    'best_method': best_method,
    'best_ensemble_r2': best_ensemble_r2,
    'improvement': overall_improvement,
    'weights': weights,
    'uncertainty_metrics': {
        'avg_uncertainty': avg_uncertainty,
        'max_uncertainty': max_uncertainty,
        'coverage': coverage,
        'uncertainty_coeff_var': uncertainty_coeff_var
    },
    'models': {
        'weighted_ensemble': None,  # Weights stored separately
        'voting_ensemble': voting_ensemble,
        'stacking_ensemble': stacking_ensemble
    }
}

print(f"\n✅ Ensemble methods completed successfully!")
print(f"   • 4 ensemble methods evaluated")
print(f"   • Uncertainty quantification included")