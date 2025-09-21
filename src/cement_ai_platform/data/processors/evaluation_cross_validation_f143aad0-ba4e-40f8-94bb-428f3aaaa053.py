import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

# Define models for cross-validation
cv_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
}

# Define scoring metrics
scoring = {
    'r2': 'r2',
    'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
    'mae': 'neg_mean_absolute_error'
}

# Perform 5-fold cross-validation
cv_results = {}
fold_count = 5

print("🔄 PERFORMING 5-FOLD CROSS-VALIDATION...")
print("="*50)

for model_name, model_obj in cv_models.items():
    print(f"Evaluating {model_name}...")

    cv_scores = cross_validate(
        model_obj, X_final, y_final,
        cv=fold_count, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )

    cv_results[model_name] = {
        'test_r2_mean': cv_scores['test_r2'].mean(),
        'test_r2_std': cv_scores['test_r2'].std(),
        'test_rmse_mean': -cv_scores['test_rmse'].mean(),
        'test_rmse_std': cv_scores['test_rmse'].std(),
        'test_mae_mean': -cv_scores['test_mae'].mean(),
        'test_mae_std': cv_scores['test_mae'].std(),
        'train_r2_mean': cv_scores['train_r2'].mean(),
        'train_r2_std': cv_scores['train_r2'].std(),
        'all_test_r2': cv_scores['test_r2'],
        'all_test_rmse': -cv_scores['test_rmse'],
        'all_test_mae': -cv_scores['test_mae']
    }

# Create cross-validation results DataFrame
cv_df = pd.DataFrame({
    model: {
        'CV_R2_Mean': results['test_r2_mean'],
        'CV_R2_Std': results['test_r2_std'],
        'CV_RMSE_Mean': results['test_rmse_mean'],
        'CV_RMSE_Std': results['test_rmse_std'],
        'CV_MAE_Mean': results['test_mae_mean'],
        'CV_MAE_Std': results['test_mae_std']
    }
    for model, results in cv_results.items()
}).T

cv_df_sorted = cv_df.sort_values('CV_R2_Mean', ascending=False)

print("\n📊 CROSS-VALIDATION RESULTS")
print("="*40)
print(cv_df_sorted.round(4))

# Create cross-validation visualization
plt.figure(figsize=(15, 10))

# R2 scores with error bars
plt.subplot(2, 2, 1)
models_list = list(cv_results.keys())
r2_means = [cv_results[model]['test_r2_mean'] for model in models_list]
r2_stds = [cv_results[model]['test_r2_std'] for model in models_list]

plt.bar(range(len(models_list)), r2_means, yerr=r2_stds, capsize=5, alpha=0.8)
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Cross-Validation R² Scores (Mean ± Std)')
plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# RMSE scores with error bars
plt.subplot(2, 2, 2)
rmse_means = [cv_results[model]['test_rmse_mean'] for model in models_list]
rmse_stds = [cv_results[model]['test_rmse_std'] for model in models_list]

plt.bar(range(len(models_list)), rmse_means, yerr=rmse_stds, capsize=5, alpha=0.8, color='orange')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Cross-Validation RMSE (Mean ± Std)')
plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Box plots for R2 distribution across folds
plt.subplot(2, 2, 3)
r2_data = [cv_results[model]['all_test_r2'] for model in models_list]
box_plot = plt.boxplot(r2_data, labels=models_list, patch_artist=True)
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('R² Score Distribution Across CV Folds')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Stability analysis (coefficient of variation)
plt.subplot(2, 2, 4)
stability_scores = []
for model in models_list:
    cv_coefficient = cv_results[model]['test_r2_std'] / abs(cv_results[model]['test_r2_mean'])
    stability_scores.append(cv_coefficient)

plt.bar(range(len(models_list)), stability_scores, alpha=0.8, color='green')
plt.xlabel('Models')
plt.ylabel('Coefficient of Variation')
plt.title('Model Stability (Lower = More Stable)')
plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare CV vs Hold-out results
holdout_results = evaluation_metrics[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE']].set_index('Model')

comparison_results = {}
for model_name in models_list:
    # Find corresponding model in holdout results
    _holdout_name = None
    for ho_name in holdout_results.index:
        if model_name.lower().replace(' ', '') in ho_name.lower().replace(' ', ''):
            _holdout_name = ho_name
            break

    if _holdout_name:
        comparison_results[model_name] = {
            'CV_R2': cv_results[model_name]['test_r2_mean'],
            'Holdout_R2': holdout_results.loc[_holdout_name, 'Test_R2'],
            'R2_Difference': abs(cv_results[model_name]['test_r2_mean'] - holdout_results.loc[_holdout_name, 'Test_R2']),
            'CV_Std': cv_results[model_name]['test_r2_std']
        }

comparison_df = pd.DataFrame(comparison_results).T

print(f"\n🔄 CV vs HOLD-OUT COMPARISON")
print("="*35)
print(comparison_df.round(4))

print(f"\n🎯 CROSS-VALIDATION INSIGHTS:")
print(f"   Most stable model: {models_list[np.argmin(stability_scores)]}")
print(f"   Best CV performance: {cv_df_sorted.index[0]}")
print(f"   Highest CV std: {cv_df_sorted['CV_R2_Std'].idxmax()}")
print(f"   Average CV R²: {cv_df_sorted['CV_R2_Mean'].mean():.4f}")

cross_validation_analysis = {
    'cv_results': cv_results,
    'cv_summary': cv_df_sorted,
    'stability_scores': dict(zip(models_list, stability_scores)),
    'best_cv_model': cv_df_sorted.index[0],
    'most_stable': models_list[np.argmin(stability_scores)]
}