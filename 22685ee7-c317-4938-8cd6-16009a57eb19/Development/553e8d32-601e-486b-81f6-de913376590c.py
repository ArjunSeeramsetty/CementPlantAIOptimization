import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats

print("🔧 HYPERPARAMETER OPTIMIZATION")
print("="*50)

# Define parameter grids for top 3 models
param_grids = {
    'Gradient Boosting': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5, 0.8]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'tanh']
    }
}

# Define model classes
model_classes = {
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Neural Network': MLPRegressor(max_iter=1000, random_state=42)
}

# Store optimization results
tuning_results = {}

# Optimize each model using RandomizedSearchCV for efficiency
for model_name in param_grids.keys():
    print(f"\n🎯 Optimizing {model_name}...")
    print("-" * 30)
    
    model = model_classes[model_name]
    param_grid = param_grids[model_name]
    
    # Use RandomizedSearchCV for efficiency (faster than GridSearchCV)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    # Evaluate on test set
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Store results
    tuning_results[model_name] = {
        'best_params': best_params,
        'cv_score': best_cv_score,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'model': best_model,
        'search_results': random_search
    }
    
    print(f"✅ Best CV R²: {best_cv_score:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Best params: {best_params}")

# Compare with baseline models
baseline_scores = {
    'Linear Regression': 0.9898,  # From existing results
    'Gradient Boosting': 0.9379,
    'Random Forest': 0.9045,
    'Neural Network': 0.9848
}

# Create comparison DataFrame
comparison_data = []
for model_name, result in tuning_results.items():
    baseline_r2 = baseline_scores.get(model_name, 0.0)
    improvement = result['test_r2'] - baseline_r2
    
    comparison_data.append({
        'Model': f"{model_name} (Tuned)",
        'Baseline_R2': baseline_r2,
        'Tuned_R2': result['test_r2'],
        'CV_Score': result['cv_score'],
        'Train_RMSE': result['train_rmse'],
        'Test_RMSE': result['test_rmse'],
        'Improvement': improvement,
        'Improvement_Pct': (improvement / baseline_r2) * 100 if baseline_r2 > 0 else 0
    })

tuned_comparison_df = pd.DataFrame(comparison_data)
tuned_comparison_df = tuned_comparison_df.sort_values('Tuned_R2', ascending=False)

print(f"\n📊 HYPERPARAMETER TUNING RESULTS")
print("="*50)
print(tuned_comparison_df.round(4).to_string(index=False))

# Calculate overall improvement statistics
avg_improvement = tuned_comparison_df['Improvement'].mean()
max_improvement = tuned_comparison_df['Improvement'].max()
best_tuned_model = tuned_comparison_df.iloc[0]['Model']
best_tuned_r2 = tuned_comparison_df.iloc[0]['Tuned_R2']

print(f"\n🚀 OPTIMIZATION SUMMARY")
print("="*40)
print(f"   • Best tuned model: {best_tuned_model}")
print(f"   • Best tuned R²: {best_tuned_r2:.4f}")  
print(f"   • Average improvement: {avg_improvement:.4f} ({avg_improvement*100:.2f}%)")
print(f"   • Maximum improvement: {max_improvement:.4f} ({max_improvement*100:.2f}%)")

# Enhanced visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Model performance comparison
models = tuned_comparison_df['Model'].str.replace(' (Tuned)', '')
baseline_scores_list = tuned_comparison_df['Baseline_R2'].values
tuned_scores = tuned_comparison_df['Tuned_R2'].values

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, baseline_scores_list, width, label='Baseline', alpha=0.7, color='lightcoral')
bars2 = ax1.bar(x_pos + width/2, tuned_scores, width, label='Tuned', alpha=0.7, color='skyblue')

ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_title('Baseline vs Tuned Model Performance', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Improvement analysis
improvements = tuned_comparison_df['Improvement'].values
colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = ax2.bar(range(len(models)), improvements, color=colors, alpha=0.8)
ax2.set_xlabel('Models')
ax2.set_ylabel('R² Improvement')
ax2.set_title('Performance Improvement from Tuning', fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.grid(True, alpha=0.3)

# Add improvement values on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.0005 if height > 0 else -0.0005),
             f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# Cross-validation scores distribution
cv_scores = [result['cv_score'] for result in tuning_results.values()]
test_scores = [result['test_r2'] for result in tuning_results.values()]

ax3.scatter(cv_scores, test_scores, s=100, alpha=0.7, color='purple')
ax3.plot([0.9, 1.0], [0.9, 1.0], 'r--', alpha=0.7, label='Perfect Agreement')

# Add model labels
for i, model_name in enumerate(tuning_results.keys()):
    ax3.annotate(model_name, (cv_scores[i], test_scores[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax3.set_xlabel('Cross-Validation R²')
ax3.set_ylabel('Test R²')
ax3.set_title('CV vs Test Performance (Tuned Models)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Feature importance for best model (if available)
best_model_obj = tuned_comparison_df.iloc[0]['Model'].replace(' (Tuned)', '')
if best_model_obj in tuning_results:
    best_estimator = tuning_results[best_model_obj]['model']
    
    if hasattr(best_estimator, 'feature_importances_'):
        importances = best_estimator.feature_importances_
        feature_names = X_train.columns
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_features = indices[:8]  # Top 8 features
        
        ax4.bar(range(len(top_features)), importances[top_features], alpha=0.8, color='lightgreen')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Feature Importance')
        ax4.set_title(f'Feature Importance ({best_model_obj})', fontweight='bold')
        ax4.set_xticks(range(len(top_features)))
        ax4.set_xticklabels([feature_names[i] for i in top_features], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, f'{best_model_obj}\nNo feature importance\navailable', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title(f'Feature Importance ({best_model_obj})', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"\n📈 DETAILED TUNING ANALYSIS")
print("="*40)
for model_name, result in tuning_results.items():
    print(f"\n{model_name}:")
    print(f"   • Best parameters: {result['best_params']}")
    print(f"   • CV score: {result['cv_score']:.4f}")
    print(f"   • Test R²: {result['test_r2']:.4f}")
    baseline_score = baseline_scores.get(model_name, 0)
    improvement = result['test_r2'] - baseline_score
    print(f"   • Improvement: {improvement:.4f} ({(improvement/baseline_score)*100:.2f}%)")

# Store optimization summary
optimization_summary = {
    'tuning_results': tuning_results,
    'comparison_df': tuned_comparison_df,
    'best_tuned_model': best_tuned_model,
    'best_tuned_r2': best_tuned_r2,
    'average_improvement': avg_improvement,
    'max_improvement': max_improvement,
    'baseline_scores': baseline_scores
}

print(f"\n✅ Hyperparameter optimization complete!")
print(f"   • {len(tuning_results)} models optimized with RandomizedSearchCV")
print(f"   • Best overall improvement: {max_improvement:.4f}")