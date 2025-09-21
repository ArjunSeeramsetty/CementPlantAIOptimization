import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Extract feature importance for tree-based models
models_with_importance = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gbr_model,
    'Extra Trees': et_model
}

# Create feature importance analysis
plt.figure(figsize=(15, 12))

importance_data = {}
feature_names_list = X_train.columns.tolist()

for i, (model_name, model_obj) in enumerate(models_with_importance.items()):
    importance_vals = model_obj.feature_importances_
    importance_data[model_name] = importance_vals

    plt.subplot(2, 2, i+1)
    _indices = np.argsort(importance_vals)[::-1]

    plt.bar(range(len(importance_vals)), importance_vals[_indices])
    plt.title(f'{model_name} - Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.xticks(range(len(importance_vals)), [feature_names_list[j] for j in _indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create comprehensive feature importance comparison
importance_df = pd.DataFrame(importance_data, index=feature_names_list)
importance_df['Average'] = importance_df.mean(axis=1)
importance_df_sorted = importance_df.sort_values('Average', ascending=False)

print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*50)
print(importance_df_sorted.round(4))

# Create heatmap of feature importances
plt.figure(figsize=(12, 8))
sns.heatmap(importance_df_sorted.drop('Average', axis=1).T,
            annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Importance'})
plt.title('Feature Importance Heatmap Across Tree-Based Models')
plt.xlabel('Features')
plt.ylabel('Models')
plt.tight_layout()
plt.show()

# Calculate feature importance statistics
feature_stats = {
    'Most Important': importance_df_sorted.index[0],
    'Least Important': importance_df_sorted.index[-1],
    'Top 3 Features': importance_df_sorted.index[:3].tolist(),
    'Average Importance': importance_df_sorted['Average'].mean(),
    'Importance Std': importance_df_sorted['Average'].std()
}

print(f"\n📈 KEY INSIGHTS:")
print(f"   Most Important Feature: {feature_stats['Most Important']} ({importance_df_sorted['Average'].iloc[0]:.3f})")
print(f"   Top 3 Features: {', '.join(feature_stats['Top 3 Features'])}")
print(f"   Feature Importance Std: {feature_stats['Importance Std']:.3f}")

# Create feature importance ranking plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance_df_sorted)), importance_df_sorted['Average'])
plt.yticks(range(len(importance_df_sorted)), importance_df_sorted.index)
plt.xlabel('Average Importance')
plt.title('Feature Importance Ranking (Average Across Tree Models)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

feature_importance_analysis = {
    'importance_df': importance_df_sorted,
    'feature_stats': feature_stats,
    'top_features': importance_df_sorted.index[:5].tolist()
}