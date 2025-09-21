import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create comprehensive evaluation report
print("🎯 MODEL EVALUATION DASHBOARD - FINAL REPORT")
print("="*60)

# 1. EXECUTIVE SUMMARY
print("\n📋 EXECUTIVE SUMMARY")
print("-" * 30)

best_model = evaluation_metrics.iloc[0]['Model']
best_test_r2 = evaluation_metrics.iloc[0]['Test_R2']
best_test_rmse = evaluation_metrics.iloc[0]['Test_RMSE']

print(f"✅ Best Performing Model: {best_model}")
print(f"   • Test R²: {best_test_r2:.4f} (Explains {best_test_r2*100:.1f}% of variance)")
print(f"   • Test RMSE: {best_test_rmse:.2f}")
print(f"   • Overfitting Score: {evaluation_metrics.iloc[0]['Overfitting_Score']:.4f}")

# Identify model strengths and weaknesses
print(f"\n🔍 Model Performance Insights:")
stable_models = evaluation_metrics[evaluation_metrics['Overfitting_Score'] < 0.1]['Model'].tolist()
overfit_models = evaluation_metrics[evaluation_metrics['Overfitting_Score'] > 0.2]['Model'].tolist()

print(f"   • Most Stable Models: {', '.join(stable_models) if stable_models else 'None'}")
print(f"   • Overfitting Concerns: {', '.join(overfit_models) if overfit_models else 'None'}")

# 2. FEATURE IMPORTANCE SUMMARY
print(f"\n🎯 KEY FEATURE INSIGHTS")
print("-" * 30)
top_features = feature_importance_analysis['top_features'][:3]
print(f"   • Top 3 Most Important Features:")
for i, feature in enumerate(top_features, 1):
    importance_val = feature_importance_analysis['importance_df'].loc[feature, 'Average']
    print(f"     {i}. {feature}: {importance_val:.3f}")

# 3. CROSS-VALIDATION VS HOLDOUT COMPARISON
print(f"\n🔄 CROSS-VALIDATION INSIGHTS")
print("-" * 30)
cv_best = cross_validation_analysis['best_cv_model']
most_stable = cross_validation_analysis['most_stable']
print(f"   • Best CV Performance: {cv_best}")
print(f"   • Most Stable Model: {most_stable}")

cv_avg = cross_validation_analysis['cv_summary']['CV_R2_Mean'].mean()
holdout_avg = evaluation_metrics['Test_R2'].mean()
print(f"   • Average CV R²: {cv_avg:.4f}")
print(f"   • Average Holdout R²: {holdout_avg:.4f}")
print(f"   • Difference: {abs(cv_avg - holdout_avg):.4f}")

# 4. PREDICTION QUALITY ANALYSIS
print(f"\n📊 PREDICTION QUALITY SUMMARY")
print("-" * 30)
best_consistent = prediction_analysis['best_consistency']
best_unbiased = prediction_analysis['best_unbiased']
print(f"   • Most Consistent Predictions: {best_consistent}")
print(f"   • Least Biased Predictions: {best_unbiased}")

# 5. MODEL COMPARISON MATRIX
print(f"\n📈 DETAILED MODEL COMPARISON")
print("-" * 30)
# Create summary comparison
summary_metrics = evaluation_metrics[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfitting_Score']].round(4)
print(summary_metrics.to_string(index=False))

# 6. RECOMMENDATIONS
print(f"\n💡 RECOMMENDATIONS")
print("-" * 30)

# Production deployment recommendation
if best_test_r2 > 0.95 and evaluation_metrics.iloc[0]['Overfitting_Score'] < 0.15:
    recommendation = f"✅ {best_model} is READY for production deployment"
elif best_test_r2 > 0.90:
    recommendation = f"⚠️  {best_model} shows good performance but monitor for overfitting"
else:
    recommendation = f"🔄 Consider additional feature engineering or model tuning"

print(f"   • Production Readiness: {recommendation}")

# Model-specific recommendations
if best_model == "Linear Regression":
    print("   • Strengths: Interpretable, fast, stable")
    print("   • Considerations: May miss non-linear patterns")
elif "Tree" in best_model or "Forest" in best_model:
    print("   • Strengths: Captures non-linearities, feature importance available")
    print("   • Considerations: Monitor for overfitting, less interpretable")
elif "Neural Network" in best_model:
    print("   • Strengths: Complex pattern recognition")
    print("   • Considerations: Requires more data, less interpretable")
elif "Gradient" in best_model:
    print("   • Strengths: Excellent performance, handles various data types")
    print("   • Considerations: Hyperparameter tuning important")

# 7. NEXT STEPS
print(f"\n🚀 SUGGESTED NEXT STEPS")
print("-" * 30)
print("   1. Hyperparameter optimization for top 3 models")
print("   2. Ensemble methods combining best performers")
print("   3. Extended validation with time-based splits")
print("   4. Production monitoring setup")
print("   5. A/B testing framework preparation")

# 8. TECHNICAL METRICS SUMMARY
print(f"\n📋 TECHNICAL SUMMARY")
print("-" * 30)
print(f"   • Total Models Evaluated: {len(evaluation_metrics)}")
print(f"   • Best R² Score: {evaluation_metrics['Test_R2'].max():.4f}")
print(f"   • Lowest RMSE: {evaluation_metrics['Test_RMSE'].min():.2f}")
print(f"   • Cross-Validation Folds: 5")
print(f"   • Dataset Size: {len(X_final)} samples, {len(X_final.columns)} features")

# Create final visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Model performance comparison
models_short = [name.replace(' ', '\n') for name in evaluation_metrics['Model']]
ax1.bar(models_short, evaluation_metrics['Test_R2'], color='skyblue', alpha=0.8)
ax1.set_title('Test R² Scores by Model', fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Feature importance top 5
top_5_features = importance_df_sorted.head(5)
ax2.barh(top_5_features.index, top_5_features['Average'], color='lightcoral', alpha=0.8)
ax2.set_title('Top 5 Feature Importances', fontweight='bold')
ax2.set_xlabel('Average Importance')
ax2.grid(True, alpha=0.3)

# Overfitting analysis
colors = ['green' if score < 0.1 else 'orange' if score < 0.2 else 'red'
          for score in evaluation_metrics['Overfitting_Score']]
ax3.bar(models_short, evaluation_metrics['Overfitting_Score'], color=colors, alpha=0.7)
ax3.set_title('Overfitting Analysis', fontweight='bold')
ax3.set_ylabel('Overfitting Score (Lower = Better)')
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate')
ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High')
ax3.legend()
ax3.grid(True, alpha=0.3)

# CV vs Holdout R² comparison
cv_r2_values = [cross_validation_analysis['cv_results'][model]['test_r2_mean']
                for model in cross_validation_analysis['cv_results'].keys()]
holdout_r2_values = evaluation_metrics['Test_R2'].values

ax4.scatter(holdout_r2_values, cv_r2_values, s=100, alpha=0.7)
ax4.plot([0.7, 1.0], [0.7, 1.0], 'r--', alpha=0.7, label='Perfect Agreement')
ax4.set_xlabel('Holdout R²')
ax4.set_ylabel('Cross-Validation R²')
ax4.set_title('CV vs Holdout Performance', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save summary statistics
evaluation_summary = {
    'best_model': best_model,
    'best_test_r2': best_test_r2,
    'best_test_rmse': best_test_rmse,
    'models_evaluated': len(evaluation_metrics),
    'top_features': top_features,
    'cv_best': cv_best,
    'most_stable': most_stable,
    'recommendation': recommendation
}

print(f"\n✅ EVALUATION COMPLETE - {len(evaluation_metrics)} models assessed with comprehensive analysis")
print("="*60)