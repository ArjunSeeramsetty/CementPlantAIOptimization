import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collect all advanced model results
advanced_results = [gbr_results, et_results, nn_results]

# Create comprehensive comparison DataFrame
all_models = all_results + advanced_results
comprehensive_df = pd.DataFrame(all_models)

print("🚀 ADVANCED MODELS vs BASELINE MODELS COMPARISON")
print("=" * 70)

# Display results table
print("\n📊 COMPLETE PERFORMANCE METRICS:")
print(comprehensive_df.round(4))

print("\n🏆 MODEL RANKINGS:")
print("\n📈 Best Test R² Score:")
best_overall = comprehensive_df.loc[comprehensive_df['test_r2'].idxmax()]
print(f"   {best_overall['model_name']}: {best_overall['test_r2']:.4f}")

print("\n📉 Lowest Test RMSE:")
best_rmse_overall = comprehensive_df.loc[comprehensive_df['test_rmse'].idxmin()]
print(f"   {best_rmse_overall['model_name']}: {best_rmse_overall['test_rmse']:.4f}")

print("\n🎯 Best Test MAE:")
best_mae_overall = comprehensive_df.loc[comprehensive_df['test_mae'].idxmin()]
print(f"   {best_mae_overall['model_name']}: {best_mae_overall['test_mae']:.4f}")

# Compare advanced vs baseline models
baseline_models = ['Linear Regression', 'Decision Tree', 'Random Forest']
advanced_models_list = ['Gradient Boosting', 'Extra Trees', 'Neural Network']

baseline_avg_r2 = comprehensive_df[comprehensive_df['model_name'].isin(baseline_models)]['test_r2'].mean()
advanced_avg_r2 = comprehensive_df[comprehensive_df['model_name'].isin(advanced_models_list)]['test_r2'].mean()

print("\n📊 PERFORMANCE COMPARISON:")
print(f"\n🔵 Baseline Models Average R²: {baseline_avg_r2:.4f}")
print(f"🔴 Advanced Models Average R²: {advanced_avg_r2:.4f}")
print(f"📈 Improvement: {((advanced_avg_r2 - baseline_avg_r2) / baseline_avg_r2) * 100:.2f}%")

print("\n🔍 DETAILED ANALYSIS:")

print("\n• Linear Regression (Baseline):")
lr_row = comprehensive_df[comprehensive_df['model_name'] == 'Linear Regression'].iloc[0]
print(f"  - Test R²: {lr_row['test_r2']:.4f} | RMSE: {lr_row['test_rmse']:.4f} | MAE: {lr_row['test_mae']:.4f}")

print("\n• Gradient Boosting (Advanced):")
gb_row = comprehensive_df[comprehensive_df['model_name'] == 'Gradient Boosting'].iloc[0]
gb_improvement = ((gb_row['test_r2'] - lr_row['test_r2']) / lr_row['test_r2']) * 100
print(f"  - Test R²: {gb_row['test_r2']:.4f} | RMSE: {gb_row['test_rmse']:.4f} | MAE: {gb_row['test_mae']:.4f}")
print(f"  - R² improvement vs Linear: {gb_improvement:.2f}%")

print("\n• Extra Trees (Advanced):")
et_row = comprehensive_df[comprehensive_df['model_name'] == 'Extra Trees'].iloc[0]
et_improvement = ((et_row['test_r2'] - lr_row['test_r2']) / lr_row['test_r2']) * 100
print(f"  - Test R²: {et_row['test_r2']:.4f} | RMSE: {et_row['test_rmse']:.4f} | MAE: {et_row['test_mae']:.4f}")
print(f"  - R² improvement vs Linear: {et_improvement:.2f}%")

print("\n• Neural Network (Advanced):")
nn_row = comprehensive_df[comprehensive_df['model_name'] == 'Neural Network'].iloc[0]
nn_improvement = ((nn_row['test_r2'] - lr_row['test_r2']) / lr_row['test_r2']) * 100
print(f"  - Test R²: {nn_row['test_r2']:.4f} | RMSE: {nn_row['test_rmse']:.4f} | MAE: {nn_row['test_mae']:.4f}")
print(f"  - R² improvement vs Linear: {nn_improvement:.2f}%")

print("\n✅ SUCCESS CRITERIA ASSESSMENT:")
print("✓ Advanced models (Gradient Boosting, Extra Trees, Neural Network) implemented")
print("✓ Performance comparison completed against baseline models")
print("✓ Comprehensive evaluation metrics generated")

# Check if any advanced model beats best baseline
best_baseline_r2 = comprehensive_df[comprehensive_df['model_name'].isin(baseline_models)]['test_r2'].max()
best_advanced_r2 = comprehensive_df[comprehensive_df['model_name'].isin(advanced_models_list)]['test_r2'].max()

if best_advanced_r2 > best_baseline_r2:
    improvement_pct = ((best_advanced_r2 - best_baseline_r2) / best_baseline_r2) * 100
    print(f"✓ BEST advanced model outperforms BEST baseline by {improvement_pct:.2f}%")
else:
    print("⚠ Advanced models did not outperform the best baseline model")

# Store final results
final_comparison = comprehensive_df.copy()
performance_summary = {
    'best_model_overall': best_overall['model_name'],
    'best_test_r2_overall': best_overall['test_r2'],
    'baseline_avg_r2': baseline_avg_r2,
    'advanced_avg_r2': advanced_avg_r2,
    'total_models_tested': len(comprehensive_df),
    'advanced_models_built': len(advanced_models_list)
}