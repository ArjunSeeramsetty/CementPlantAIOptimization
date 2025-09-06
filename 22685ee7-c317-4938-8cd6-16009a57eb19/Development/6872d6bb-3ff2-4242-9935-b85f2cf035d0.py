import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create comprehensive performance metrics table
performance_metrics = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Neural Network'],
    'Train_R2': [lr_train_r2, dt_train_r2, rf_train_r2, gbr_train_r2, et_train_r2, nn_train_r2],
    'Test_R2': [lr_test_r2, dt_test_r2, rf_test_r2, gbr_test_r2, et_test_r2, nn_test_r2],
    'Train_RMSE': [np.sqrt(lr_train_mse), np.sqrt(dt_train_mse), np.sqrt(rf_train_mse), 
                   np.sqrt(gbr_train_mse), np.sqrt(et_train_mse), np.sqrt(nn_train_mse)],
    'Test_RMSE': [np.sqrt(lr_test_mse), np.sqrt(dt_test_mse), np.sqrt(rf_test_mse), 
                  np.sqrt(gbr_test_mse), np.sqrt(et_test_mse), np.sqrt(nn_test_mse)],
    'Train_MAE': [lr_train_mae, dt_train_mae, rf_train_mae, gbr_train_mae, et_train_mae, nn_train_mae],
    'Test_MAE': [lr_test_mae, dt_test_mae, rf_test_mae, gbr_test_mae, et_test_mae, nn_test_mae]
}

metrics_df = pd.DataFrame(performance_metrics)

# Calculate overfitting metrics
metrics_df['R2_Difference'] = metrics_df['Train_R2'] - metrics_df['Test_R2']
metrics_df['RMSE_Ratio'] = metrics_df['Test_RMSE'] / metrics_df['Train_RMSE']
metrics_df['Overfitting_Score'] = metrics_df['R2_Difference'] + (metrics_df['RMSE_Ratio'] - 1)

# Sort by test R2 score
metrics_df_sorted = metrics_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)

print("📊 COMPREHENSIVE MODEL PERFORMANCE METRICS")
print("="*60)
print(metrics_df_sorted.round(4))

# Create performance visualization
plt.figure(figsize=(15, 10))

# R2 Score comparison
plt.subplot(2, 2, 1)
_x_pos = np.arange(len(metrics_df_sorted))
plt.bar(_x_pos - 0.2, metrics_df_sorted['Train_R2'], 0.4, label='Train R²', alpha=0.8)
plt.bar(_x_pos + 0.2, metrics_df_sorted['Test_R2'], 0.4, label='Test R²', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('R² Score Comparison')
plt.xticks(_x_pos, metrics_df_sorted['Model'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# RMSE comparison
plt.subplot(2, 2, 2)
plt.bar(_x_pos - 0.2, metrics_df_sorted['Train_RMSE'], 0.4, label='Train RMSE', alpha=0.8)
plt.bar(_x_pos + 0.2, metrics_df_sorted['Test_RMSE'], 0.4, label='Test RMSE', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.xticks(_x_pos, metrics_df_sorted['Model'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# Overfitting analysis
plt.subplot(2, 2, 3)
colors = ['green' if score < 0.1 else 'orange' if score < 0.2 else 'red' 
          for score in metrics_df_sorted['Overfitting_Score']]
plt.bar(metrics_df_sorted['Model'], metrics_df_sorted['Overfitting_Score'], color=colors, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Overfitting Score')
plt.title('Overfitting Analysis (Lower is Better)')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate')
plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High')
plt.legend()
plt.grid(True, alpha=0.3)

# Test performance ranking
plt.subplot(2, 2, 4)
plt.scatter(metrics_df_sorted['Test_R2'], metrics_df_sorted['Test_RMSE'], 
           s=100, alpha=0.7, c=range(len(metrics_df_sorted)), cmap='viridis')
for i, model in enumerate(metrics_df_sorted['Model']):
    plt.annotate(model, (metrics_df_sorted['Test_R2'].iloc[i], metrics_df_sorted['Test_RMSE'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.xlabel('Test R² Score')
plt.ylabel('Test RMSE')
plt.title('Performance Trade-off Analysis')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🏆 BEST MODEL: {metrics_df_sorted.iloc[0]['Model']}")
print(f"   Test R²: {metrics_df_sorted.iloc[0]['Test_R2']:.4f}")
print(f"   Test RMSE: {metrics_df_sorted.iloc[0]['Test_RMSE']:.4f}")
print(f"   Overfitting Score: {metrics_df_sorted.iloc[0]['Overfitting_Score']:.4f}")

evaluation_metrics = metrics_df_sorted