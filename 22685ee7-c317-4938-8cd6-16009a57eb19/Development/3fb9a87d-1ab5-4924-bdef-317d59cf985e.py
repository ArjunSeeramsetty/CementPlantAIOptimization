from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Calculate performance metrics
rf_train_mse = mean_squared_error(y_train, rf_train_pred)
rf_test_mse = mean_squared_error(y_test, rf_test_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)

print("🌲 RANDOM FOREST RESULTS")
print("=" * 40)
print(f"Training R² Score: {rf_train_r2:.4f}")
print(f"Test R² Score: {rf_test_r2:.4f}")
print(f"Training RMSE: {np.sqrt(rf_train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(rf_test_mse):.4f}")
print(f"Training MAE: {rf_train_mae:.4f}")
print(f"Test MAE: {rf_test_mae:.4f}")

# Store results for comparison
rf_results = {
    'model_name': 'Random Forest',
    'train_r2': rf_train_r2,
    'test_r2': rf_test_r2,
    'train_rmse': np.sqrt(rf_train_mse),
    'test_rmse': np.sqrt(rf_test_mse),
    'train_mae': rf_train_mae,
    'test_mae': rf_test_mae
}