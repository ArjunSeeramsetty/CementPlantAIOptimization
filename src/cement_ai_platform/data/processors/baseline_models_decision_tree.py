from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Initialize Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

# Calculate performance metrics
dt_train_mse = mean_squared_error(y_train, dt_train_pred)
dt_test_mse = mean_squared_error(y_test, dt_test_pred)
dt_train_r2 = r2_score(y_train, dt_train_pred)
dt_test_r2 = r2_score(y_test, dt_test_pred)
dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
dt_test_mae = mean_absolute_error(y_test, dt_test_pred)

print("🌳 DECISION TREE RESULTS")
print("=" * 40)
print(f"Training R² Score: {dt_train_r2:.4f}")
print(f"Test R² Score: {dt_test_r2:.4f}")
print(f"Training RMSE: {np.sqrt(dt_train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(dt_test_mse):.4f}")
print(f"Training MAE: {dt_train_mae:.4f}")
print(f"Test MAE: {dt_test_mae:.4f}")

# Store results for comparison
dt_results = {
    'model_name': 'Decision Tree',
    'train_r2': dt_train_r2,
    'test_r2': dt_test_r2,
    'train_rmse': np.sqrt(dt_train_mse),
    'test_rmse': np.sqrt(dt_test_mse),
    'train_mae': dt_train_mae,
    'test_mae': dt_test_mae
}