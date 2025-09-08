from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Initialize Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

# Calculate performance metrics
lr_train_mse = mean_squared_error(y_train, lr_train_pred)
lr_test_mse = mean_squared_error(y_test, lr_test_pred)
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)

print("🔴 LINEAR REGRESSION RESULTS")
print("=" * 40)
print(f"Training R² Score: {lr_train_r2:.4f}")
print(f"Test R² Score: {lr_test_r2:.4f}")
print(f"Training RMSE: {np.sqrt(lr_train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(lr_test_mse):.4f}")
print(f"Training MAE: {lr_train_mae:.4f}")
print(f"Test MAE: {lr_test_mae:.4f}")

# Store results for comparison
lr_results = {
    'model_name': 'Linear Regression',
    'train_r2': lr_train_r2,
    'test_r2': lr_test_r2,
    'train_rmse': np.sqrt(lr_train_mse),
    'test_rmse': np.sqrt(lr_test_mse),
    'train_mae': lr_train_mae,
    'test_mae': lr_test_mae
}