from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Create Gradient Boosting model (alternative to XGBoost)
print("🚀 Training Gradient Boosting Regressor...")

# Initialize Gradient Boosting with optimized hyperparameters
gbr_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Train the model
gbr_model.fit(X_train, y_train)

# Make predictions
gbr_train_pred = gbr_model.predict(X_train)
gbr_test_pred = gbr_model.predict(X_test)

# Calculate metrics
gbr_train_mse = mean_squared_error(y_train, gbr_train_pred)
gbr_test_mse = mean_squared_error(y_test, gbr_test_pred)
gbr_train_r2 = r2_score(y_train, gbr_train_pred)
gbr_test_r2 = r2_score(y_test, gbr_test_pred)
gbr_train_mae = mean_absolute_error(y_train, gbr_train_pred)
gbr_test_mae = mean_absolute_error(y_test, gbr_test_pred)

# Store results
gbr_results = {
    'model_name': 'Gradient Boosting',
    'train_r2': gbr_train_r2,
    'test_r2': gbr_test_r2,
    'train_rmse': np.sqrt(gbr_train_mse),
    'test_rmse': np.sqrt(gbr_test_mse),
    'train_mae': gbr_train_mae,
    'test_mae': gbr_test_mae
}

print(f"✅ Gradient Boosting Training Complete!")
print(f"📊 Test R² Score: {gbr_test_r2:.4f}")
print(f"📉 Test RMSE: {np.sqrt(gbr_test_mse):.4f}")
print(f"🎯 Test MAE: {gbr_test_mae:.4f}")