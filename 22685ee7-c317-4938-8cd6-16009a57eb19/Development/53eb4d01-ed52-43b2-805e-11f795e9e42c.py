from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Create ExtraTrees model (extremely randomized trees)
print("🌳 Training Extra Trees Regressor...")

# Initialize ExtraTrees with optimized hyperparameters
et_model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train the model
et_model.fit(X_train, y_train)

# Make predictions
et_train_pred = et_model.predict(X_train)
et_test_pred = et_model.predict(X_test)

# Calculate metrics
et_train_mse = mean_squared_error(y_train, et_train_pred)
et_test_mse = mean_squared_error(y_test, et_test_pred)
et_train_r2 = r2_score(y_train, et_train_pred)
et_test_r2 = r2_score(y_test, et_test_pred)
et_train_mae = mean_absolute_error(y_train, et_train_pred)
et_test_mae = mean_absolute_error(y_test, et_test_pred)

# Store results
et_results = {
    'model_name': 'Extra Trees',
    'train_r2': et_train_r2,
    'test_r2': et_test_r2,
    'train_rmse': np.sqrt(et_train_mse),
    'test_rmse': np.sqrt(et_test_mse),
    'train_mae': et_train_mae,
    'test_mae': et_test_mae
}

print(f"✅ Extra Trees Training Complete!")
print(f"📊 Test R² Score: {et_test_r2:.4f}")
print(f"📉 Test RMSE: {np.sqrt(et_test_mse):.4f}")
print(f"🎯 Test MAE: {et_test_mae:.4f}")