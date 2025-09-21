from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Create Neural Network model
print("🧠 Training Neural Network (MLP)...")

# Initialize MLPRegressor with optimized hyperparameters
nn_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20
)

# Train the model
nn_model.fit(X_train, y_train)

# Make predictions
nn_train_pred = nn_model.predict(X_train)
nn_test_pred = nn_model.predict(X_test)

# Calculate metrics
nn_train_mse = mean_squared_error(y_train, nn_train_pred)
nn_test_mse = mean_squared_error(y_test, nn_test_pred)
nn_train_r2 = r2_score(y_train, nn_train_pred)
nn_test_r2 = r2_score(y_test, nn_test_pred)
nn_train_mae = mean_absolute_error(y_train, nn_train_pred)
nn_test_mae = mean_absolute_error(y_test, nn_test_pred)

# Store results
nn_results = {
    'model_name': 'Neural Network',
    'train_r2': nn_train_r2,
    'test_r2': nn_test_r2,
    'train_rmse': np.sqrt(nn_train_mse),
    'test_rmse': np.sqrt(nn_test_mse),
    'train_mae': nn_train_mae,
    'test_mae': nn_test_mae
}

print(f"✅ Neural Network Training Complete!")
print(f"📊 Test R² Score: {nn_test_r2:.4f}")
print(f"📉 Test RMSE: {np.sqrt(nn_test_mse):.4f}")
print(f"🎯 Test MAE: {nn_test_mae:.4f}")
print(f"🔄 Training iterations: {nn_model.n_iter_}")
print(f"📉 Final training loss: {nn_model.loss_:.6f}")