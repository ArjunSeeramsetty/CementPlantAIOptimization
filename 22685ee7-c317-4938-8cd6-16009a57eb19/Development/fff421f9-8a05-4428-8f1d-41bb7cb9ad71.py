# Train the PINN model with available data
print("Training PINN model with available cement process data...")

# Convert DataFrames to numpy arrays for physics constraint calculations
X_train_pinn = X_train.values if hasattr(X_train, 'values') else X_train
X_test_pinn = X_test.values if hasattr(X_test, 'values') else X_test
y_train_pinn = y_train.values if hasattr(y_train, 'values') else y_train
y_test_pinn = y_test.values if hasattr(y_test, 'values') else y_test

print(f"Training data shape: {X_train_pinn.shape}")
print(f"Training target shape: {y_train_pinn.shape}")

# Train the PINN model
pinn_model.fit(X_train_pinn, y_train_pinn)

# Make predictions
pinn_train_pred = pinn_model.predict(X_train_pinn)
pinn_test_pred = pinn_model.predict(X_test_pinn)

# Calculate standard ML metrics
pinn_train_mse = mean_squared_error(y_train_pinn, pinn_train_pred)
pinn_test_mse = mean_squared_error(y_test_pinn, pinn_test_pred)
pinn_train_r2 = r2_score(y_train_pinn, pinn_train_pred)
pinn_test_r2 = r2_score(y_test_pinn, pinn_test_pred)

print(f"\nStandard ML Performance:")
print(f"Train R²: {pinn_train_r2:.4f}")
print(f"Test R²: {pinn_test_r2:.4f}")
print(f"Train RMSE: {np.sqrt(pinn_train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(pinn_test_mse):.4f}")

# Evaluate physics compliance
print(f"\nEvaluating physics compliance...")

# Training data physics evaluation
train_physics_eval = pinn_model.evaluate_physics_compliance(X_train_pinn, y_train_pinn)

print(f"\nTraining Data Physics Compliance:")
print(f"Total physics loss: {train_physics_eval['total_physics_loss']:.6f}")
print(f"Violation rate: {train_physics_eval['violation_rate']:.2f}%")
print(f"Physics compliant: {train_physics_eval['compliant']}")

print(f"\nConstraint violations breakdown:")
for constraint, violation in train_physics_eval['constraint_violations'].items():
    print(f"  {constraint}: {violation:.6f}")

# Test data physics evaluation  
test_physics_eval = pinn_model.evaluate_physics_compliance(X_test_pinn, y_test_pinn)

print(f"\nTest Data Physics Compliance:")
print(f"Total physics loss: {test_physics_eval['total_physics_loss']:.6f}")
print(f"Violation rate: {test_physics_eval['violation_rate']:.2f}%")
print(f"Physics compliant: {test_physics_eval['compliant']}")

# Store results for analysis
pinn_results = {
    'train_r2': pinn_train_r2,
    'test_r2': pinn_test_r2,
    'train_rmse': np.sqrt(pinn_train_mse),
    'test_rmse': np.sqrt(pinn_test_mse),
    'train_physics_loss': train_physics_eval['total_physics_loss'],
    'test_physics_loss': test_physics_eval['total_physics_loss'],
    'train_violation_rate': train_physics_eval['violation_rate'],
    'test_violation_rate': test_physics_eval['violation_rate'],
    'train_compliant': train_physics_eval['compliant'],
    'test_compliant': test_physics_eval['compliant'],
    'constraint_violations': test_physics_eval['constraint_violations']
}

print(f"\nPINN Model Summary:")
print(f"Physics constraints implemented: {len(pinn_model.physics_weights)}")
print(f"Target violation rate: <5%")
print(f"Achieved violation rate: {test_physics_eval['violation_rate']:.2f}%")
print(f"Success criteria met: {test_physics_eval['violation_rate'] < 5.0}")

pinn_training_summary = {
    'model_trained': True,
    'physics_constraints_count': len(pinn_model.physics_weights),
    'violation_rate_target': 5.0,
    'achieved_violation_rate': test_physics_eval['violation_rate'],
    'success_criteria_met': test_physics_eval['violation_rate'] < 5.0
}