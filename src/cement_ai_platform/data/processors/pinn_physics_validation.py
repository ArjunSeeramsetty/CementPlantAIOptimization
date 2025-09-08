import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Train the PINN model with physics constraints
print("Training Physics-Informed Neural Network...")

# Use numpy arrays for calculations
X_train_pinn = X_train.values if hasattr(X_train, 'values') else X_train
X_test_pinn = X_test.values if hasattr(X_test, 'values') else X_test  
y_train_pinn = y_train.values if hasattr(y_train, 'values') else y_train
y_test_pinn = y_test.values if hasattr(y_test, 'values') else y_test

print(f"Training data: {X_train_pinn.shape}, Target: {y_train_pinn.shape}")

# Train the PINN model
pinn_model.fit(X_train_pinn, y_train_pinn)

# Make predictions
pinn_train_predictions = pinn_model.predict(X_train_pinn)
pinn_test_predictions = pinn_model.predict(X_test_pinn)

# Calculate standard ML performance metrics
pinn_train_mse = mean_squared_error(y_train_pinn, pinn_train_predictions)
pinn_test_mse = mean_squared_error(y_test_pinn, pinn_test_predictions)
pinn_train_r2 = r2_score(y_train_pinn, pinn_train_predictions)
pinn_test_r2 = r2_score(y_test_pinn, pinn_test_predictions)

print(f"\n=== PINN Model Performance ===")
print(f"Train R²: {pinn_train_r2:.4f}")
print(f"Test R²: {pinn_test_r2:.4f}")
print(f"Train RMSE: {np.sqrt(pinn_train_mse):.4f}")
print(f"Test RMSE: {np.sqrt(pinn_test_mse):.4f}")

# Evaluate physics compliance
print(f"\n=== Physics Compliance Analysis ===")
train_physics_compliance = pinn_model.evaluate_physics_compliance(X_train_pinn, y_train_pinn)
test_physics_compliance = pinn_model.evaluate_physics_compliance(X_test_pinn, y_test_pinn)

print(f"Training Data:")
print(f"  Physics loss: {train_physics_compliance['total_physics_loss']:.6f}")
print(f"  Violation rate: {train_physics_compliance['violation_rate']:.2f}%")
print(f"  Compliant: {train_physics_compliance['compliant']}")

print(f"\nTest Data:")
print(f"  Physics loss: {test_physics_compliance['total_physics_loss']:.6f}")
print(f"  Violation rate: {test_physics_compliance['violation_rate']:.2f}%")
print(f"  Compliant: {test_physics_compliance['compliant']}")

# Detailed constraint analysis
print(f"\n=== Individual Physics Constraints ===")
for constraint_name, violation_score in test_physics_compliance['constraint_violations'].items():
    compliance_status = "PASS" if violation_score < 0.1 else "FAIL"
    print(f"  {constraint_name}: {violation_score:.6f} [{compliance_status}]")

# Final results summary
pinn_final_results = {
    'model_type': 'Physics-Informed Neural Network',
    'physics_constraints_implemented': len(pinn_model.physics_weights),
    'ml_performance': {
        'train_r2': pinn_train_r2,
        'test_r2': pinn_test_r2,
        'train_rmse': np.sqrt(pinn_train_mse),
        'test_rmse': np.sqrt(pinn_test_mse)
    },
    'physics_compliance': {
        'train_violation_rate': train_physics_compliance['violation_rate'],
        'test_violation_rate': test_physics_compliance['violation_rate'],
        'target_violation_rate': 5.0,
        'meets_target': test_physics_compliance['violation_rate'] < 5.0
    },
    'constraint_details': test_physics_compliance['constraint_violations']
}

print(f"\n=== PINN SUMMARY ===")
print(f"Physics constraints: {pinn_final_results['physics_constraints_implemented']}")
print(f"Target violation rate: <5%")
print(f"Achieved violation rate: {pinn_final_results['physics_compliance']['test_violation_rate']:.2f}%")
print(f"SUCCESS CRITERIA MET: {pinn_final_results['physics_compliance']['meets_target']}")

if pinn_final_results['physics_compliance']['meets_target']:
    print("✅ PINN model successfully incorporates physics constraints with <5% violation rate!")
else:
    print("❌ PINN model needs further tuning to meet <5% violation rate target.")