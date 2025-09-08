# Train Thermal Energy Models
print("\n📊 THERMAL ENERGY PREDICTION MODELS (kJ/kg clinker)")
print("=" * 60)

# Define feature columns (exclude target variables)
feature_columns = [col for col in processed_data.columns 
                  if col not in ['thermal_energy_kjkg', 'electrical_energy_kwht']]

X = processed_data[feature_columns]
y_thermal = processed_data['thermal_energy_kjkg']

# Feature selection for thermal energy
X_thermal, thermal_features, thermal_selector = energy_predictor.select_energy_features(
    X, y_thermal, 'thermal', n_features=25
)

print(f"🎯 Selected thermal features: {thermal_features[:5]}... ({len(thermal_features)} total)")

# Train thermal energy models
thermal_models, thermal_results, thermal_scaler, thermal_test_data = energy_predictor.train_energy_models(
    pd.DataFrame(X_thermal, columns=thermal_features), y_thermal, 'thermal'
)

# Store results in predictor
energy_predictor.thermal_models = thermal_models
energy_predictor.feature_scalers['thermal'] = thermal_scaler
energy_predictor.performance_results['thermal'] = thermal_results

print(f"\n🏆 THERMAL ENERGY MODEL PERFORMANCE:")
print("-" * 40)
for model_name, metrics in thermal_results.items():
    print(f"{model_name:15} | R²: {metrics['r2_score']:.4f} | RMSE: {metrics['rmse']:.1f} kJ/kg | CV: {metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")

# Find best thermal model
best_thermal_model = max(thermal_results.items(), key=lambda x: x[1]['r2_score'])
print(f"\n🥇 Best Thermal Model: {best_thermal_model[0]} (R² = {best_thermal_model[1]['r2_score']:.4f})")

# Create ensemble for thermal energy
print("\n🔗 Creating thermal energy ensemble...")
thermal_X_test, thermal_y_test = thermal_test_data

# Calculate ensemble prediction
ensemble_thermal_pred = np.zeros(len(thermal_X_test))
ensemble_weights = {}
total_weight = 0

for model_name, model in thermal_models.items():
    r2 = thermal_results[model_name]['r2_score']
    weight = max(0, r2) ** 2  # Square to emphasize better models
    ensemble_weights[model_name] = weight
    total_weight += weight

# Normalize weights
for model_name in ensemble_weights.keys():
    ensemble_weights[model_name] /= total_weight

# Generate ensemble predictions
for model_name, model in thermal_models.items():
    if 'regression' in model_name:
        X_test_scaled = thermal_scaler.transform(thermal_X_test)
        pred = model.predict(X_test_scaled)
    else:
        pred = model.predict(thermal_X_test)
    
    ensemble_thermal_pred += pred * ensemble_weights[model_name]
    print(f"  {model_name}: weight = {ensemble_weights[model_name]:.3f}")

# Calculate ensemble performance
thermal_ensemble_r2 = r2_score(thermal_y_test, ensemble_thermal_pred)
thermal_ensemble_rmse = np.sqrt(mean_squared_error(thermal_y_test, ensemble_thermal_pred))

print(f"\n🎯 Thermal Ensemble Performance:")
print(f"  R² Score: {thermal_ensemble_r2:.4f}")
print(f"  RMSE: {thermal_ensemble_rmse:.2f} kJ/kg")

# Store thermal ensemble results
energy_predictor.performance_results['thermal']['ensemble'] = {
    'r2_score': thermal_ensemble_r2,
    'weights': ensemble_weights,
    'rmse': thermal_ensemble_rmse
}
energy_predictor.thermal_features = thermal_features