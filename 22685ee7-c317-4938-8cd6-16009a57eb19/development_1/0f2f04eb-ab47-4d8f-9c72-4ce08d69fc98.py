import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class CementEnergyPredictor:
    """Comprehensive energy consumption prediction for thermal and electrical energy"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.thermal_models = {}
        self.electrical_models = {}
        self.feature_scalers = {}
        self.performance_results = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5, 
                min_samples_leaf=2, random_state=random_state
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, 
                min_samples_split=5, random_state=random_state
            ),
            'ridge_regression': Ridge(alpha=1.0),
            'linear_regression': LinearRegression()
        }
    
    def engineer_energy_features(self, df):
        """Advanced feature engineering for energy consumption prediction"""
        print("🔧 Engineering energy consumption features...")
        
        energy_df = df.copy()
        
        # 1. Fuel Efficiency Features
        energy_df['heat_per_c3s'] = energy_df['heat_consumption'] / (energy_df['C3S'] + 1e-6)
        energy_df['burnability_efficiency'] = energy_df['burnability_index'] / (energy_df['heat_consumption'] + 1e-6)
        
        # Temperature efficiency
        target_temp = 1450
        energy_df['temp_deviation'] = abs(energy_df['kiln_temperature'] - target_temp)
        energy_df['temp_efficiency'] = 1 / (1 + energy_df['temp_deviation'] / target_temp)
        
        # 2. Production Rate Features
        energy_df['speed_heat_ratio'] = energy_df['kiln_speed'] / (energy_df['heat_consumption'] + 1e-6)
        energy_df['coal_efficiency'] = energy_df['coal_feed_rate'] / (energy_df['heat_consumption'] + 1e-6)
        energy_df['production_intensity'] = energy_df['kiln_speed'] * energy_df['kiln_temperature'] / 1000
        
        # 3. Temperature Deviation Features
        energy_df['temp_squared_dev'] = energy_df['temp_deviation'] ** 2
        energy_df['temp_stress_factor'] = energy_df['temp_deviation'] / (energy_df['kiln_speed'] + 1e-6)
        
        # 4. Chemical Interaction Terms
        energy_df['lsf_energy_factor'] = energy_df['LSF'] * energy_df['heat_consumption']
        energy_df['sm_temp_interaction'] = energy_df['SM'] * energy_df['kiln_temperature']
        energy_df['free_lime_penalty'] = energy_df['free_lime'] * energy_df['heat_consumption']
        
        # 5. Mill Energy Features
        energy_df['raw_mill_energy'] = energy_df['raw_mill_fineness'] * 0.8  # kWh/t estimate
        energy_df['cement_mill_energy'] = energy_df['cement_mill_fineness'] * 0.15
        energy_df['total_grinding_energy'] = energy_df['raw_mill_energy'] + energy_df['cement_mill_energy']
        
        # 6. Draft Pressure Energy Impact
        energy_df['draft_energy_loss'] = abs(energy_df['draft_pressure']) * 2
        
        # 7. Thermal Energy (kJ/kg clinker) - Target 1
        energy_df['thermal_energy_kjkg'] = (
            energy_df['heat_consumption'] * 4.184 *  # Convert kcal to kJ
            (1 + energy_df['temp_deviation'] / target_temp * 0.1) *
            (1 + energy_df['free_lime'] * 0.05) *
            (2.0 - energy_df['burnability_index'] / 100)
        )
        
        # 8. Electrical Energy (kWh/t cement) - Target 2
        base_electrical = 35
        energy_df['electrical_energy_kwht'] = (
            base_electrical +
            energy_df['total_grinding_energy'] +
            energy_df['draft_energy_loss'] +
            energy_df['kiln_speed'] * 3 +
            energy_df['coal_feed_rate'] * 0.002 +
            (energy_df['kiln_temperature'] - 1400) * 0.01
        )
        
        # 9. Energy Ratio Features
        energy_df['thermal_electrical_ratio'] = (
            energy_df['thermal_energy_kjkg'] / (energy_df['electrical_energy_kwht'] + 1e-6)
        )
        
        # 10. Efficiency Indicators
        energy_df['overall_efficiency'] = (
            (energy_df['C3S'] + energy_df['C2S']) / 
            (energy_df['thermal_energy_kjkg'] / 1000 + energy_df['electrical_energy_kwht'] + 1e-6)
        )
        
        print(f"✅ Generated {len(energy_df.columns) - len(df.columns)} new energy features")
        
        return energy_df

    def select_energy_features(self, X, y, target_type='thermal', n_features=25):
        """Select most relevant features for energy prediction"""
        print(f"🎯 Selecting top {n_features} features for {target_type} energy prediction...")
        
        # Remove infinite and NaN values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Use SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X_clean.shape[1]))
        X_selected = selector.fit_transform(X_clean, y)
        
        # Get selected feature names
        feature_names = X.columns[selector.get_support()].tolist()
        
        print(f"✅ Selected features: {feature_names[:10]}...")  # Show first 10
        
        return X_selected, feature_names, selector

    def train_energy_models(self, X, y, target_type='thermal'):
        """Train multiple models for energy prediction"""
        print(f"🤖 Training {target_type} energy prediction models...")
        
        models = {}
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, model in self.model_configs.items():
            print(f"  Training {model_name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'regression' in model_name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                X_cv = X_train_scaled
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_cv = X_train
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='r2')
            
            models[model_name] = model
            results[model_name] = {
                'r2_score': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"    R² Score: {r2:.4f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    Cross-val R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return models, results, scaler, (X_test, y_test)

# Initialize predictor
print("🎯 Initializing CementEnergyPredictor...")
energy_predictor = CementEnergyPredictor(random_state=42)

print("🔧 Engineering features for cement dataset...")
processed_data = energy_predictor.engineer_energy_features(cement_dataset)

print(f"📊 Processed dataset shape: {processed_data.shape}")
print(f"📈 Energy targets created:")
print(f"  - Thermal energy range: {processed_data['thermal_energy_kjkg'].min():.1f} - {processed_data['thermal_energy_kjkg'].max():.1f} kJ/kg")
print(f"  - Electrical energy range: {processed_data['electrical_energy_kwht'].min():.1f} - {processed_data['electrical_energy_kwht'].max():.1f} kWh/t")

# Display feature engineering summary
new_features = [col for col in processed_data.columns if col not in cement_dataset.columns]
print(f"\n🔧 Engineered Features ({len(new_features)}):")
for i, feature in enumerate(new_features):
    if i < 10:  # Show first 10
        print(f"  - {feature}")
    elif i == 10:
        print(f"  ... and {len(new_features) - 10} more")
        break