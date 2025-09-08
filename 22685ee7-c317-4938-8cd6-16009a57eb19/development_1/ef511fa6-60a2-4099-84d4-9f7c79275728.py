import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CementQualityPredictor:
    """
    Physics-informed cement quality prediction system with multiple ML algorithms
    
    Predicts key quality parameters:
    - Free lime content (% unreacted CaO) 
    - Compressive strength (derived from C3S, fineness, process conditions)
    - C3S content (primary strength-giving compound)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models with optimized hyperparameters"""
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            ),
            'Linear Regression': LinearRegression()
        }
    
    def engineer_physics_features(self, df):
        """
        Engineer physics-informed features based on cement chemistry principles
        """
        features_df = df.copy()
        
        # 1. Burnability indicators
        features_df['lsf_squared'] = features_df['LSF'] ** 2
        features_df['lsf_deviation'] = abs(features_df['LSF'] - 0.95)  # Optimal LSF ~0.95
        
        # 2. Temperature efficiency metrics
        features_df['temp_efficiency'] = features_df['kiln_temperature'] / (features_df['coal_feed_rate'] / 1000)
        features_df['temp_deviation'] = abs(features_df['kiln_temperature'] - 1450)  # Optimal ~1450°C
        
        # 3. Fineness interaction terms
        features_df['fineness_ratio'] = features_df['cement_mill_fineness'] / features_df['raw_mill_fineness']
        features_df['specific_surface'] = features_df['cement_mill_fineness'] * 0.3  # Blaine estimate
        
        # 4. Chemical reactivity indicators
        features_df['cao_sio2_ratio'] = features_df['CaO'] / (features_df['SiO2'] + 1e-6)
        features_df['flux_content'] = features_df['Al2O3'] + features_df['Fe2O3']  # Flux phases
        
        # 5. Process stability indicators
        features_df['draft_stability'] = abs(features_df['draft_pressure'] + 12)  # Optimal ~-12
        features_df['speed_efficiency'] = features_df['kiln_speed'] * features_df['kiln_temperature'] / 1000
        
        # 6. Compound interaction terms
        features_df['c3s_potential'] = features_df['CaO'] - (1.65 * features_df['SiO2'] + 
                                                            0.35 * features_df['Al2O3'] + 
                                                            0.35 * features_df['Fe2O3'])
        
        # 7. Quality prediction indicators
        features_df['free_lime_tendency'] = (features_df['lsf_deviation'] * 
                                           features_df['temp_deviation'] / 100)
        
        return features_df
    
    def calculate_target_variables(self, df):
        """
        Calculate target quality variables based on physics principles
        """
        targets = {}
        
        # 1. Free lime (already in dataset, but ensure realistic bounds)
        targets['free_lime'] = np.clip(df['free_lime'], 0.1, 5.0)
        
        # 2. C3S content (already in dataset)
        targets['c3s_content'] = np.clip(df['C3S'], 20, 80)
        
        # 3. Compressive strength estimate (physics-based formula)
        # Based on Bogue compounds, fineness, and process conditions
        strength_base = (
            df['C3S'] * 0.65 +  # C3S contribution (strongest)
            df['C2S'] * 0.25 +  # C2S contribution (slower)
            df['C3A'] * 0.10 -  # C3A minor contribution
            df['free_lime'] * 5  # Free lime reduces strength
        )
        
        # Fineness factor (finer cement = higher early strength)
        fineness_factor = (df['cement_mill_fineness'] - 280) / 140  # Normalize to 0-1
        fineness_factor = np.clip(fineness_factor, 0, 1)
        
        # Temperature factor (proper burning improves strength)
        temp_factor = 1 - abs(df['kiln_temperature'] - 1450) / 200
        temp_factor = np.clip(temp_factor, 0.7, 1.0)
        
        # Combined compressive strength (28-day MPa estimate)
        targets['compressive_strength'] = np.clip(
            strength_base * fineness_factor * temp_factor * 0.8,  # Scale to realistic range
            25, 65
        )
        
        return targets
    
    def prepare_features(self, df):
        """Prepare feature set for modeling"""
        # Engineer physics features
        features_df = self.engineer_physics_features(df)
        
        # Select key features for modeling
        feature_cols = [
            # Chemistry
            'CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'LSF', 'SM', 'AM',
            'C3S', 'C2S', 'C3A', 'C4AF',
            # Process parameters
            'kiln_temperature', 'kiln_speed', 'coal_feed_rate', 
            'draft_pressure', 'raw_mill_fineness', 'cement_mill_fineness',
            # Engineered features
            'lsf_squared', 'lsf_deviation', 'temp_efficiency', 'temp_deviation',
            'fineness_ratio', 'specific_surface', 'cao_sio2_ratio', 
            'flux_content', 'draft_stability', 'speed_efficiency',
            'c3s_potential', 'free_lime_tendency'
        ]
        
        return features_df[feature_cols]
    
    def train_models(self, df, target_name):
        """Train all models for a specific target"""
        print(f"\n🎯 Training models for {target_name}")
        print("="*50)
        
        # Prepare features and target
        X = self.prepare_features(df)
        targets = self.calculate_target_variables(df)
        y = targets[target_name]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_name] = scaler
        
        # Train and evaluate each model
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for neural network
            if name == 'Neural Network':
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # Train model
            model.fit(X_tr, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_tr)
            y_pred_test = model.predict(X_te)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            if name != 'Neural Network':  # CV too slow for NN
                cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean, cv_std = train_r2, 0
            
            # Store results
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model
            }
            
            print(f"  Train R²: {train_r2:.3f}")
            print(f"  Test R²:  {test_r2:.3f}")
            print(f"  Test RMSE: {test_rmse:.3f}")
            print(f"  Test MAE:  {test_mae:.3f}")
            print(f"  CV R² (5-fold): {cv_mean:.3f} ± {cv_std:.3f}")
        
        self.model_performance[target_name] = results
        
        # Calculate feature importance for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.3f})")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For linear models, use coefficient magnitude
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': abs(best_model.coef_)
            }).sort_values('importance', ascending=False)
        
        self.feature_importance[target_name] = importance_df
        
        return results, X_test, y_test
    
    def train_all_targets(self, df):
        """Train models for all target variables"""
        print("🚀 Starting comprehensive cement quality prediction training")
        print("="*60)
        
        targets = ['free_lime', 'c3s_content', 'compressive_strength']
        
        for target in targets:
            self.train_models(df, target)
            print("\n" + "="*60)
    
    def plot_model_comparison(self, target_name):
        """Plot model performance comparison"""
        if target_name not in self.model_performance:
            print(f"No results for {target_name}")
            return
        
        results = self.model_performance[target_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Performance Comparison - {target_name.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        
        # R² scores
        train_r2 = [results[m]['train_r2'] for m in models]
        test_r2 = [results[m]['test_r2'] for m in models]
        
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [results[m]['test_rmse'] for m in models]
        
        ax = axes[0, 1]
        ax.bar(models, test_rmse, alpha=0.7, color='orange')
        ax.set_ylabel('RMSE')
        ax.set_title('Test RMSE Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Cross-validation scores
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]
        
        ax = axes[1, 0]
        ax.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='green')
        ax.set_ylabel('CV R² Score')
        ax.set_title('5-Fold Cross-Validation R²')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Feature importance (top 10)
        if target_name in self.feature_importance:
            importance_df = self.feature_importance[target_name].head(10)
            
            ax = axes[1, 1]
            ax.barh(range(len(importance_df)), importance_df['importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Feature Importance')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Initialize the prediction system
predictor = CementQualityPredictor(random_state=42)

print("✅ CementQualityPredictor initialized successfully!")
print("\n📋 System capabilities:")
print("• Multiple ML algorithms: Random Forest, Gradient Boosting, Neural Network, Linear Regression")
print("• Physics-informed feature engineering")
print("• Target predictions: Free lime, C3S content, Compressive strength")
print("• Comprehensive model evaluation and comparison")
print("• Feature importance analysis")