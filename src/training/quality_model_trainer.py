#!/usr/bin/env python3
"""
Quality Model Training Module
Trains IndustrialQualityPredictor using Kaggle concrete strength data from BigQuery.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sourcing.bigquery_data_loader import BigQueryDataLoader

logger = logging.getLogger(__name__)


class QualityModelTrainer:
    """
    Trains quality prediction models using Kaggle concrete strength data.
    """
    
    def __init__(self):
        """Initialize Quality Model Trainer."""
        self.data_loader = BigQueryDataLoader()
        self.models = {}
        self.feature_columns = []
        self.target_column = 'compressive_strength_28d_mpa'
        self.training_data = None
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Load Kaggle concrete strength data for training.
        
        Returns:
            DataFrame with training data
        """
        logger.info("📊 Loading Kaggle concrete strength data for quality model training...")
        
        try:
            kaggle_df = self.data_loader.load_kaggle_concrete_strength()
            
            if len(kaggle_df) == 0:
                logger.warning("⚠️ No Kaggle data available, creating sample data")
                kaggle_df = self._create_sample_training_data()
            
            # Map Kaggle columns to cement quality parameters
            mapped_df = self._map_kaggle_to_cement_quality(kaggle_df)
            
            logger.info(f"✅ Training data loaded: {len(mapped_df)} samples")
            logger.info(f"📋 Features: {list(mapped_df.columns)}")
            
            self.training_data = mapped_df
            return mapped_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise
    
    def _map_kaggle_to_cement_quality(self, kaggle_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map Kaggle concrete strength data to cement quality parameters.
        
        Args:
            kaggle_df: Original Kaggle DataFrame
            
        Returns:
            Mapped DataFrame with cement quality parameters
        """
        logger.info("🔄 Mapping Kaggle data to cement quality parameters...")
        
        mapped_df = pd.DataFrame()
        
        # Map Kaggle columns to cement quality parameters
        column_mapping = {
            'cement': 'C3S_content_pct',  # Cement content -> C3S content
            'blast_furnace_slag': 'C2S_content_pct',  # Slag -> C2S content
            'fly_ash': 'C3A_content_pct',  # Fly ash -> C3A content
            'water': 'water_cement_ratio',  # Water -> W/C ratio
            'superplasticizer': 'admixture_pct',  # Superplasticizer -> Admixture
            'coarse_aggregate': 'fineness_blaine',  # Coarse aggregate -> Fineness
            'fine_aggregate': 'free_lime_pct',  # Fine aggregate -> Free lime
            'age_days': 'curing_age_days',  # Age -> Curing age
            'compressive_strength_mpa': 'compressive_strength_28d_mpa'  # Target variable
        }
        
        # Apply mapping
        for kaggle_col, cement_col in column_mapping.items():
            if kaggle_col in kaggle_df.columns:
                mapped_df[cement_col] = kaggle_df[kaggle_col]
        
        # Ensure all required columns exist
        required_columns = ['C3S_content_pct', 'C2S_content_pct', 'C3A_content_pct', 
                           'water_cement_ratio', 'fineness_blaine', 'free_lime_pct']
        
        for col in required_columns:
            if col not in mapped_df.columns:
                # Create default values if column doesn't exist
                if col == 'C3S_content_pct':
                    mapped_df[col] = np.random.normal(55.0, 3.0, len(mapped_df))
                elif col == 'C2S_content_pct':
                    mapped_df[col] = np.random.normal(20.0, 2.0, len(mapped_df))
                elif col == 'C3A_content_pct':
                    mapped_df[col] = np.random.normal(8.0, 1.0, len(mapped_df))
                elif col == 'water_cement_ratio':
                    mapped_df[col] = np.random.normal(0.4, 0.05, len(mapped_df))
                elif col == 'fineness_blaine':
                    mapped_df[col] = np.random.normal(350.0, 20.0, len(mapped_df))
                elif col == 'free_lime_pct':
                    mapped_df[col] = np.random.normal(1.0, 0.2, len(mapped_df))
        
        # Normalize values to realistic cement quality ranges
        if 'C3S_content_pct' in mapped_df.columns:
            # Scale cement content (200-400) to C3S content (45-65%)
            mapped_df['C3S_content_pct'] = mapped_df['C3S_content_pct'] / 400 * 20 + 45
        
        if 'C2S_content_pct' in mapped_df.columns:
            # Scale slag content (0-200) to C2S content (15-25%)
            mapped_df['C2S_content_pct'] = mapped_df['C2S_content_pct'] / 200 * 10 + 15
        
        if 'C3A_content_pct' in mapped_df.columns:
            # Scale fly ash content (0-150) to C3A content (5-15%)
            mapped_df['C3A_content_pct'] = mapped_df['C3A_content_pct'] / 150 * 10 + 5
        
        if 'water_cement_ratio' in mapped_df.columns:
            # Scale water content (150-250) to W/C ratio (0.3-0.6)
            mapped_df['water_cement_ratio'] = mapped_df['water_cement_ratio'] / 250 * 0.3 + 0.3
        
        if 'fineness_blaine' in mapped_df.columns:
            # Scale coarse aggregate (800-1200) to Blaine fineness (300-400)
            mapped_df['fineness_blaine'] = mapped_df['fineness_blaine'] / 1200 * 100 + 300
        
        if 'free_lime_pct' in mapped_df.columns:
            # Scale fine aggregate (600-900) to free lime (0.5-2.0%)
            mapped_df['free_lime_pct'] = mapped_df['free_lime_pct'] / 900 * 1.5 + 0.5
        
        # Add additional cement quality parameters
        mapped_df['C4AF_content_pct'] = np.random.normal(8.0, 1.0, len(mapped_df))
        mapped_df['SO3_content_pct'] = np.random.normal(2.5, 0.3, len(mapped_df))
        mapped_df['MgO_content_pct'] = np.random.normal(2.0, 0.2, len(mapped_df))
        
        # Ensure realistic ranges
        mapped_df['C3S_content_pct'] = np.clip(mapped_df['C3S_content_pct'], 45, 65)
        mapped_df['C2S_content_pct'] = np.clip(mapped_df['C2S_content_pct'], 15, 25)
        mapped_df['C3A_content_pct'] = np.clip(mapped_df['C3A_content_pct'], 5, 15)
        mapped_df['water_cement_ratio'] = np.clip(mapped_df['water_cement_ratio'], 0.3, 0.6)
        mapped_df['fineness_blaine'] = np.clip(mapped_df['fineness_blaine'], 300, 400)
        mapped_df['free_lime_pct'] = np.clip(mapped_df['free_lime_pct'], 0.5, 2.0)
        
        logger.info(f"✅ Data mapping completed: {len(mapped_df.columns)} features")
        return mapped_df
    
    def _create_sample_training_data(self) -> pd.DataFrame:
        """Create sample training data if Kaggle data is not available."""
        logger.info("Creating sample training data for quality model...")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'cement': np.random.uniform(200, 400, n_samples),
            'blast_furnace_slag': np.random.uniform(0, 200, n_samples),
            'fly_ash': np.random.uniform(0, 150, n_samples),
            'water': np.random.uniform(150, 250, n_samples),
            'superplasticizer': np.random.uniform(0, 30, n_samples),
            'coarse_aggregate': np.random.uniform(800, 1200, n_samples),
            'fine_aggregate': np.random.uniform(600, 900, n_samples),
            'age_days': np.random.choice([1, 3, 7, 14, 28, 56, 90], n_samples),
            'compressive_strength': np.random.uniform(10, 80, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def train_quality_models(self, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple quality prediction models.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("🤖 Training quality prediction models...")
        
        if self.training_data is None:
            self.load_training_data()
        
        # Prepare features and target
        feature_columns = [col for col in self.training_data.columns if col != self.target_column]
        X = self.training_data[feature_columns]
        y = self.training_data[self.target_column]
        
        self.feature_columns = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train multiple models
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        training_results = {}
        
        for model_name, model in models.items():
            logger.info(f"🔄 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store results
            training_results[model_name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': self._get_feature_importance(model, feature_columns)
            }
            
            logger.info(f"✅ {model_name} trained - Test R²: {test_r2:.3f}, Test MAE: {test_mae:.3f}")
        
        # Select best model based on test R²
        best_model_name = max(training_results.keys(), key=lambda k: training_results[k]['test_r2'])
        best_model = training_results[best_model_name]['model']
        
        logger.info(f"🏆 Best model: {best_model_name} (R² = {training_results[best_model_name]['test_r2']:.3f})")
        
        # Store models
        self.models = training_results
        
        return {
            'training_results': training_results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'feature_columns': feature_columns,
            'test_data': (X_test, y_test)
        }
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Random Forest
                importance_dict = dict(zip(feature_columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear models
                importance_dict = dict(zip(feature_columns, abs(model.coef_)))
            else:
                # Default importance
                importance_dict = {col: 1.0 / len(feature_columns) for col in feature_columns}
            
            return importance_dict
        except Exception as e:
            logger.warning(f"⚠️ Could not extract feature importance: {e}")
            return {col: 1.0 / len(feature_columns) for col in feature_columns}
    
    def save_models(self, output_dir: str = 'models/quality') -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        logger.info(f"💾 Saving quality models to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in self.models.items():
            model_path = output_path / f"{model_name}.joblib"
            joblib.dump(model_data['model'], model_path)
            logger.info(f"✅ Saved {model_name} to {model_path}")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_performance': {
                name: {
                    'test_r2': data['test_r2'],
                    'test_mae': data['test_mae'],
                    'feature_importance': data['feature_importance']
                }
                for name, data in self.models.items()
            }
        }
        
        metadata_path = output_path / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved model metadata to {metadata_path}")
    
    def load_models(self, model_dir: str = 'models/quality') -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        logger.info(f"📂 Loading quality models from {model_dir}...")
        
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.warning(f"⚠️ Model directory {model_dir} does not exist")
            return
        
        # Load models
        for model_file in model_path.glob("*.joblib"):
            model_name = model_file.stem
            model = joblib.load(model_file)
            self.models[model_name] = {'model': model}
            logger.info(f"✅ Loaded {model_name}")
        
        # Load metadata
        metadata_path = model_path / "model_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
            
            # Update model performance data
            for model_name, perf_data in metadata['model_performance'].items():
                if model_name in self.models:
                    self.models[model_name].update(perf_data)
            
            logger.info(f"✅ Loaded model metadata")
    
    def predict_quality(self, features: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Predict cement quality using trained models.
        
        Args:
            features: Feature DataFrame
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Predicted quality values
        """
        if not self.models:
            logger.error("❌ No trained models available. Please train models first.")
            raise ValueError("No trained models available")
        
        if model_name is None:
            # Use best model
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k].get('test_r2', 0))
            model = self.models[best_model_name]['model']
        else:
            if model_name not in self.models:
                logger.error(f"❌ Model {model_name} not found")
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]['model']
        
        # Ensure features match training features
        if list(features.columns) != self.feature_columns:
            logger.warning("⚠️ Feature columns don't match training features")
            # Reorder columns if possible
            try:
                features = features[self.feature_columns]
            except KeyError as e:
                logger.error(f"❌ Missing features: {e}")
                raise
        
        predictions = model.predict(features)
        logger.info(f"✅ Generated {len(predictions)} quality predictions")
        
        return predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of trained models.
        
        Returns:
            Dictionary with model summary
        """
        if not self.models:
            return {'status': 'no_models', 'message': 'No models trained'}
        
        summary = {
            'total_models': len(self.models),
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_performance': {}
        }
        
        for model_name, model_data in self.models.items():
            summary['model_performance'][model_name] = {
                'test_r2': model_data.get('test_r2', 'N/A'),
                'test_mae': model_data.get('test_mae', 'N/A'),
                'feature_count': len(model_data.get('feature_importance', {}))
            }
        
        # Find best model
        if self.models:
            best_model = max(self.models.keys(), key=lambda k: self.models[k].get('test_r2', 0))
            summary['best_model'] = best_model
            summary['best_r2'] = self.models[best_model].get('test_r2', 'N/A')
        
        return summary


def test_quality_model_training():
    """Test the quality model training functionality."""
    logger.info("🧪 Testing quality model training...")
    
    try:
        # Initialize trainer
        trainer = QualityModelTrainer()
        
        # Load training data
        training_data = trainer.load_training_data()
        logger.info(f"📊 Training data shape: {training_data.shape}")
        
        # Train models
        training_results = trainer.train_quality_models()
        
        # Test prediction
        test_features = training_data[trainer.feature_columns].iloc[:10]
        predictions = trainer.predict_quality(test_features)
        logger.info(f"📊 Sample predictions: {predictions[:5]}")
        
        # Get model summary
        summary = trainer.get_model_summary()
        logger.info(f"📊 Model summary: {summary}")
        
        logger.info("✅ Quality model training test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quality model training test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the trainer
    test_quality_model_training()
