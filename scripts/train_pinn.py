"""
PINN Training for Quality Prediction Model
Trains Physics-Informed Neural Network for free lime prediction using synthetic and real data
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network for cement quality prediction
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], output_dim: int = 1):
        super(PINNModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Network
    """
    
    def __init__(self, epochs: int = 200, batch_size: int = 64, learning_rate: float = 0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
        
    def prepare_data(self, df: pd.DataFrame, input_columns: List[str], output_columns: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for PINN training
        
        Args:
            df: Input DataFrame
            input_columns: List of input feature columns
            output_columns: List of output target columns
            
        Returns:
            Tuple of (X, y) tensors
        """
        logger.info("🔄 Preparing data for PINN training...")
        
        # Select and clean data
        X = df[input_columns].copy()
        y = df[output_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values)
        
        logger.info(f"✅ Data prepared: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} features")
        
        return X_tensor, y_tensor
    
    def train(self, X: torch.Tensor, y: torch.Tensor) -> PINNModel:
        """
        Train PINN model
        
        Args:
            X: Input features tensor
            y: Target values tensor
            
        Returns:
            Trained PINN model
        """
        logger.info("🔄 Training PINN model...")
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        self.model = PINNModel(input_dim=input_dim, output_dim=output_dim)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Training loop
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            # Record training history
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss / (len(X_train) // self.batch_size + 1),
                'val_loss': val_loss
            }
            self.training_history.append(epoch_history)
            
            # Log progress
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: Train Loss = {epoch_history['train_loss']:.6f}, Val Loss = {val_loss:.6f}")
        
        logger.info("✅ PINN model training completed")
        return self.model
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate trained model
        
        Args:
            X: Input features tensor
            y: Target values tensor
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("🔄 Evaluating PINN model...")
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            predictions_np = predictions.numpy()
            y_np = y.numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_np, predictions_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_np, predictions_np)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
        
        logger.info(f"✅ Model evaluation completed: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        return metrics
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Input features tensor
            
        Returns:
            Predictions array
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.numpy()
    
    def save_model(self, filepath: str):
        """
        Save trained model
        
        Args:
            filepath: Path to save model
        """
        logger.info(f"🔄 Saving PINN model to {filepath}...")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'training_history': self.training_history,
            'model_config': {
                'input_dim': self.model.network[0].in_features,
                'output_dim': self.model.network[-1].out_features
            }
        }
        
        torch.save(model_state, filepath)
        logger.info(f"✅ Model saved successfully")

def train_pinn_model() -> PINNModel:
    """
    Train PINN model for free lime prediction
    
    Returns:
        Trained PINN model
    """
    logger.info("🔄 Starting PINN model training for free lime prediction...")
    
    try:
        # Load datasets
        logger.info("📊 Loading datasets...")
        df_real = load_real_quality_data()
        df_synth = load_synthetic_quality_data()
        
        # Combine datasets
        logger.info("🔄 Combining real and synthetic data...")
        df_combined = combine_quality_datasets(df_real, df_synth)
        
        # Prepare training data
        input_columns = [
            'cement_kg_m3', 'blast_furnace_slag_kg_m3', 'fly_ash_kg_m3', 
            'water_kg_m3', 'superplasticizer_kg_m3', 'coarse_aggregate_kg_m3', 
            'fine_aggregate_kg_m3', 'age_days', 'compressive_strength_mpa'
        ]
        output_columns = ['free_lime_percent']
        
        # Filter available columns
        available_inputs = [col for col in input_columns if col in df_combined.columns]
        
        if not available_inputs:
            logger.warning("⚠️ No input columns found, using default features")
            available_inputs = ['compressive_strength_mpa', 'age_days']
        
        # Initialize trainer
        trainer = PINNTrainer(epochs=200, batch_size=64)
        
        # Prepare data
        X, y = trainer.prepare_data(df_combined, available_inputs, output_columns)
        
        # Train model
        model = trainer.train(X, y)
        
        # Evaluate model
        metrics = trainer.evaluate(X, y)
        
        # Save model
        os.makedirs("demo/models/pinn", exist_ok=True)
        model_path = "demo/models/pinn/free_lime_pinn.pt"
        trainer.save_model(model_path)
        
        # Save training results
        save_training_results(trainer, metrics)
        
        logger.info("✅ PINN model training completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ PINN model training failed: {e}")
        raise

def load_real_quality_data() -> pd.DataFrame:
    """
    Load real quality data
    
    Returns:
        Real quality DataFrame
    """
    quality_path = "demo/data/real/quality_parameters.csv"
    
    if os.path.exists(quality_path):
        df = pd.read_csv(quality_path)
        logger.info(f"✅ Loaded real quality data: {len(df)} rows")
        return df
    else:
        logger.warning("⚠️ Real quality data not found, creating sample data")
        return create_sample_quality_data()

def load_synthetic_quality_data() -> pd.DataFrame:
    """
    Load synthetic quality data
    
    Returns:
        Synthetic quality DataFrame
    """
    quality_path = "demo/data/synthetic/quality_parameters.csv"
    
    if os.path.exists(quality_path):
        df = pd.read_csv(quality_path)
        logger.info(f"✅ Loaded synthetic quality data: {len(df)} rows")
        return df
    else:
        logger.warning("⚠️ Synthetic quality data not found, creating sample data")
        return create_sample_quality_data(n_samples=1000)

def combine_quality_datasets(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    """
    Combine real and synthetic quality datasets
    
    Args:
        df_real: Real quality DataFrame
        df_synth: Synthetic quality DataFrame
        
    Returns:
        Combined DataFrame
    """
    logger.info("🔄 Combining quality datasets...")
    
    # Add data source column
    df_real['data_source'] = 'real'
    df_synth['data_source'] = 'synthetic'
    
    # Combine datasets
    df_combined = pd.concat([df_real, df_synth], ignore_index=True)
    
    logger.info(f"✅ Combined dataset created: {len(df_combined)} records")
    
    return df_combined

def create_sample_quality_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample quality data for training
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        Sample quality DataFrame
    """
    logger.info(f"🔄 Creating sample quality data ({n_samples} samples)...")
    
    data = []
    
    for i in range(n_samples):
        # Generate realistic cement quality parameters
        cement_content = np.random.normal(300, 50)
        blast_furnace_slag = np.random.normal(100, 20)
        fly_ash = np.random.normal(50, 15)
        water_content = np.random.normal(180, 30)
        superplasticizer = np.random.normal(5, 2)
        coarse_aggregate = np.random.normal(1000, 100)
        fine_aggregate = np.random.normal(800, 80)
        age_days = np.random.choice([7, 14, 28, 56, 90])
        
        # Calculate compressive strength based on mix proportions
        compressive_strength = (
            30 + 
            cement_content * 0.05 + 
            blast_furnace_slag * 0.02 + 
            fly_ash * 0.03 - 
            water_content * 0.1 + 
            superplasticizer * 2 +
            np.random.normal(0, 5)
        )
        
        # Calculate free lime based on compressive strength and age
        free_lime = max(0.5, 3.0 - (compressive_strength - 30) * 0.02 - age_days * 0.01)
        
        data.append({
            'timestamp': datetime.now().isoformat(),
            'cement_kg_m3': cement_content,
            'blast_furnace_slag_kg_m3': blast_furnace_slag,
            'fly_ash_kg_m3': fly_ash,
            'water_kg_m3': water_content,
            'superplasticizer_kg_m3': superplasticizer,
            'coarse_aggregate_kg_m3': coarse_aggregate,
            'fine_aggregate_kg_m3': fine_aggregate,
            'age_days': age_days,
            'compressive_strength_mpa': compressive_strength,
            'free_lime_percent': free_lime,
            'free_lime_pct': free_lime  # Duplicate for compatibility
        })
    
    return pd.DataFrame(data)

def save_training_results(trainer: PINNTrainer, metrics: Dict[str, float]):
    """
    Save training results and metrics
    
    Args:
        trainer: Trained PINN trainer
        metrics: Evaluation metrics
    """
    logger.info("🔄 Saving training results...")
    
    results = {
        'training_history': trainer.training_history,
        'evaluation_metrics': metrics,
        'model_config': {
            'epochs': trainer.epochs,
            'batch_size': trainer.batch_size,
            'learning_rate': trainer.learning_rate
        },
        'training_timestamp': datetime.now().isoformat()
    }
    
    # Save results
    os.makedirs("demo/models/pinn", exist_ok=True)
    results_path = "demo/models/pinn/training_results.json"
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Training results saved to {results_path}")

def validate_pinn_model(model: PINNModel, test_data: pd.DataFrame) -> bool:
    """
    Validate PINN model performance
    
    Args:
        model: Trained PINN model
        test_data: Test data DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("🔄 Validating PINN model...")
    
    try:
        # Create sample test data
        if len(test_data) == 0:
            test_data = create_sample_quality_data(50)
        
        # Prepare test data
        input_columns = ['compressive_strength_mpa', 'age_days']
        output_columns = ['free_lime_percent']
        
        available_inputs = [col for col in input_columns if col in test_data.columns]
        
        if not available_inputs:
            logger.error("❌ No input columns available for validation")
            return False
        
        # Create trainer for validation
        trainer = PINNTrainer()
        trainer.model = model
        
        # Prepare test data
        X_test, y_test = trainer.prepare_data(test_data, available_inputs, output_columns)
        
        # Evaluate model
        metrics = trainer.evaluate(X_test, y_test)
        
        # Check if model performance is acceptable
        if metrics['r2_score'] > 0.5 and metrics['rmse'] < 2.0:
            logger.info("✅ PINN model validation passed")
            return True
        else:
            logger.warning(f"⚠️ PINN model performance below threshold: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.4f}")
            return False
            
    except Exception as e:
        logger.error(f"❌ PINN model validation failed: {e}")
        return False

def get_pinn_summary(model: PINNModel, trainer: PINNTrainer) -> Dict[str, Any]:
    """
    Get PINN model summary
    
    Args:
        model: Trained PINN model
        trainer: PINN trainer
        
    Returns:
        Model summary dictionary
    """
    summary = {
        'model_type': 'Physics-Informed Neural Network',
        'input_dim': model.network[0].in_features,
        'output_dim': model.network[-1].out_features,
        'hidden_layers': len([layer for layer in model.network if isinstance(layer, nn.Linear)]) - 1,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_epochs': len(trainer.training_history),
        'final_train_loss': trainer.training_history[-1]['train_loss'] if trainer.training_history else 'Unknown',
        'final_val_loss': trainer.training_history[-1]['val_loss'] if trainer.training_history else 'Unknown',
        'training_timestamp': datetime.now().isoformat()
    }
    
    return summary

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Train PINN model
        model = train_pinn_model()
        
        # Get model summary
        trainer = PINNTrainer()
        summary = get_pinn_summary(model, trainer)
        
        logger.info(f"📊 PINN Model Summary:")
        logger.info(f"   Model Type: {summary['model_type']}")
        logger.info(f"   Input Dimension: {summary['input_dim']}")
        logger.info(f"   Output Dimension: {summary['output_dim']}")
        logger.info(f"   Hidden Layers: {summary['hidden_layers']}")
        logger.info(f"   Total Parameters: {summary['total_parameters']}")
        logger.info(f"   Training Epochs: {summary['training_epochs']}")
        
        logger.info("✅ PINN model training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ PINN model training failed: {e}")
        raise
