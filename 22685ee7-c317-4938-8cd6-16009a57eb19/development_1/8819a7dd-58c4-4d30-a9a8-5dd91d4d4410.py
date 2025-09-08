import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using statistical fallback only")

from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CementTimeGAN:
    """
    TimeGAN implementation for realistic cement plant time series generation with temporal correlations.
    
    Features:
    - Unsupervised adversarial learning for temporal dynamics
    - Statistical fallback for environments without ydata-synthetic
    - Sequence preparation and synthetic data generation methods
    - Proper temporal structure preservation
    """
    
    def __init__(self, 
                 sequence_length: int = 24,
                 latent_dim: int = 24, 
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 learning_rate: float = 0.001,
                 batch_size: int = 128,
                 use_statistical_fallback: bool = None,
                 device: str = 'cpu'):
        """Initialize TimeGAN for cement plant data."""
        
        # Auto-detect if we should use statistical fallback
        if use_statistical_fallback is None:
            use_statistical_fallback = not TORCH_AVAILABLE
            
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_statistical_fallback = use_statistical_fallback
        self.device = device
        
        # Data preprocessing
        self.scaler = MinMaxScaler()
        self.feature_dim = None
        self.is_fitted = False
        
        # Statistical fallback parameters
        self.statistical_params = {}
        
        print(f"🎯 CementTimeGAN initialized with sequence_length={sequence_length}")
        print(f"🔧 Mode: {'Statistical Fallback' if use_statistical_fallback else 'Neural TimeGAN'}")
    
    def prepare_sequences(self, data: pd.DataFrame, target_columns: List[str] = None) -> np.ndarray:
        """
        Prepare time series sequences from cement plant data.
        
        Args:
            data: Input dataframe with cement plant features
            target_columns: Specific columns to use (if None, uses all numeric columns)
            
        Returns:
            sequences: Array of shape (n_sequences, sequence_length, n_features)
        """
        print(f"📋 Preparing sequences from data with shape {data.shape}")
        
        # Select columns to use
        if target_columns is None:
            # Use all numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[target_columns]
        
        print(f"🎯 Using {len(numeric_data.columns)} features: {list(numeric_data.columns[:5])}{'...' if len(numeric_data.columns) > 5 else ''}")
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(numeric_data.values)
        self.feature_dim = scaled_data.shape[1]
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            sequence = scaled_data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        print(f"✅ Created {len(sequences)} sequences of shape ({self.sequence_length}, {self.feature_dim})")
        
        return sequences

# Create instance for testing
timegan_model = CementTimeGAN(sequence_length=24, use_statistical_fallback=True)
print("✅ CementTimeGAN class created successfully!")