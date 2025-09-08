import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class CementTimeGAN:
    """
    TimeGAN implementation for realistic cement plant time series generation.
    Includes statistical fallback when PyTorch is not available.
    """
    
    def __init__(self, sequence_length: int = 24, use_statistical_fallback: bool = None):
        # Auto-detect if we should use statistical fallback
        if use_statistical_fallback is None:
            use_statistical_fallback = not TORCH_AVAILABLE
            
        self.sequence_length = sequence_length
        self.use_statistical_fallback = use_statistical_fallback
        self.scaler = MinMaxScaler()
        self.feature_dim = None
        self.is_fitted = False
        self.statistical_params = {}
        
        print(f"🎯 CementTimeGAN initialized (sequence_length={sequence_length})")
        print(f"🔧 Mode: {'Statistical Fallback' if use_statistical_fallback else 'Neural TimeGAN'}")
    
    def prepare_sequences(self, data: pd.DataFrame, target_columns: List[str] = None) -> np.ndarray:
        """Prepare time series sequences from cement plant data."""
        print(f"📋 Preparing sequences from data with shape {data.shape}")
        
        # Select columns to use
        if target_columns is None:
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[target_columns]
        
        print(f"🎯 Using {len(numeric_data.columns)} features")
        
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

# Create TimeGAN instance
cement_timegan = CementTimeGAN(sequence_length=24, use_statistical_fallback=True)
print("✅ CementTimeGAN instance created!")