import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CementTimeGAN:
    """TimeGAN for cement plant time series generation with statistical fallback."""
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.feature_dim = None
        self.is_fitted = False
        self.statistical_params = {}
        print(f"🎯 CementTimeGAN initialized (sequence_length={sequence_length})")
    
    def prepare_sequences(self, data, target_columns=None):
        """Prepare time series sequences from cement plant data."""
        print(f"📋 Preparing sequences from data with shape {data.shape}")
        
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

# Create TimeGAN model
timegan_model = CementTimeGAN(sequence_length=24)
print("✅ CementTimeGAN created successfully!")