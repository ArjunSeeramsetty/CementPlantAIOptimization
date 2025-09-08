# Add methods to the existing CementTimeGAN class

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

def _learn_statistical_patterns(self, sequences: np.ndarray):
    """Learn statistical patterns from sequences for fallback method."""
    print("📊 Learning statistical patterns for fallback generation...")
    
    n_sequences, seq_len, n_features = sequences.shape
    
    # Feature-wise statistics
    self.statistical_params = {
        'mean_sequence': np.mean(sequences, axis=0),  # (seq_len, n_features)
        'std_sequence': np.std(sequences, axis=0),    # (seq_len, n_features)
        'feature_correlations': np.corrcoef(sequences.reshape(-1, n_features).T),
        'temporal_transitions': [],
        'feature_ranges': {
            'min': np.min(sequences, axis=(0, 1)),
            'max': np.max(sequences, axis=(0, 1))
        }
    }
    
    # Learn temporal transitions
    for t in range(seq_len - 1):
        current_step = sequences[:, t, :]
        next_step = sequences[:, t + 1, :]
        
        # Compute correlation between consecutive timesteps
        transition_corr = np.corrcoef(current_step.T, next_step.T)[:n_features, n_features:]
        self.statistical_params['temporal_transitions'].append(transition_corr)
    
    print(f"✅ Statistical patterns learned:")
    print(f"   - Feature correlations: {self.statistical_params['feature_correlations'].shape}")
    print(f"   - Temporal transitions: {len(self.statistical_params['temporal_transitions'])}")

def _generate_statistical_sequences(self, n_sequences: int) -> np.ndarray:
    """Generate synthetic sequences using statistical fallback method."""
    print(f"🎲 Generating {n_sequences} sequences using statistical method...")
    
    synthetic_sequences = []
    
    for _ in range(n_sequences):
        sequence = np.zeros((self.sequence_length, self.feature_dim))
        
        # Initialize first timestep with statistical noise
        sequence[0] = (self.statistical_params['mean_sequence'][0] + 
                      np.random.normal(0, self.statistical_params['std_sequence'][0] * 0.5))
        
        # Generate subsequent timesteps using temporal patterns
        for t in range(1, self.sequence_length):
            if t - 1 < len(self.statistical_params['temporal_transitions']):
                # Use learned temporal transition
                transition_matrix = self.statistical_params['temporal_transitions'][t - 1]
                
                # Apply transition with noise
                next_values = np.dot(sequence[t - 1], transition_matrix.T)
                noise = np.random.normal(0, self.statistical_params['std_sequence'][t] * 0.3)
                sequence[t] = next_values + noise
            else:
                # Fallback to mean + noise for longer sequences
                sequence[t] = (self.statistical_params['mean_sequence'][t % len(self.statistical_params['mean_sequence'])] + 
                              np.random.normal(0, self.statistical_params['std_sequence'][t % len(self.statistical_params['std_sequence'])] * 0.4))
            
            # Ensure values stay within reasonable bounds
            sequence[t] = np.clip(sequence[t], 
                                self.statistical_params['feature_ranges']['min'] - 0.1,
                                self.statistical_params['feature_ranges']['max'] + 0.1)
        
        synthetic_sequences.append(sequence)
    
    return np.array(synthetic_sequences)

# Add methods to CementTimeGAN class
CementTimeGAN.prepare_sequences = prepare_sequences
CementTimeGAN._learn_statistical_patterns = _learn_statistical_patterns
CementTimeGAN._generate_statistical_sequences = _generate_statistical_sequences

print("✅ TimeGAN methods added successfully!")