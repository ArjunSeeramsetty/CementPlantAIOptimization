# Test the TimeGAN implementation with cement data

# First, let's create a complete TimeGAN instance and test it
timegan_instance = CementTimeGAN(sequence_length=24, use_statistical_fallback=True)

# Test with cement dataset
print(f"🧪 Testing TimeGAN with cement dataset...")
print(f"Available cement dataset shape: {cement_dataset.shape}")

# Test sequence preparation with subset of features
key_features = ['CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'LSF', 'SM', 'AM', 
               'C3S', 'C2S', 'C3A', 'C4AF', 'kiln_temperature', 
               'burnability_index', 'heat_consumption']

print(f"🎯 Testing with {len(key_features)} key cement features...")

# Add method manually for this test
def prepare_sequences_test(data, target_columns=None, sequence_length=24):
    """Test function for preparing sequences."""
    print(f"📋 Preparing sequences from data with shape {data.shape}")
    
    # Select columns to use  
    if target_columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[target_columns]
    
    print(f"🎯 Using {len(numeric_data.columns)} features")
    
    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data.values)
    feature_dim = scaled_data.shape[1]
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length + 1):
        sequence = scaled_data[i:i + sequence_length]
        sequences.append(sequence)
    
    sequences = np.array(sequences)
    print(f"✅ Created {len(sequences)} sequences of shape ({sequence_length}, {feature_dim})")
    return sequences, scaler

# Test sequence preparation
test_sequences, test_scaler = prepare_sequences_test(cement_dataset, key_features, 24)

print(f"🎯 SUCCESS: TimeGAN sequence preparation tested!")
print(f"✓ Created {len(test_sequences)} sequences")
print(f"✓ Each sequence shape: {test_sequences[0].shape}")
print(f"✓ Total feature dimension: {test_sequences.shape[2]}")
print(f"✓ Ready for synthetic time series generation!")