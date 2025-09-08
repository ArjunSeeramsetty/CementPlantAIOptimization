import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences_from_cement_data(cement_dataset, key_parameters, sequence_length=24, samples_per_day=50):
    """
    Prepare time series sequences from cement plant data for TimeGAN training.
    
    Args:
        cement_dataset: DataFrame with cement plant data
        key_parameters: List of column names to use for time series
        sequence_length: Length of each sequence (24 for 24-hour periods)
        samples_per_day: Number of daily sequences to generate
    
    Returns:
        sequences: 3D array [n_sequences, sequence_length, n_features]
        scaler: Fitted MinMaxScaler for inverse transform
    """
    print(f"🔄 Preparing time series sequences from cement data...")
    print(f"📊 Input dataset shape: {cement_dataset.shape}")
    print(f"🎯 Key parameters: {key_parameters}")
    print(f"⏱️ Sequence length: {sequence_length} hours")
    print(f"📅 Generating {samples_per_day} daily sequences")
    
    # Extract key parameters from the dataset
    parameter_data = cement_dataset[key_parameters].copy()
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(parameter_data)
    
    print(f"✅ Data normalized - shape: {normalized_data.shape}")
    
    # Generate temporal sequences by creating realistic variations
    sequences = []
    
    for day in range(samples_per_day):
        # Start with a random base state from the real data
        base_idx = np.random.randint(0, len(normalized_data))
        base_state = normalized_data[base_idx].copy()
        
        # Generate 24-hour sequence with realistic temporal patterns
        daily_sequence = []
        
        for hour in range(sequence_length):
            # Add temporal patterns and realistic variations
            # Kiln temperature tends to be more stable with small variations
            if 'kiln_temperature' in key_parameters:
                temp_idx = key_parameters.index('kiln_temperature')
                base_state[temp_idx] += np.random.normal(0, 0.01)  # Small temperature drift
                base_state[temp_idx] = np.clip(base_state[temp_idx], 0, 1)
            
            # Kiln speed has process-related variations
            if 'kiln_speed' in key_parameters:
                speed_idx = key_parameters.index('kiln_speed')
                base_state[speed_idx] += np.random.normal(0, 0.02)  # More speed variation
                base_state[speed_idx] = np.clip(base_state[speed_idx], 0, 1)
            
            # Coal feed rate correlates with temperature (inverse relationship)
            if 'coal_feed_rate' in key_parameters and 'kiln_temperature' in key_parameters:
                coal_idx = key_parameters.index('coal_feed_rate')
                temp_idx = key_parameters.index('kiln_temperature')
                # Inverse correlation: higher temp -> slightly lower coal feed
                coal_adjustment = -0.1 * (base_state[temp_idx] - 0.5) + np.random.normal(0, 0.015)
                base_state[coal_idx] += coal_adjustment
                base_state[coal_idx] = np.clip(base_state[coal_idx], 0, 1)
            
            # Draft pressure variations
            if 'draft_pressure' in key_parameters:
                draft_idx = key_parameters.index('draft_pressure')
                base_state[draft_idx] += np.random.normal(0, 0.02)
                base_state[draft_idx] = np.clip(base_state[draft_idx], 0, 1)
            
            # Mill fineness parameters - more stable but with some drift
            for param in ['raw_mill_fineness', 'cement_mill_fineness']:
                if param in key_parameters:
                    param_idx = key_parameters.index(param)
                    base_state[param_idx] += np.random.normal(0, 0.005)  # Very small variations
                    base_state[param_idx] = np.clip(base_state[param_idx], 0, 1)
            
            # Add hourly patterns (some parameters vary by time of day)
            hour_factor = np.sin(2 * np.pi * hour / 24) * 0.02  # Small sinusoidal pattern
            base_state = base_state + hour_factor * np.random.random(len(key_parameters)) * 0.5
            base_state = np.clip(base_state, 0, 1)
            
            daily_sequence.append(base_state.copy())
        
        sequences.append(daily_sequence)
        
        if (day + 1) % 10 == 0:
            print(f"✓ Generated {day + 1} daily sequences")
    
    # Convert to numpy array
    sequences = np.array(sequences)
    print(f"✅ Final sequences shape: {sequences.shape}")
    print(f"📈 Generated {len(sequences)} sequences of {sequence_length} hours each")
    
    return sequences, scaler

# Prepare sequences using the 6 key cement plant parameters
key_cement_parameters = [
    'kiln_temperature', 
    'kiln_speed', 
    'coal_feed_rate', 
    'draft_pressure', 
    'raw_mill_fineness', 
    'cement_mill_fineness'
]

# Generate training sequences
training_sequences, data_scaler = prepare_sequences_from_cement_data(
    cement_dataset, 
    key_cement_parameters, 
    sequence_length=24, 
    samples_per_day=100  # Generate 100 daily sequences for training
)

print(f"\n🎯 SUCCESS: Time series sequences prepared!")
print(f"📊 Training data shape: {training_sequences.shape}")
print(f"🔧 Data scaler fitted for parameters: {key_cement_parameters}")
print(f"⚡ Ready for TimeGAN training!")