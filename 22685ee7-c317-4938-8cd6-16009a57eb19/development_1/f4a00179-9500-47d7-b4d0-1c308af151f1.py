import numpy as np
import pandas as pd

# Initialize the enhanced data generator
generator = EnhancedCementDataGenerator(seed=42)

# Generate 2500 samples for robust dataset
n_samples = 2500
print(f"Generating {n_samples} thermodynamic cement chemistry samples...")

# Generate raw material compositions
raw_materials_data = generator.generate_raw_material_composition(n_samples)
print(f"Generated raw material compositions: {raw_materials_data.shape}")

print(f"\n✅ Enhanced thermodynamic cement data generator ready!")
print(f"Next: Generate full dataset with chemistry relationships and process physics")