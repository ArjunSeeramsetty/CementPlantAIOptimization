import numpy as np
import pandas as pd

# Initialize the enhanced data generator
generator = EnhancedCementDataGenerator(seed=42)

# Generate 2500 samples for robust dataset
n_samples = 2500
print(f"Generating {n_samples} thermodynamic cement chemistry samples...")

# Generate raw material compositions
raw_materials = generator.generate_raw_material_composition(n_samples)
print(f"Generated raw material compositions: {raw_materials.shape}")

# Initialize lists for all sample data
cement_samples = []

for i in range(n_samples):
    # Get raw material composition for this sample
    raw_comp = raw_materials.iloc[i]
    
    # Calculate raw mix proportions with some variation
    target_lsf = np.random.normal(0.95, 0.03)
    target_sm = np.random.normal(2.5, 0.2) 
    target_am = np.random.normal(2.0, 0.15)
    
    mix_props = generator.calculate_raw_mix_proportions(target_lsf, target_sm, target_am)
    
    # Calculate raw meal composition from raw materials and mix proportions
    raw_meal_cao = (raw_comp['limestone_CaO'] * mix_props['limestone_ratio'] + 
                    0.5 * mix_props['clay_ratio'])  # Clay has some CaO
    raw_meal_sio2 = (raw_comp['limestone_SiO2'] * mix_props['limestone_ratio'] +
                     raw_comp['clay_SiO2'] * mix_props['clay_ratio'] +
                     raw_comp['sand_SiO2'] * mix_props['sand_ratio'])
    raw_meal_al2o3 = raw_comp['clay_Al2O3'] * mix_props['clay_ratio']
    raw_meal_fe2o3 = raw_comp['iron_ore_Fe2O3'] * mix_props['iron_ore_ratio']
    
    # Normalize to realistic cement chemistry ranges
    total_oxides = raw_meal_cao + raw_meal_sio2 + raw_meal_al2o3 + raw_meal_fe2o3
    cao_pct = (raw_meal_cao / total_oxides) * 100
    sio2_pct = (raw_meal_sio2 / total_oxides) * 100
    al2o3_pct = (raw_meal_al2o3 / total_oxides) * 100
    fe2o3_pct = (raw_meal_fe2o3 / total_oxides) * 100
    
    # Calculate moduli using CementChemistry class
    lsf_calc = generator.chemistry.calculate_lsf(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)
    sm_calc = generator.chemistry.calculate_silica_modulus(sio2_pct, al2o3_pct, fe2o3_pct)
    am_calc = generator.chemistry.calculate_alumina_modulus(al2o3_pct, fe2o3_pct)
    
    # Calculate Bogue compounds
    bogue_compounds = generator.chemistry.calculate_bogue_compounds(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)
    
    # Generate process parameters with disturbances
    base_process_params = {
        'kiln_temperature': generator.process_ranges['kiln_temperature']['target'],
        'kiln_speed': generator.process_ranges['kiln_speed']['target'],
        'coal_feed_rate': generator.process_ranges['coal_feed_rate']['target'],
        'draft_pressure': generator.process_ranges['draft_pressure']['target'],
        'raw_mill_fineness': generator.process_ranges['raw_mill_fineness']['target'],
        'cement_mill_fineness': generator.process_ranges['cement_mill_fineness']['target']
    }
    
    process_params = generator.apply_process_disturbances(base_process_params)
    
    # Calculate derived thermodynamic properties
    # Burnability index (function of LSF, fineness, temperature)
    burnability_base = 50 + 30 * (lsf_calc - 0.9) + 2 * (process_params['kiln_temperature'] - 1450)
    burnability_index = max(0, min(100, burnability_base + np.random.normal(0, 5)))
    
    # Heat consumption (kcal/kg clinker) - function of chemistry and process
    heat_consumption_base = 750 + 50 * (lsf_calc - 0.95) + 0.1 * (process_params['kiln_temperature'] - 1450)
    heat_consumption = max(700, heat_consumption_base + np.random.normal(0, 20))
    
    # Free lime (unreacted CaO) - inversely related to burnability and temperature
    free_lime_base = 2.5 - 0.05 * (process_params['kiln_temperature'] - 1450) - 0.02 * burnability_index
    free_lime = max(0.1, free_lime_base + np.random.normal(0, 0.3))
    
    # Compile sample data
    sample = {
        # Raw material inputs
        'limestone_CaO': raw_comp['limestone_CaO'],
        'limestone_SiO2': raw_comp['limestone_SiO2'],
        'clay_SiO2': raw_comp['clay_SiO2'],
        'clay_Al2O3': raw_comp['clay_Al2O3'],
        'iron_ore_Fe2O3': raw_comp['iron_ore_Fe2O3'],
        'sand_SiO2': raw_comp['sand_SiO2'],
        
        # Mix proportions
        'limestone_ratio': mix_props['limestone_ratio'],
        'clay_ratio': mix_props['clay_ratio'], 
        'iron_ore_ratio': mix_props['iron_ore_ratio'],
        'sand_ratio': mix_props['sand_ratio'],
        
        # Clinker chemistry
        'CaO': cao_pct,
        'SiO2': sio2_pct,
        'Al2O3': al2o3_pct,
        'Fe2O3': fe2o3_pct,
        
        # Moduli
        'LSF': lsf_calc,
        'SM': sm_calc,
        'AM': am_calc,
        
        # Bogue compounds
        'C3S': bogue_compounds['C3S'],
        'C2S': bogue_compounds['C2S'], 
        'C3A': bogue_compounds['C3A'],
        'C4AF': bogue_compounds['C4AF'],
        
        # Process parameters
        'kiln_temperature': process_params['kiln_temperature'],
        'kiln_speed': process_params['kiln_speed'],
        'coal_feed_rate': process_params['coal_feed_rate'],
        'draft_pressure': process_params['draft_pressure'],
        'raw_mill_fineness': process_params['raw_mill_fineness'],
        'cement_mill_fineness': process_params['cement_mill_fineness'],
        
        # Thermodynamic properties
        'burnability_index': burnability_index,
        'heat_consumption': heat_consumption,
        'free_lime': free_lime
    }
    
    cement_samples.append(sample)

# Create comprehensive dataset
cement_dataset = pd.DataFrame(cement_samples)

print(f"\n🎯 SUCCESS: Generated {len(cement_dataset)} cement chemistry samples!")
print(f"Dataset shape: {cement_dataset.shape}")
print(f"Columns: {len(cement_dataset.columns)} features")

# Display key statistics
print(f"\n📊 Key Chemistry Statistics:")
print(f"LSF range: {cement_dataset['LSF'].min():.3f} - {cement_dataset['LSF'].max():.3f}")
print(f"SM range: {cement_dataset['SM'].min():.2f} - {cement_dataset['SM'].max():.2f}")
print(f"C3S range: {cement_dataset['C3S'].min():.1f}% - {cement_dataset['C3S'].max():.1f}%")
print(f"Burnability: {cement_dataset['burnability_index'].min():.1f} - {cement_dataset['burnability_index'].max():.1f}")

print(f"\n✅ Task Complete: Enhanced thermodynamic cement data with realistic chemistry relationships generated!")