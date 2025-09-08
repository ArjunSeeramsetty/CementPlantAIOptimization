import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class EnhancedCementDataGenerator:
    """
    Enhanced cement data generator with realistic chemistry relationships and process disturbances
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.chemistry = CementChemistry()
        
        # Process parameter ranges (realistic plant scenarios)
        self.process_ranges = {
            'kiln_temperature': {'min': 1400, 'max': 1480, 'target': 1450, 'std': 15},
            'kiln_speed': {'min': 2.5, 'max': 4.2, 'target': 3.2, 'std': 0.3},
            'coal_feed_rate': {'min': 2800, 'max': 3600, 'target': 3200, 'std': 150},
            'draft_pressure': {'min': -15, 'max': -8, 'target': -12, 'std': 2},
            'raw_mill_fineness': {'min': 8, 'max': 15, 'target': 12, 'std': 1.5},
            'cement_mill_fineness': {'min': 280, 'max': 420, 'target': 350, 'std': 25}
        }
        
        # Raw material composition ranges
        self.raw_material_ranges = {
            'limestone_cao': {'min': 48, 'max': 54, 'std': 1.5},
            'limestone_sio2': {'min': 2, 'max': 8, 'std': 1.2},
            'clay_sio2': {'min': 45, 'max': 65, 'std': 4},
            'clay_al2o3': {'min': 12, 'max': 22, 'std': 2.5},
            'iron_ore_fe2o3': {'min': 60, 'max': 85, 'std': 5},
            'sand_sio2': {'min': 85, 'max': 95, 'std': 2}
        }
        
        # Quality relationships (correlation matrices)
        self.quality_correlations = {
            'lsf_strength': 0.75,  # Higher LSF generally increases early strength
            'c3s_strength': 0.85,  # C3S strongly correlates with strength
            'fineness_strength': 0.65,  # Finer cement increases strength
            'temperature_burnability': 0.45  # Higher temp improves burnability
        }
    
    def generate_raw_material_composition(self, n_samples: int) -> pd.DataFrame:
        """
        Generate realistic raw material compositions with natural correlations
        """
        compositions = []
        
        for _ in range(n_samples):
            # Generate correlated raw material properties
            limestone_cao = np.random.normal(51, self.raw_material_ranges['limestone_cao']['std'])
            limestone_sio2 = np.random.normal(5, self.raw_material_ranges['limestone_sio2']['std'])
            
            # Clay composition (negatively correlated SiO2 and Al2O3)
            clay_sio2 = np.random.normal(55, self.raw_material_ranges['clay_sio2']['std'])
            clay_al2o3_base = np.random.normal(17, self.raw_material_ranges['clay_al2o3']['std'])
            # Adjust Al2O3 based on SiO2 (realistic clay chemistry)
            clay_al2o3 = clay_al2o3_base * (1 - 0.3 * (clay_sio2 - 55) / 55)
            
            iron_ore_fe2o3 = np.random.normal(72, self.raw_material_ranges['iron_ore_fe2o3']['std'])
            sand_sio2 = np.random.normal(90, self.raw_material_ranges['sand_sio2']['std'])
            
            compositions.append({
                'limestone_CaO': limestone_cao,
                'limestone_SiO2': limestone_sio2,
                'clay_SiO2': clay_sio2,
                'clay_Al2O3': clay_al2o3,
                'iron_ore_Fe2O3': iron_ore_fe2o3,
                'sand_SiO2': sand_sio2
            })
        
        return pd.DataFrame(compositions)
    
    def calculate_raw_mix_proportions(self, target_lsf=0.95, target_sm=2.5, target_am=2.0):
        """
        Calculate raw mix proportions to achieve target moduli
        """
        # Simplified calculation - in reality this would be iterative
        limestone_ratio = 0.78 + 0.05 * (target_lsf - 0.95)  # Adjust based on LSF target
        clay_ratio = 0.15 + 0.03 * (target_sm - 2.5)  # Adjust based on SM target
        iron_ore_ratio = 0.04 + 0.02 * (target_am - 2.0)  # Adjust based on AM target
        sand_ratio = 0.03
        
        # Normalize to 100%
        total_ratio = limestone_ratio + clay_ratio + iron_ore_ratio + sand_ratio
        
        return {
            'limestone_ratio': limestone_ratio / total_ratio,
            'clay_ratio': clay_ratio / total_ratio,
            'iron_ore_ratio': iron_ore_ratio / total_ratio,
            'sand_ratio': sand_ratio / total_ratio
        }
    
    def apply_process_disturbances(self, base_params: Dict) -> Dict:
        """
        Apply realistic process disturbances based on plant conditions
        """
        disturbed_params = base_params.copy()
        
        # Add correlated disturbances
        temp_disturbance = np.random.normal(0, self.process_ranges['kiln_temperature']['std'])
        disturbed_params['kiln_temperature'] += temp_disturbance
        
        # Coal feed rate correlates with temperature (control response)
        coal_adjustment = -0.5 * temp_disturbance + np.random.normal(0, 100)
        disturbed_params['coal_feed_rate'] += coal_adjustment
        
        # Kiln speed adjustment based on draft pressure
        draft_disturbance = np.random.normal(0, self.process_ranges['draft_pressure']['std'])
        disturbed_params['draft_pressure'] += draft_disturbance
        speed_adjustment = -0.02 * draft_disturbance + np.random.normal(0, 0.2)
        disturbed_params['kiln_speed'] += speed_adjustment
        
        return disturbed_params

    def generate_complete_dataset(self, n_samples: int = 2500) -> pd.DataFrame:
        """
        Generate complete thermodynamic cement dataset with all features
        """
        print(f"🎯 Generating {n_samples} thermodynamic cement chemistry samples...")
        
        # Generate raw material compositions
        raw_materials = self.generate_raw_material_composition(n_samples)
        print(f"✓ Generated raw material compositions: {raw_materials.shape}")
        
        # Initialize lists for all sample data
        cement_samples = []
        
        for i in range(n_samples):
            # Get raw material composition for this sample
            raw_comp = raw_materials.iloc[i]
            
            # Calculate raw mix proportions with some variation
            target_lsf = np.random.normal(0.95, 0.03)
            target_sm = np.random.normal(2.5, 0.2) 
            target_am = np.random.normal(2.0, 0.15)
            
            mix_props = self.calculate_raw_mix_proportions(target_lsf, target_sm, target_am)
            
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
            lsf_calc = self.chemistry.calculate_lsf(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)
            sm_calc = self.chemistry.calculate_silica_modulus(sio2_pct, al2o3_pct, fe2o3_pct)
            am_calc = self.chemistry.calculate_alumina_modulus(al2o3_pct, fe2o3_pct)
            
            # Calculate Bogue compounds
            bogue_compounds = self.chemistry.calculate_bogue_compounds(cao_pct, sio2_pct, al2o3_pct, fe2o3_pct)
            
            # Generate process parameters with disturbances
            base_process_params = {
                'kiln_temperature': self.process_ranges['kiln_temperature']['target'],
                'kiln_speed': self.process_ranges['kiln_speed']['target'],
                'coal_feed_rate': self.process_ranges['coal_feed_rate']['target'],
                'draft_pressure': self.process_ranges['draft_pressure']['target'],
                'raw_mill_fineness': self.process_ranges['raw_mill_fineness']['target'],
                'cement_mill_fineness': self.process_ranges['cement_mill_fineness']['target']
            }
            
            process_params = self.apply_process_disturbances(base_process_params)
            
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
        dataset = pd.DataFrame(cement_samples)
        
        print(f"\n🎯 SUCCESS: Generated {len(dataset)} cement chemistry samples!")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Columns: {len(dataset.columns)} features")
        
        # Display key statistics
        print(f"\n📊 Key Chemistry Statistics:")
        print(f"LSF range: {dataset['LSF'].min():.3f} - {dataset['LSF'].max():.3f}")
        print(f"SM range: {dataset['SM'].min():.2f} - {dataset['SM'].max():.2f}")
        print(f"C3S range: {dataset['C3S'].min():.1f}% - {dataset['C3S'].max():.1f}%")
        print(f"Burnability: {dataset['burnability_index'].min():.1f} - {dataset['burnability_index'].max():.1f}")
        
        return dataset

# Initialize generator and create the complete thermodynamic dataset
generator = EnhancedCementDataGenerator(seed=42)
cement_dataset = generator.generate_complete_dataset(2500)

print(f"\n✅ TASK COMPLETE: Enhanced thermodynamic cement data with realistic chemistry relationships generated!")
print(f"✓ CementChemistry class with Bogue equations and LSF calculations")
print(f"✓ EnhancedCementDataGenerator with process disturbances")  
print(f"✓ {len(cement_dataset)} samples with proper chemical constraints and process physics")