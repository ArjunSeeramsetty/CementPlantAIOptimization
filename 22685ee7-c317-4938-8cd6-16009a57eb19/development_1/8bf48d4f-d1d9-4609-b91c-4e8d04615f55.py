# Add application methods to existing disturbance simulator
def apply_single_disturbance(self, data: pd.DataFrame, disturbance_type: str, 
                            intensity: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
    """Apply a single disturbance type to the dataset"""
    
    if disturbance_type not in self.disturbance_scenarios:
        raise ValueError(f"Unknown disturbance type: {disturbance_type}")
        
    scenario = self.disturbance_scenarios[disturbance_type]
    disturbed_data = data.copy()
    n_samples = len(data)
    
    # Determine intensity
    if intensity is None:
        intensity = np.random.uniform(*scenario['intensity_range'])
    
    # Generate appropriate disturbance pattern
    if disturbance_type == 'feed_rate_variation':
        disturbance_pattern = self.generate_feed_rate_disturbance(n_samples, intensity)
    elif disturbance_type == 'fuel_quality_change':
        disturbance_pattern = self.generate_fuel_quality_disturbance(n_samples, intensity)
    elif disturbance_type == 'equipment_degradation':
        disturbance_pattern = self.generate_equipment_degradation(n_samples, intensity)
    elif disturbance_type == 'raw_material_composition_shift':
        disturbance_pattern = self.generate_raw_material_shift(n_samples, intensity)
    elif disturbance_type == 'maintenance_mode':
        disturbance_pattern = self.generate_maintenance_mode(n_samples, intensity)
    else:
        disturbance_pattern = np.random.normal(0, intensity, n_samples)
        
    # Apply to affected parameters
    affected_params = scenario['affected_params']
    applied_changes = {}
    
    for param in affected_params:
        if param in disturbed_data.columns:
            original_values = disturbed_data[param].values
            
            # Apply multiplicative disturbance (more realistic for most parameters)
            if param in ['kiln_temperature', 'coal_feed_rate', 'kiln_speed']:
                # Process parameters - multiplicative
                disturbed_values = original_values * (1 + disturbance_pattern)
            elif param in ['LSF', 'SM', 'AM']:
                # Chemical moduli - additive with constraints
                disturbed_values = original_values + disturbance_pattern * original_values.std()
                # Apply realistic constraints
                if param == 'LSF':
                    disturbed_values = np.clip(disturbed_values, 0.85, 1.05)
                elif param == 'SM':
                    disturbed_values = np.clip(disturbed_values, 2.0, 3.5)
                elif param == 'AM':
                    disturbed_values = np.clip(disturbed_values, 1.0, 2.5)
            elif param in ['C3S', 'C2S', 'C3A', 'C4AF']:
                # Bogue compounds - multiplicative with normalization
                disturbed_values = original_values * (1 + disturbance_pattern * 0.5)  # Smaller effect
                disturbed_values = np.clip(disturbed_values, 0, 100)
            else:
                # Default multiplicative
                disturbed_values = original_values * (1 + disturbance_pattern)
            
            # Store changes
            change_magnitude = np.mean(np.abs(disturbed_values - original_values) / original_values)
            applied_changes[param] = {
                'mean_change_pct': change_magnitude * 100,
                'max_change_pct': np.max(np.abs(disturbed_values - original_values) / original_values) * 100
            }
            
            # Update dataset
            disturbed_data[param] = disturbed_values
    
    # Propagate effects to dependent parameters (simplified for now)
    if 'kiln_temperature' in disturbed_data.columns and 'burnability_index' in disturbed_data.columns:
        temp_deviation = (disturbed_data['kiln_temperature'] - 1450) / 1450
        burnability_effect = temp_deviation * 10
        disturbed_data['burnability_index'] += burnability_effect
        disturbed_data['burnability_index'] = np.clip(disturbed_data['burnability_index'], 0, 100)
    
    disturbance_info = {
        'type': disturbance_type,
        'description': scenario['description'],
        'intensity': intensity,
        'applied_changes': applied_changes,
        'n_samples_affected': n_samples
    }
    
    return disturbed_data, disturbance_info

def apply_combined_disturbances(self, data: pd.DataFrame, 
                               scenario_probabilities: Optional[Dict[str, float]] = None):
    """Apply multiple disturbances based on realistic occurrence probabilities"""
    
    if scenario_probabilities is None:
        scenario_probabilities = {name: scenario['probability'] 
                                 for name, scenario in self.disturbance_scenarios.items()}
    
    disturbed_data = data.copy()
    applied_disturbances = []
    
    # Determine which disturbances occur
    active_disturbances = []
    for disturbance_type, probability in scenario_probabilities.items():
        if np.random.random() < probability:
            active_disturbances.append(disturbance_type)
    
    print(f"🎯 Applying {len(active_disturbances)} disturbance scenarios:")
    for dist_type in active_disturbances:
        print(f"   ✓ {dist_type}: {self.disturbance_scenarios[dist_type]['description']}")
    
    # Apply each active disturbance
    for disturbance_type in active_disturbances:
        disturbed_data, disturbance_info = self.apply_single_disturbance(
            disturbed_data, disturbance_type
        )
        applied_disturbances.append(disturbance_info)
    
    # Apply seasonal effects
    disturbed_data = self.apply_seasonal_effects(disturbed_data)
    
    return disturbed_data, applied_disturbances

def create_disturbance_scenarios(self, n_scenarios: int = 5):
    """Create multiple realistic disturbance scenarios for testing"""
    
    scenarios = []
    base_data = self.base_dataset.copy()
    
    print(f"🎯 Creating {n_scenarios} realistic disturbance scenarios...")
    
    for i in range(n_scenarios):
        print(f"\nScenario {i+1}:")
        
        # Vary the probability of disturbances for different scenarios
        if i == 0:
            # Normal operation (minimal disturbances)
            probabilities = {name: prob * 0.3 for name, prob in 
                           [(k, v['probability']) for k, v in self.disturbance_scenarios.items()]}
            print("   📋 Normal operation scenario")
        elif i == 1:
            # High stress scenario (multiple disturbances)
            probabilities = {name: min(0.8, prob * 2.0) for name, prob in 
                           [(k, v['probability']) for k, v in self.disturbance_scenarios.items()]}
            print("   ⚠️ High stress scenario")
        elif i == 2:
            # Maintenance scenario
            probabilities = {'maintenance_mode': 0.9, 'equipment_degradation': 0.6}
            for name in self.disturbance_scenarios:
                if name not in probabilities:
                    probabilities[name] = self.disturbance_scenarios[name]['probability'] * 0.2
            print("   🔧 Maintenance scenario")
        elif i == 3:
            # Raw material quality scenario
            probabilities = {'raw_material_composition_shift': 0.8, 'fuel_quality_change': 0.6}
            for name in self.disturbance_scenarios:
                if name not in probabilities:
                    probabilities[name] = self.disturbance_scenarios[name]['probability'] * 0.4
            print("   🏭 Raw material quality scenario")
        else:
            # Random scenario
            probabilities = {name: np.random.uniform(0.1, 0.7) 
                           for name in self.disturbance_scenarios}
            print("   🎲 Random disturbance scenario")
        
        # Generate scenario
        scenario_data, disturbance_info = self.apply_combined_disturbances(base_data, probabilities)
        scenarios.append((scenario_data, disturbance_info))
        
        # Print summary
        print(f"     Applied {len(disturbance_info)} disturbances")
        for info in disturbance_info:
            if info['applied_changes']:
                avg_change = np.mean([change['mean_change_pct'] 
                                    for change in info['applied_changes'].values()])
                print(f"     - {info['type']}: {avg_change:.1f}% average change")
    
    return scenarios

# Bind methods to the existing simulator instance
import types
disturbance_sim.apply_single_disturbance = types.MethodType(apply_single_disturbance, disturbance_sim)
disturbance_sim.apply_combined_disturbances = types.MethodType(apply_combined_disturbances, disturbance_sim)
disturbance_sim.create_disturbance_scenarios = types.MethodType(create_disturbance_scenarios, disturbance_sim)

print(f"\n✅ Disturbance Application Methods Added!")
print(f"🔧 Ready to apply comprehensive disturbance scenarios to cement data")