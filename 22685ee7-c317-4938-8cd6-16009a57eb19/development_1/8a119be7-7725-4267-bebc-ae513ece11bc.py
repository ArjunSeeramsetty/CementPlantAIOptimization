import numpy as np
import pandas as pd

def create_comprehensive_disturbance_scenarios():
    """Create comprehensive process disturbance scenarios for cement plant data"""
    
    # Start with base dataset
    base_data = cement_dataset.copy()
    n_samples = len(base_data)
    
    print(f"🎯 Creating comprehensive process disturbance scenarios...")
    print(f"📊 Base dataset: {n_samples} samples with {len(base_data.columns)} parameters")
    
    # Scenario 1: Feed Rate Variations (Cyclical + Random)
    print(f"\n1️⃣ FEED RATE VARIATION SCENARIO")
    
    scenario_1 = base_data.copy()
    time_series = np.arange(n_samples)
    
    # Multi-frequency feed rate disturbance
    primary_cycle = np.sin(2 * np.pi * time_series / 480) * 0.08  # 8-hour shift cycle, 8% variation
    daily_cycle = np.sin(2 * np.pi * time_series / 1440) * 0.05  # 24-hour cycle, 5% variation
    random_walk = np.cumsum(np.random.normal(0, 0.02, n_samples))  # Random drift
    random_walk = (random_walk - np.mean(random_walk)) / np.std(random_walk) * 0.04
    
    feed_disturbance = primary_cycle + daily_cycle + random_walk + np.random.normal(0, 0.03, n_samples)
    
    # Apply to feed-related parameters
    scenario_1['kiln_temperature'] *= (1 + feed_disturbance)
    scenario_1['coal_feed_rate'] *= (1 + feed_disturbance * 1.2)  # Coal adjusts more
    scenario_1['draft_pressure'] *= (1 + feed_disturbance * 0.8)  # Draft follows
    
    # Propagate to quality parameters
    temp_effect = (scenario_1['kiln_temperature'] - 1450) / 1450
    scenario_1['burnability_index'] += temp_effect * 15
    scenario_1['burnability_index'] = np.clip(scenario_1['burnability_index'], 0, 100)
    
    print(f"   ✓ Applied cyclical feed rate variations (8h + 24h cycles)")
    print(f"   ✓ Temperature range: {scenario_1['kiln_temperature'].min():.1f} - {scenario_1['kiln_temperature'].max():.1f}°C")
    
    # Scenario 2: Fuel Quality Changes (Step Changes + Drift)
    print(f"\n2️⃣ FUEL QUALITY CHANGE SCENARIO")
    
    scenario_2 = base_data.copy()
    
    # Generate step changes for fuel quality (shipment changes)
    fuel_quality_change = np.zeros(n_samples)
    n_shipments = 5  # 5 fuel shipment changes
    shipment_points = np.random.choice(n_samples, n_shipments, replace=False)
    shipment_points = np.sort(shipment_points)
    
    current_quality = 0
    for i, point in enumerate(shipment_points):
        quality_change = np.random.uniform(-0.1, 0.1)  # ±10% quality change
        if i == 0:
            fuel_quality_change[:point] = current_quality
        else:
            fuel_quality_change[shipment_points[i-1]:point] = current_quality
        current_quality += quality_change
    fuel_quality_change[shipment_points[-1]:] = current_quality
    
    # Add gradual drift within shipments
    drift = np.linspace(0, np.random.uniform(-0.05, 0.05), n_samples)
    fuel_quality_change += drift
    
    # Apply to fuel-related parameters
    scenario_2['coal_feed_rate'] *= (1 + fuel_quality_change)
    scenario_2['kiln_temperature'] *= (1 + fuel_quality_change * 0.6)  # Temperature responds
    scenario_2['heat_consumption'] *= (1 - fuel_quality_change * 0.8)  # Better fuel = less consumption
    
    print(f"   ✓ Applied {n_shipments} fuel quality step changes")
    print(f"   ✓ Heat consumption range: {scenario_2['heat_consumption'].min():.1f} - {scenario_2['heat_consumption'].max():.1f} kcal/kg")
    
    # Scenario 3: Equipment Degradation (Progressive + Episodic)
    print(f"\n3️⃣ EQUIPMENT DEGRADATION SCENARIO")
    
    scenario_3 = base_data.copy()
    
    # Progressive degradation over time
    progressive_degradation = np.linspace(0, -0.06, n_samples)  # 6% degradation over time
    
    # Episodic degradation events
    episodic_events = np.zeros(n_samples)
    n_events = 3
    for _ in range(n_events):
        event_start = np.random.randint(n_samples // 4, 3 * n_samples // 4)
        event_duration = np.random.randint(100, 300)
        event_end = min(event_start + event_duration, n_samples)
        
        # Degradation with gradual recovery
        event_intensity = np.random.uniform(0.03, 0.08)
        recovery_curve = -event_intensity * np.exp(-np.linspace(0, 2, event_end - event_start))
        episodic_events[event_start:event_end] += recovery_curve
    
    total_degradation = progressive_degradation + episodic_events
    
    # Apply to equipment-related parameters
    scenario_3['kiln_speed'] *= (1 + total_degradation)
    scenario_3['raw_mill_fineness'] *= (1 - total_degradation * 0.5)  # Degradation reduces fineness
    scenario_3['cement_mill_fineness'] *= (1 - total_degradation * 0.3)
    
    print(f"   ✓ Applied progressive degradation with {n_events} episodic events")
    print(f"   ✓ Kiln speed range: {scenario_3['kiln_speed'].min():.2f} - {scenario_3['kiln_speed'].max():.2f} rpm")
    
    # Scenario 4: Raw Material Composition Shift (Quarry Face Changes)
    print(f"\n4️⃣ RAW MATERIAL COMPOSITION SCENARIO")
    
    scenario_4 = base_data.copy()
    
    # Piecewise constant composition changes (new quarry faces)
    n_quarry_changes = 4
    segment_lengths = np.random.multinomial(n_samples, np.ones(n_quarry_changes) / n_quarry_changes)
    
    chemistry_shifts = np.zeros(n_samples)
    current_pos = 0
    current_shift = 0
    
    for segment_length in segment_lengths:
        if segment_length == 0:
            continue
            
        # New quarry face composition shift
        new_shift = np.random.uniform(-0.03, 0.03)  # ±3% chemistry shift
        
        # Smooth transition over first 50 samples
        transition_length = min(50, segment_length // 2)
        if transition_length > 0:
            transition = np.linspace(current_shift, new_shift, transition_length)
            chemistry_shifts[current_pos:current_pos + transition_length] = transition
            chemistry_shifts[current_pos + transition_length:current_pos + segment_length] = new_shift
        else:
            chemistry_shifts[current_pos:current_pos + segment_length] = new_shift
            
        current_pos += segment_length
        current_shift = new_shift
        
        if current_pos >= n_samples:
            break
    
    # Apply to chemical parameters with constraints
    scenario_4['LSF'] = scenario_4['LSF'] + chemistry_shifts * scenario_4['LSF'].std()
    scenario_4['LSF'] = np.clip(scenario_4['LSF'], 0.85, 1.05)
    
    scenario_4['SM'] = scenario_4['SM'] + chemistry_shifts * scenario_4['SM'].std()
    scenario_4['SM'] = np.clip(scenario_4['SM'], 2.0, 3.5)
    
    scenario_4['AM'] = scenario_4['AM'] + chemistry_shifts * scenario_4['AM'].std()
    scenario_4['AM'] = np.clip(scenario_4['AM'], 1.0, 2.5)
    
    # Adjust Bogue compounds accordingly
    scenario_4['C3S'] *= (1 + chemistry_shifts * 0.3)
    scenario_4['C2S'] *= (1 - chemistry_shifts * 0.2)
    scenario_4['C3A'] *= (1 + chemistry_shifts * 0.4)
    scenario_4['C4AF'] *= (1 + chemistry_shifts * 0.1)
    
    # Clip Bogue compounds
    for compound in ['C3S', 'C2S', 'C3A', 'C4AF']:
        scenario_4[compound] = np.clip(scenario_4[compound], 0, 100)
    
    print(f"   ✓ Applied {n_quarry_changes} quarry face composition changes")
    print(f"   ✓ LSF range: {scenario_4['LSF'].min():.3f} - {scenario_4['LSF'].max():.3f}")
    
    # Scenario 5: Maintenance Mode (Scheduled Shutdowns)
    print(f"\n5️⃣ MAINTENANCE MODE SCENARIO")
    
    scenario_5 = base_data.copy()
    
    # Scheduled maintenance events
    maintenance_effect = np.zeros(n_samples)
    n_maintenance = 2  # 2 major maintenance events
    
    for _ in range(n_maintenance):
        maint_start = np.random.randint(n_samples // 4, 3 * n_samples // 4)
        maint_duration = np.random.randint(80, 150)
        maint_end = min(maint_start + maint_duration, n_samples)
        
        # Maintenance profile: ramp down, full maintenance, ramp up
        ramp_duration = maint_duration // 4
        
        # Create maintenance profile
        ramp_down = np.linspace(0, -0.2, ramp_duration)  # 20% reduction
        full_maint = np.ones(maint_duration - 2 * ramp_duration) * (-0.2)
        ramp_up = np.linspace(-0.2, 0, ramp_duration)
        
        maint_profile = np.concatenate([ramp_down, full_maint, ramp_up])
        actual_length = min(len(maint_profile), maint_end - maint_start)
        maintenance_effect[maint_start:maint_start + actual_length] = maint_profile[:actual_length]
    
    # Apply to process parameters
    scenario_5['kiln_temperature'] *= (1 + maintenance_effect)
    scenario_5['kiln_speed'] *= (1 + maintenance_effect)
    scenario_5['coal_feed_rate'] *= (1 + maintenance_effect)
    
    print(f"   ✓ Applied {n_maintenance} maintenance events")
    print(f"   ✓ Minimum operation level: {(1 + maintenance_effect.min()) * 100:.1f}%")
    
    # Combine scenarios into a comprehensive dataset
    print(f"\n🔄 COMBINED SCENARIO (All Disturbances)")
    
    combined_scenario = base_data.copy()
    
    # Apply all disturbances with reduced intensities (realistic overlap)
    combined_scenario['kiln_temperature'] *= (1 + feed_disturbance * 0.6 + 
                                             fuel_quality_change * 0.4 + 
                                             maintenance_effect * 0.8)
    
    combined_scenario['coal_feed_rate'] *= (1 + feed_disturbance * 0.8 + 
                                           fuel_quality_change * 0.6 + 
                                           maintenance_effect * 0.9)
    
    combined_scenario['kiln_speed'] *= (1 + total_degradation * 0.7 + 
                                       maintenance_effect * 0.6)
    
    # Apply chemistry shifts
    combined_scenario['LSF'] = scenario_4['LSF']  # Use full chemistry shift
    combined_scenario['SM'] = scenario_4['SM']
    combined_scenario['AM'] = scenario_4['AM']
    
    # Propagate combined effects
    temp_effect = (combined_scenario['kiln_temperature'] - 1450) / 1450
    combined_scenario['burnability_index'] += temp_effect * 12
    combined_scenario['burnability_index'] = np.clip(combined_scenario['burnability_index'], 0, 100)
    
    combined_scenario['free_lime'] += -temp_effect * 0.3  # Higher temp = lower free lime
    combined_scenario['free_lime'] = np.clip(combined_scenario['free_lime'], 0.1, 5.0)
    
    print(f"   ✓ Combined all disturbance types with realistic intensities")
    print(f"   ✓ Temperature range: {combined_scenario['kiln_temperature'].min():.1f} - {combined_scenario['kiln_temperature'].max():.1f}°C")
    
    # Package results
    disturbance_scenarios = {
        'base': base_data,
        'feed_rate_variation': scenario_1,
        'fuel_quality_change': scenario_2, 
        'equipment_degradation': scenario_3,
        'raw_material_shift': scenario_4,
        'maintenance_mode': scenario_5,
        'combined_disturbances': combined_scenario
    }
    
    print(f"\n✅ SUCCESS: Created {len(disturbance_scenarios)} comprehensive disturbance scenarios!")
    print(f"📊 Each scenario contains {n_samples} samples with realistic plant upset conditions")
    
    return disturbance_scenarios

# Generate the disturbance scenarios
comprehensive_disturbance_scenarios = create_comprehensive_disturbance_scenarios()

print(f"\n🎯 COMPREHENSIVE PROCESS DISTURBANCE SIMULATION COMPLETE!")
print(f"✅ Enhanced dataset with realistic plant upset conditions and operational scenarios")