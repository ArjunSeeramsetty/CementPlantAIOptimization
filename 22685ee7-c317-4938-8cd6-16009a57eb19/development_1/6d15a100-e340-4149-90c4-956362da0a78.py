import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_disturbance_scenarios():
    """Analyze and visualize the comprehensive disturbance scenarios"""
    
    print("🔍 ANALYZING PROCESS DISTURBANCE SCENARIOS")
    print("=" * 50)
    
    scenarios = comprehensive_disturbance_scenarios
    scenario_names = list(scenarios.keys())
    
    # Statistical analysis of each scenario
    print(f"\n📊 STATISTICAL SUMMARY")
    
    for name, data in scenarios.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        
        # Key process parameters analysis
        key_params = ['kiln_temperature', 'coal_feed_rate', 'kiln_speed', 'LSF', 'C3S']
        
        for param in key_params:
            if param in data.columns:
                baseline = scenarios['base'][param]
                current = data[param]
                
                # Calculate changes
                mean_change = ((current.mean() - baseline.mean()) / baseline.mean()) * 100
                std_change = ((current.std() - baseline.std()) / baseline.std()) * 100
                
                print(f"   {param:20s}: {mean_change:+6.2f}% mean, {std_change:+6.2f}% variability")
    
    # Disturbance intensity analysis
    print(f"\n⚡ DISTURBANCE INTENSITY ANALYSIS")
    
    base_data = scenarios['base']
    intensity_summary = {}
    
    for scenario_name, scenario_data in scenarios.items():
        if scenario_name == 'base':
            continue
            
        # Calculate overall disturbance intensity
        total_deviation = 0
        param_count = 0
        
        for param in ['kiln_temperature', 'coal_feed_rate', 'kiln_speed', 'LSF', 'SM', 'AM']:
            if param in scenario_data.columns:
                baseline_values = base_data[param].values
                disturbed_values = scenario_data[param].values
                
                # Normalized mean absolute deviation
                deviation = np.mean(np.abs(disturbed_values - baseline_values) / baseline_values)
                total_deviation += deviation
                param_count += 1
        
        avg_intensity = (total_deviation / param_count) * 100 if param_count > 0 else 0
        intensity_summary[scenario_name] = avg_intensity
        
        print(f"   {scenario_name:25s}: {avg_intensity:6.2f}% average intensity")
    
    # Create visualization
    print(f"\n📈 CREATING DISTURBANCE VISUALIZATION")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Process Disturbance Scenarios Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature variations across scenarios
    ax = axes[0, 0]
    for i, (name, data) in enumerate(scenarios.items()):
        if name == 'base':
            ax.plot(data['kiln_temperature'][:500], 'k-', linewidth=2, label=name, alpha=0.8)
        else:
            ax.plot(data['kiln_temperature'][:500], label=name, alpha=0.7)
    ax.set_title('Kiln Temperature Variations (First 500 Samples)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Coal feed rate variations
    ax = axes[0, 1]
    for name, data in scenarios.items():
        if name == 'base':
            ax.plot(data['coal_feed_rate'][:500], 'k-', linewidth=2, label=name, alpha=0.8)
        else:
            ax.plot(data['coal_feed_rate'][:500], label=name, alpha=0.7)
    ax.set_title('Coal Feed Rate Variations (First 500 Samples)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Feed Rate (kg/h)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: LSF chemistry variations
    ax = axes[0, 2]
    for name, data in scenarios.items():
        if name == 'base':
            ax.plot(data['LSF'][:500], 'k-', linewidth=2, label=name, alpha=0.8)
        else:
            ax.plot(data['LSF'][:500], label=name, alpha=0.7)
    ax.set_title('LSF Chemistry Variations (First 500 Samples)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('LSF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Disturbance intensity comparison
    ax = axes[1, 0]
    scenario_labels = list(intensity_summary.keys())
    intensities = list(intensity_summary.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(intensities)))
    
    bars = ax.bar(range(len(scenario_labels)), intensities, color=colors, alpha=0.8)
    ax.set_title('Disturbance Intensity by Scenario')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Average Intensity (%)')
    ax.set_xticks(range(len(scenario_labels)))
    ax.set_xticklabels([label.replace('_', '\n') for label in scenario_labels], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, intensity in zip(bars, intensities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{intensity:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Parameter variability comparison
    ax = axes[1, 1]
    params = ['kiln_temperature', 'coal_feed_rate', 'kiln_speed', 'LSF']
    base_stds = [scenarios['base'][param].std() for param in params]
    combined_stds = [scenarios['combined_disturbances'][param].std() for param in params]
    
    x = np.arange(len(params))
    width = 0.35
    
    ax.bar(x - width/2, base_stds, width, label='Base', alpha=0.8, color='gray')
    ax.bar(x + width/2, combined_stds, width, label='Combined Disturbances', alpha=0.8, color='red')
    
    ax.set_title('Parameter Variability: Base vs Combined Disturbances')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Standard Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in params])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Quality impact analysis
    ax = axes[1, 2]
    quality_params = ['burnability_index', 'free_lime', 'C3S']
    
    scenario_impacts = {}
    for scenario_name, scenario_data in scenarios.items():
        if scenario_name == 'base':
            continue
            
        impacts = []
        for param in quality_params:
            if param in scenario_data.columns:
                base_mean = scenarios['base'][param].mean()
                scenario_mean = scenario_data[param].mean()
                impact = ((scenario_mean - base_mean) / base_mean) * 100
                impacts.append(impact)
            else:
                impacts.append(0)
        
        scenario_impacts[scenario_name] = impacts
    
    # Create stacked bar chart for quality impacts
    x = np.arange(len(quality_params))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_impacts)))
    
    for i, (scenario_name, impacts) in enumerate(scenario_impacts.items()):
        ax.bar(x + i * width, impacts, width, label=scenario_name.replace('_', ' '),
               color=colors[i], alpha=0.8)
    
    ax.set_title('Quality Parameter Impact by Scenario')
    ax.set_xlabel('Quality Parameters')
    ax.set_ylabel('Change from Baseline (%)')
    ax.set_xticks(x + width * (len(scenario_impacts) - 1) / 2)
    ax.set_xticklabels([p.replace('_', '\n') for p in quality_params])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n📋 COMPREHENSIVE SUMMARY")
    print("=" * 50)
    
    total_scenarios = len(scenarios) - 1  # Exclude base
    total_samples = len(scenarios['base'])
    
    print(f"✅ Successfully created {total_scenarios} disturbance scenarios")
    print(f"📊 Each scenario contains {total_samples:,} samples with realistic plant conditions")
    print(f"🎯 Disturbance types implemented:")
    print(f"   • Feed rate variations (cyclical + random walk)")
    print(f"   • Fuel quality changes (step changes + drift)")  
    print(f"   • Equipment degradation (progressive + episodic)")
    print(f"   • Raw material composition shifts (quarry face changes)")
    print(f"   • Maintenance mode operations (scheduled shutdowns)")
    print(f"   • Combined multi-disturbance scenarios")
    
    print(f"\n⚡ Disturbance intensity range: {min(intensity_summary.values()):.1f}% - {max(intensity_summary.values()):.1f}%")
    
    most_intense = max(intensity_summary.items(), key=lambda x: x[1])
    print(f"🔥 Most intense scenario: {most_intense[0]} ({most_intense[1]:.1f}% average)")
    
    return scenarios

# Run the analysis
analyzed_scenarios = analyze_disturbance_scenarios()

print(f"\n✅ PROCESS DISTURBANCE SIMULATION COMPLETE!")
print(f"🎯 Enhanced dataset ready with comprehensive realistic plant upset conditions")