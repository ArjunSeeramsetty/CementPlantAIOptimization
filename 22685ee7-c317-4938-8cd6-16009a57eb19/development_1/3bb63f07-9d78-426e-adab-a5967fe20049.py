import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationObjectives:
    """Data structure for multi-objective optimization objectives"""
    energy_efficiency: float  # Lower energy consumption (minimize)
    quality_score: float      # Higher quality metrics (maximize) 
    sustainability_score: float  # Environmental impact (minimize)

@dataclass  
class DecisionVariables:
    """Decision variables for cement kiln optimization"""
    fuel_flow_rate: float     # kg/h (2800-3600)
    kiln_speed: float         # rpm (2.5-4.2)
    feed_rate: float          # tonnes/h (80-120)
    oxygen_content: float     # % (2-6)
    alt_fuel_usage: float     # % alternative fuel (0-30)

class OptimizationDataPrep:
    """
    Multi-objective optimization data preparation framework for cement manufacturing.
    Handles energy efficiency, quality, and sustainability objectives with constraints.
    """
    
    def __init__(self, cement_data: pd.DataFrame):
        """Initialize with cement dataset"""
        self.data = cement_data.copy()
        self.n_samples = len(cement_data)
        print(f"🎯 OptimizationDataPrep initialized with {self.n_samples} samples")
        
        # Define decision variable bounds
        self.decision_bounds = {
            'fuel_flow_rate': (2800, 3600),     # kg/h
            'kiln_speed': (2.5, 4.2),           # rpm  
            'feed_rate': (80, 120),              # tonnes/h
            'oxygen_content': (2, 6),            # %
            'alt_fuel_usage': (0, 30)            # % alternative fuel
        }
        
        # Initialize constraint thresholds
        self.constraints = {
            'temperature_min': 1400,    # °C
            'temperature_max': 1480,    # °C
            'quality_min': 0.75,        # Minimum quality score
            'free_lime_max': 3.0,       # % maximum free lime
            'burnability_min': 40.0,    # Minimum burnability index
            'c3s_min': 50.0,           # % minimum C3S content
            'lsf_min': 0.88,           # Minimum LSF
            'lsf_max': 1.02,           # Maximum LSF
        }
        
        print(f"✓ Decision variable bounds defined: {len(self.decision_bounds)} variables")
        print(f"✓ Constraint system initialized: {len(self.constraints)} constraints")
    
    def calculate_energy_efficiency_objective(self, decision_vars: DecisionVariables, 
                                            sample_data: pd.Series) -> float:
        """
        Calculate energy efficiency objective (minimize energy consumption)
        
        Args:
            decision_vars: Decision variable values
            sample_data: Process data for one sample
            
        Returns:
            Energy efficiency score (lower is better)
        """
        # Base heat consumption from sample data
        base_heat_consumption = sample_data['heat_consumption']
        
        # Adjustments based on decision variables
        fuel_efficiency = 1.0 - (decision_vars.fuel_flow_rate - 3200) / 3200 * 0.15
        speed_efficiency = 1.0 + (decision_vars.kiln_speed - 3.2) / 3.2 * 0.1
        alt_fuel_benefit = 1.0 - decision_vars.alt_fuel_usage / 100 * 0.2  # Alt fuels reduce energy
        
        # Calculate adjusted energy consumption
        energy_consumption = base_heat_consumption * fuel_efficiency * speed_efficiency * alt_fuel_benefit
        
        # Add oxygen enrichment effect (higher O2 reduces energy)
        oxygen_factor = 1.0 - (decision_vars.oxygen_content - 2) / 4 * 0.08
        energy_consumption *= oxygen_factor
        
        return energy_consumption
    
    def calculate_quality_objective(self, decision_vars: DecisionVariables, 
                                  sample_data: pd.Series) -> float:
        """
        Calculate quality objective (maximize quality score)
        
        Args:
            decision_vars: Decision variable values
            sample_data: Process data for one sample
            
        Returns:
            Quality score (higher is better, will be negated for minimization)
        """
        # Base quality from C3S content and burnability
        base_quality = (sample_data['C3S'] / 100 * 0.6 + 
                       sample_data['burnability_index'] / 100 * 0.4)
        
        # Process parameter effects on quality
        speed_quality_factor = 1.0 - abs(decision_vars.kiln_speed - 3.2) / 3.2 * 0.15
        oxygen_quality_factor = 1.0 + (decision_vars.oxygen_content - 2) / 4 * 0.12
        
        # Feed rate effect (optimal around 100 tonnes/h)
        optimal_feed_rate = 100
        feed_rate_factor = 1.0 - abs(decision_vars.feed_rate - optimal_feed_rate) / optimal_feed_rate * 0.1
        
        # Alternative fuel penalty on quality (slight reduction)
        alt_fuel_quality_penalty = 1.0 - decision_vars.alt_fuel_usage / 100 * 0.05
        
        quality_score = (base_quality * speed_quality_factor * oxygen_quality_factor * 
                        feed_rate_factor * alt_fuel_quality_penalty)
        
        return quality_score
    
    def calculate_sustainability_objective(self, decision_vars: DecisionVariables, 
                                         sample_data: pd.Series) -> float:
        """
        Calculate sustainability objective (minimize environmental impact)
        
        Args:
            decision_vars: Decision variable values
            sample_data: Process data for one sample
            
        Returns:
            Sustainability impact score (lower is better)
        """
        # Base emissions from fuel consumption
        base_emissions = decision_vars.fuel_flow_rate / 1000  # Normalize
        
        # Alternative fuel benefit (reduces CO2 emissions)
        alt_fuel_co2_reduction = decision_vars.alt_fuel_usage / 100 * 0.3
        emissions_factor = 1.0 - alt_fuel_co2_reduction
        
        # Oxygen enrichment increases energy but reduces other emissions
        oxygen_trade_off = 1.0 + (decision_vars.oxygen_content - 2) / 4 * 0.05
        
        # Kiln speed efficiency factor
        speed_sustainability_factor = 1.0 + abs(decision_vars.kiln_speed - 3.2) / 3.2 * 0.08
        
        sustainability_impact = base_emissions * emissions_factor * oxygen_trade_off * speed_sustainability_factor
        
        return sustainability_impact
    
    def temperature_constraint(self, decision_vars: DecisionVariables, 
                             sample_data: pd.Series) -> float:
        """Temperature constraint: must be within operational limits"""
        # Calculate predicted temperature based on decision variables
        base_temp = sample_data['kiln_temperature']
        fuel_temp_effect = (decision_vars.fuel_flow_rate - 3200) / 3200 * 30
        oxygen_temp_effect = (decision_vars.oxygen_content - 4) / 4 * 20
        speed_temp_effect = -(decision_vars.kiln_speed - 3.2) / 3.2 * 15
        
        predicted_temp = base_temp + fuel_temp_effect + oxygen_temp_effect + speed_temp_effect
        
        # Return violation amount (negative if constraint violated)
        temp_min_violation = predicted_temp - self.constraints['temperature_min']
        temp_max_violation = self.constraints['temperature_max'] - predicted_temp
        
        return min(temp_min_violation, temp_max_violation)
    
    def quality_constraint(self, decision_vars: DecisionVariables, 
                          sample_data: pd.Series) -> float:
        """Quality constraint: minimum quality level must be met"""
        quality_score = self.calculate_quality_objective(decision_vars, sample_data)
        return quality_score - self.constraints['quality_min']
    
    def chemistry_constraints(self, decision_vars: DecisionVariables, 
                            sample_data: pd.Series) -> List[float]:
        """Chemistry-based constraints (LSF, C3S, free lime)"""
        constraints = []
        
        # LSF constraints (must be within range for proper burning)
        lsf_min_constraint = sample_data['LSF'] - self.constraints['lsf_min'] 
        lsf_max_constraint = self.constraints['lsf_max'] - sample_data['LSF']
        constraints.extend([lsf_min_constraint, lsf_max_constraint])
        
        # C3S content constraint (minimum for strength)
        c3s_constraint = sample_data['C3S'] - self.constraints['c3s_min']
        constraints.append(c3s_constraint)
        
        # Free lime constraint (maximum allowable)
        free_lime_constraint = self.constraints['free_lime_max'] - sample_data['free_lime']
        constraints.append(free_lime_constraint)
        
        # Burnability constraint (minimum required)
        burnability_constraint = sample_data['burnability_index'] - self.constraints['burnability_min']
        constraints.append(burnability_constraint)
        
        return constraints
    
    def operational_constraints(self, decision_vars: DecisionVariables) -> List[float]:
        """Operational constraints (bounds checking and operational limits)"""
        constraints = []
        
        # Decision variable bounds (these should be handled by optimizer bounds, but double-check)
        for var_name, bounds in self.decision_bounds.items():
            var_value = getattr(decision_vars, var_name)
            min_bound, max_bound = bounds
            
            # Lower bound constraint
            constraints.append(var_value - min_bound)
            # Upper bound constraint  
            constraints.append(max_bound - var_value)
        
        # Fuel flow rate vs feed rate relationship (operational stability)
        fuel_to_feed_ratio = decision_vars.fuel_flow_rate / (decision_vars.feed_rate * 1000)
        stable_ratio_constraint = 0.05 - abs(fuel_to_feed_ratio - 0.032)  # Optimal around 32 kg fuel/tonne feed
        constraints.append(stable_ratio_constraint)
        
        return constraints
    
    def create_optimization_dataset(self) -> Dict:
        """
        Create optimization-ready dataset with objectives and constraints
        
        Returns:
            Dictionary containing prepared optimization data
        """
        print("🔄 Creating optimization-ready dataset...")
        
        # Initialize objective and constraint data storage
        objectives_data = []
        constraints_data = []
        decision_vars_data = []
        
        # Sample decision variables for each data point to evaluate ranges
        for idx, row in self.data.iterrows():
            if idx % 500 == 0:
                print(f"  Processing sample {idx}/{self.n_samples}...")
            
            # Sample decision variables within bounds for this evaluation
            sample_decision_vars = DecisionVariables(
                fuel_flow_rate=np.random.uniform(*self.decision_bounds['fuel_flow_rate']),
                kiln_speed=np.random.uniform(*self.decision_bounds['kiln_speed']),
                feed_rate=np.random.uniform(*self.decision_bounds['feed_rate']),
                oxygen_content=np.random.uniform(*self.decision_bounds['oxygen_content']),
                alt_fuel_usage=np.random.uniform(*self.decision_bounds['alt_fuel_usage'])
            )
            
            # Calculate objectives
            energy_obj = self.calculate_energy_efficiency_objective(sample_decision_vars, row)
            quality_obj = self.calculate_quality_objective(sample_decision_vars, row)
            sustainability_obj = self.calculate_sustainability_objective(sample_decision_vars, row)
            
            objectives_data.append({
                'energy_efficiency': energy_obj,
                'quality_score': quality_obj,
                'sustainability_score': sustainability_obj,
                'sample_idx': idx
            })
            
            # Calculate constraints
            temp_constraint = self.temperature_constraint(sample_decision_vars, row)
            qual_constraint = self.quality_constraint(sample_decision_vars, row)
            chem_constraints = self.chemistry_constraints(sample_decision_vars, row)
            op_constraints = self.operational_constraints(sample_decision_vars)
            
            constraints_data.append({
                'temperature_constraint': temp_constraint,
                'quality_constraint': qual_constraint,
                'chemistry_constraints': chem_constraints,
                'operational_constraints': op_constraints,
                'sample_idx': idx
            })
            
            # Store decision variables
            decision_vars_data.append({
                'fuel_flow_rate': sample_decision_vars.fuel_flow_rate,
                'kiln_speed': sample_decision_vars.kiln_speed,
                'feed_rate': sample_decision_vars.feed_rate,
                'oxygen_content': sample_decision_vars.oxygen_content,
                'alt_fuel_usage': sample_decision_vars.alt_fuel_usage,
                'sample_idx': idx
            })
        
        # Convert to DataFrames for easier analysis
        objectives_df = pd.DataFrame(objectives_data)
        constraints_df = pd.DataFrame(constraints_data)
        decision_vars_df = pd.DataFrame(decision_vars_data)
        
        # Create summary statistics
        optimization_summary = {
            'objectives_stats': {
                'energy_efficiency': {
                    'mean': objectives_df['energy_efficiency'].mean(),
                    'std': objectives_df['energy_efficiency'].std(),
                    'min': objectives_df['energy_efficiency'].min(),
                    'max': objectives_df['energy_efficiency'].max()
                },
                'quality_score': {
                    'mean': objectives_df['quality_score'].mean(),
                    'std': objectives_df['quality_score'].std(),
                    'min': objectives_df['quality_score'].min(),
                    'max': objectives_df['quality_score'].max()
                },
                'sustainability_score': {
                    'mean': objectives_df['sustainability_score'].mean(),
                    'std': objectives_df['sustainability_score'].std(),
                    'min': objectives_df['sustainability_score'].min(),
                    'max': objectives_df['sustainability_score'].max()
                }
            },
            'constraint_violations': {
                'temperature_violations': (constraints_df['temperature_constraint'] < 0).sum(),
                'quality_violations': (constraints_df['quality_constraint'] < 0).sum(),
            },
            'decision_variable_ranges': self.decision_bounds
        }
        
        print(f"✅ Optimization dataset created successfully!")
        print(f"✓ Objectives calculated for {len(objectives_df)} samples")
        print(f"✓ Constraints evaluated for {len(constraints_df)} samples") 
        print(f"✓ Decision variables sampled: {len(decision_vars_df)} configurations")
        
        return {
            'objectives': objectives_df,
            'constraints': constraints_df, 
            'decision_variables': decision_vars_df,
            'original_data': self.data,
            'summary': optimization_summary,
            'optimization_prep_instance': self
        }

# Initialize OptimizationDataPrep with cement dataset
optimization_prep = OptimizationDataPrep(cement_dataset)

# Create the optimization-ready dataset
optimization_dataset = optimization_prep.create_optimization_dataset()

print(f"\n🎯 SUCCESS: Multi-objective optimization framework ready!")
print(f"✓ OptimizationDataPrep class with 3 objectives and 4 constraint types")
print(f"✓ Decision variables: {list(optimization_prep.decision_bounds.keys())}")
print(f"✓ Energy efficiency, quality, and sustainability objectives defined")
print(f"✓ Temperature, quality, chemistry, and operational constraints implemented")
print(f"✓ Optimization-ready dataset with {len(optimization_dataset['objectives'])} samples")