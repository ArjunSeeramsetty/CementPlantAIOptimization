import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

@dataclass
class OptimizationObjective:
    """Defines an optimization objective with name, function, and direction"""
    name: str
    function: Callable
    minimize: bool  # True for minimization, False for maximization
    description: str

@dataclass
class OptimizationConstraint:
    """Defines an optimization constraint with bounds"""
    name: str
    function: Callable
    lower_bound: float
    upper_bound: float
    description: str

class OptimizationDataPrep:
    """
    Multi-objective optimization data preparation framework for cement manufacturing.
    
    Prepares data with objectives for:
    - Energy Efficiency: Minimizing heat consumption and energy use
    - Quality: Maximizing cement strength and minimizing free lime
    - Sustainability: Minimizing environmental impact and resource consumption
    """
    
    def __init__(self, cement_data: pd.DataFrame, seed: int = 42):
        """Initialize the optimization framework with cement data"""
        self.data = cement_data.copy()
        self.seed = seed
        np.random.seed(seed)
        
        # Validate required columns
        self._validate_data()
        
        # Initialize objectives and constraints
        self.objectives: List[OptimizationObjective] = []
        self.constraints: List[OptimizationConstraint] = []
        
        # Define decision variable ranges
        self.decision_variables = self._define_decision_variables()
        
        print(f"✅ OptimizationDataPrep initialized with {len(self.data)} samples")
        print(f"📊 Available features: {len(self.data.columns)} columns")
    
    def _validate_data(self):
        """Validate that required columns exist in the dataset"""
        required_cols = [
            'heat_consumption', 'burnability_index', 'free_lime', 'C3S', 'C2S', 'C3A', 'C4AF',
            'kiln_temperature', 'coal_feed_rate', 'raw_mill_fineness', 'cement_mill_fineness',
            'LSF', 'SM', 'AM', 'CaO', 'SiO2', 'Al2O3', 'Fe2O3'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✓ Data validation passed - all required columns present")
    
    def _define_decision_variables(self) -> Dict[str, Dict]:
        """Define the decision variable ranges for optimization"""
        variables = {}
        
        # Process variables that can be controlled
        process_vars = ['kiln_temperature', 'coal_feed_rate', 'raw_mill_fineness', 'cement_mill_fineness', 'kiln_speed']
        
        # Raw material chemistry variables (can be adjusted through raw mix design)
        chemistry_vars = ['CaO', 'SiO2', 'Al2O3', 'Fe2O3', 'LSF', 'SM', 'AM']
        
        for var in process_vars + chemistry_vars:
            if var in self.data.columns:
                q10, q90 = self.data[var].quantile([0.1, 0.9])
                variables[var] = {
                    'type': 'process' if var in process_vars else 'chemistry',
                    'min': q10,
                    'max': q90,
                    'current_mean': self.data[var].mean(),
                    'unit': self._get_variable_unit(var)
                }
        
        return variables
    
    def _get_variable_unit(self, var: str) -> str:
        """Get appropriate units for variables"""
        units = {
            'kiln_temperature': '°C', 'coal_feed_rate': 'kg/h', 'raw_mill_fineness': '%',
            'cement_mill_fineness': 'cm²/g', 'kiln_speed': 'rpm',
            'CaO': '%', 'SiO2': '%', 'Al2O3': '%', 'Fe2O3': '%',
            'LSF': 'ratio', 'SM': 'ratio', 'AM': 'ratio'
        }
        return units.get(var, 'unit')

# Initialize the optimization framework
opt_prep = OptimizationDataPrep(cement_dataset)

# Define objective functions
def energy_efficiency_objective(variables: Dict[str, float]) -> float:
    """Energy efficiency - minimize heat consumption"""
    temp = variables.get('kiln_temperature', 1450)
    coal_rate = variables.get('coal_feed_rate', 3200) 
    lsf = variables.get('LSF', 0.95)
    
    base_heat = 750  # kcal/kg clinker
    temp_effect = (temp - 1450) / 50 * 30
    coal_effect = (coal_rate - 3200) / 400 * 25
    chemistry_effect = abs(lsf - 0.95) * 100 * 20
    
    return max(base_heat + temp_effect + coal_effect + chemistry_effect, 700)

def quality_objective(variables: Dict[str, float]) -> float:
    """Quality - maximize strength, minimize defects"""
    c3s = variables.get('C3S', 50)
    fineness = variables.get('cement_mill_fineness', 350)
    temp = variables.get('kiln_temperature', 1450)
    lsf = variables.get('LSF', 0.95)
    
    strength_potential = c3s * 0.8 + (fineness - 300) * 0.1
    free_lime_penalty = max(0.1, 3.0 - (temp - 1400) * 0.05) * 10
    chemistry_penalty = abs(lsf - 0.95) * 50
    
    return strength_potential - free_lime_penalty - chemistry_penalty

def sustainability_objective(variables: Dict[str, float]) -> float:
    """Sustainability - minimize CO2 emissions"""
    temp = variables.get('kiln_temperature', 1450)
    coal_rate = variables.get('coal_feed_rate', 3200)
    cao = variables.get('CaO', 65)
    
    process_co2 = (temp - 1400) * 0.2 + (coal_rate - 3000) * 0.0001
    calcination_co2 = cao * 0.785  # CO2 from CaCO3 -> CaO + CO2
    
    return max(process_co2 + calcination_co2, 500)

# Create objectives
energy_obj = OptimizationObjective("energy_efficiency", energy_efficiency_objective, True, "Minimize heat consumption")
quality_obj = OptimizationObjective("cement_quality", quality_objective, False, "Maximize cement quality")
sustainability_obj = OptimizationObjective("sustainability", sustainability_objective, True, "Minimize CO2 emissions")

opt_prep.objectives = [energy_obj, quality_obj, sustainability_obj]

# Define constraint functions
def temp_constraint(vars): return vars.get('kiln_temperature', 1450)
def lsf_constraint(vars): return vars.get('LSF', 0.95)  
def sm_constraint(vars): return vars.get('SM', 2.5)
def coal_constraint(vars): return vars.get('coal_feed_rate', 3200)

# Create constraints
constraints = [
    OptimizationConstraint("temperature", temp_constraint, 1400, 1480, "Temperature limits"),
    OptimizationConstraint("LSF", lsf_constraint, 0.85, 1.05, "LSF limits"),
    OptimizationConstraint("SM", sm_constraint, 2.0, 3.5, "Silica modulus limits"),
    OptimizationConstraint("coal_rate", coal_constraint, 2800, 3600, "Coal feed rate limits")
]

opt_prep.constraints = constraints

# Test with sample solution
sample_solution = {
    'kiln_temperature': 1450, 'coal_feed_rate': 3200, 'LSF': 0.95,
    'C3S': 55, 'cement_mill_fineness': 350, 'CaO': 65, 'SM': 2.5
}

print(f"\n🎯 Multi-Objective Framework Complete!")
print(f"✓ Objectives: {len(opt_prep.objectives)}")
print(f"✓ Constraints: {len(opt_prep.constraints)}")
print(f"✓ Decision Variables: {len(opt_prep.decision_variables)}")

print(f"\n📊 Sample Solution Evaluation:")
for obj in opt_prep.objectives:
    value = obj.function(sample_solution)
    direction = "minimize" if obj.minimize else "maximize"
    print(f"  {obj.name}: {value:.2f} ({direction})")

print(f"\n🔒 Constraints Check:")
for const in opt_prep.constraints:
    value = const.function(sample_solution)
    feasible = const.lower_bound <= value <= const.upper_bound
    status = "✓" if feasible else "✗"
    print(f"  {const.name}: {value:.2f} [{const.lower_bound}-{const.upper_bound}] {status}")

print(f"\n✅ OPTIMIZATION DATA PREPARATION COMPLETE!")
print(f"Dataset ready with {len(opt_prep.objectives)} objectives and {len(opt_prep.constraints)} constraints")
print(f"Framework prepared for multi-objective optimization algorithms")