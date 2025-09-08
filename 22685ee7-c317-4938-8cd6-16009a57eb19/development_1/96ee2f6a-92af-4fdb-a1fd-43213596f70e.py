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
        """
        Initialize the optimization framework with cement data
        
        Args:
            cement_data: DataFrame with cement process and chemistry data
            seed: Random seed for reproducibility
        """
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
        
        # Calculate realistic bounds based on data percentiles
        variables = {}
        
        # Process variables that can be controlled
        process_vars = [
            'kiln_temperature', 'coal_feed_rate', 'raw_mill_fineness', 
            'cement_mill_fineness', 'kiln_speed'
        ]
        
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
            'kiln_temperature': '°C',
            'coal_feed_rate': 'kg/h', 
            'raw_mill_fineness': '%',
            'cement_mill_fineness': 'cm²/g',
            'kiln_speed': 'rpm',
            'CaO': '%', 'SiO2': '%', 'Al2O3': '%', 'Fe2O3': '%',
            'LSF': 'ratio', 'SM': 'ratio', 'AM': 'ratio'
        }
        return units.get(var, 'unit')
    
    def add_energy_efficiency_objective(self) -> 'OptimizationDataPrep':
        """Add energy efficiency objective (minimize heat consumption)"""
        
        def energy_efficiency_score(variables: Dict[str, float]) -> float:
            """
            Calculate energy efficiency score based on heat consumption and burnability
            Lower values indicate better efficiency
            """
            # Predict heat consumption based on key variables
            temp_factor = (variables.get('kiln_temperature', 1450) - 1450) / 50
            coal_factor = (variables.get('coal_feed_rate', 3200) - 3200) / 400
            lsf_factor = abs(variables.get('LSF', 0.95) - 0.95) * 100
            
            # Base heat consumption + adjustments
            base_heat = 750  # kcal/kg clinker baseline
            heat_consumption = base_heat + temp_factor * 30 + coal_factor * 25 + lsf_factor * 20
            
            return max(heat_consumption, 700)  # Minimum realistic value
        
        objective = OptimizationObjective(
            name="energy_efficiency",
            function=energy_efficiency_score,
            minimize=True,
            description="Minimize heat consumption (kcal/kg clinker) for energy efficiency"
        )
        
        self.objectives.append(objective)
        print("✓ Added energy efficiency objective (minimize heat consumption)")
        return self
    
    def add_quality_objective(self) -> 'OptimizationDataPrep':
        """Add cement quality objective (maximize strength, minimize defects)"""
        
        def quality_score(variables: Dict[str, float]) -> float:
            """
            Calculate quality score based on C3S content, fineness, and free lime
            Higher values indicate better quality
            """
            # Key quality indicators
            c3s = variables.get('C3S', 50)  # Alite content for early strength
            fineness = variables.get('cement_mill_fineness', 350)
            lsf = variables.get('LSF', 0.95)
            
            # Quality score components
            strength_potential = c3s * 0.8 + (fineness - 300) * 0.1  # C3S and fineness drive strength
            burnability_penalty = abs(lsf - 0.95) * 50  # Penalty for non-optimal LSF
            
            # Predict free lime (quality defect) - lower is better
            temp = variables.get('kiln_temperature', 1450)
            free_lime_pred = max(0.1, 3.0 - (temp - 1400) * 0.05 - lsf * 2)
            
            quality = strength_potential - burnability_penalty - free_lime_pred * 10
            return quality
        
        objective = OptimizationObjective(
            name="cement_quality", 
            function=quality_score,
            minimize=False,  # Maximize quality
            description="Maximize cement quality (strength potential, minimize free lime)"
        )
        
        self.objectives.append(objective)
        print("✓ Added cement quality objective (maximize strength, minimize defects)")
        return self
    
    def add_sustainability_objective(self) -> 'OptimizationDataPrep':
        """Add sustainability objective (minimize environmental impact)"""
        
        def sustainability_score(variables: Dict[str, float]) -> float:
            """
            Calculate sustainability score based on CO2 emissions and resource efficiency
            Lower values indicate better sustainability
            """
            # CO2 emissions factors
            temp = variables.get('kiln_temperature', 1450)
            coal_rate = variables.get('coal_feed_rate', 3200)
            lsf = variables.get('LSF', 0.95)
            
            # Process CO2 (from temperature and coal)
            process_co2 = (temp - 1400) * 0.2 + (coal_rate - 3000) * 0.0001
            
            # Calcination CO2 (from raw material chemistry)
            cao_content = variables.get('CaO', 65)
            calcination_co2 = cao_content * 0.785  # Theoretical CO2 from limestone
            
            # Efficiency penalties
            lsf_penalty = abs(lsf - 0.95) * 20  # Non-optimal chemistry increases waste
            
            total_co2 = process_co2 + calcination_co2 + lsf_penalty
            return max(total_co2, 500)  # kg CO2/tonne cement minimum
        
        objective = OptimizationObjective(
            name="sustainability",
            function=sustainability_score, 
            minimize=True,
            description="Minimize CO2 emissions and environmental impact"
        )
        
        self.objectives.append(objective)
        print("✓ Added sustainability objective (minimize CO2 emissions)")
        return self
    
    def add_temperature_constraints(self, min_temp: float = 1400, max_temp: float = 1480) -> 'OptimizationDataPrep':
        """Add kiln temperature operational constraints"""
        
        def temperature_constraint(variables: Dict[str, float]) -> float:
            return variables.get('kiln_temperature', 1450)
        
        constraint = OptimizationConstraint(
            name="kiln_temperature",
            function=temperature_constraint,
            lower_bound=min_temp,
            upper_bound=max_temp, 
            description=f"Kiln temperature must be between {min_temp}°C and {max_temp}°C"
        )
        
        self.constraints.append(constraint)
        print(f"✓ Added temperature constraints: {min_temp}°C - {max_temp}°C")
        return self
    
    def add_chemistry_constraints(self) -> 'OptimizationDataPrep':
        """Add cement chemistry constraints for proper clinker formation"""
        
        # LSF (Lime Saturation Factor) constraint
        def lsf_constraint(variables: Dict[str, float]) -> float:
            return variables.get('LSF', 0.95)
        
        lsf_constraint_obj = OptimizationConstraint(
            name="LSF",
            function=lsf_constraint,
            lower_bound=0.85,
            upper_bound=1.05,
            description="LSF must be between 0.85-1.05 for proper clinker formation"
        )
        
        # Silica Modulus constraint  
        def sm_constraint(variables: Dict[str, float]) -> float:
            return variables.get('SM', 2.5)
        
        sm_constraint_obj = OptimizationConstraint(
            name="SM",
            function=sm_constraint,
            lower_bound=2.0,
            upper_bound=3.5,
            description="Silica Modulus must be between 2.0-3.5"
        )
        
        # Alumina Modulus constraint
        def am_constraint(variables: Dict[str, float]) -> float:
            return variables.get('AM', 2.0)
        
        am_constraint_obj = OptimizationConstraint(
            name="AM", 
            function=am_constraint,
            lower_bound=1.2,
            upper_bound=3.0,
            description="Alumina Modulus must be between 1.2-3.0"
        )
        
        self.constraints.extend([lsf_constraint_obj, sm_constraint_obj, am_constraint_obj])
        print("✓ Added chemistry constraints: LSF, SM, AM limits")
        return self
    
    def add_operational_constraints(self) -> 'OptimizationDataPrep':
        """Add operational constraints for safe plant operation"""
        
        # Coal feed rate constraint
        def coal_constraint(variables: Dict[str, float]) -> float:
            return variables.get('coal_feed_rate', 3200)
        
        coal_constraint_obj = OptimizationConstraint(
            name="coal_feed_rate",
            function=coal_constraint, 
            lower_bound=2800,
            upper_bound=3600,
            description="Coal feed rate must be between 2800-3600 kg/h"
        )
        
        # Cement fineness constraint
        def fineness_constraint(variables: Dict[str, float]) -> float:
            return variables.get('cement_mill_fineness', 350)
        
        fineness_constraint_obj = OptimizationConstraint(
            name="cement_mill_fineness",
            function=fineness_constraint,
            lower_bound=280,
            upper_bound=420,
            description="Cement fineness must be between 280-420 cm²/g"
        )
        
        self.constraints.extend([coal_constraint_obj, fineness_constraint_obj])
        print("✓ Added operational constraints: coal feed rate, cement fineness")
        return self
    
    def get_optimization_setup(self) -> Dict:
        """Get complete optimization setup for multi-objective solver"""
        
        setup = {
            'objectives': [
                {
                    'name': obj.name,
                    'minimize': obj.minimize,
                    'description': obj.description,
                    'function': obj.function
                } for obj in self.objectives
            ],
            'constraints': [
                {
                    'name': const.name,
                    'lower_bound': const.lower_bound,
                    'upper_bound': const.upper_bound,
                    'description': const.description,
                    'function': const.function
                } for const in self.constraints
            ],
            'decision_variables': self.decision_variables,
            'data_summary': {
                'n_samples': len(self.data),
                'n_features': len(self.data.columns),
                'date_range': 'Process data for multi-objective optimization'
            }
        }
        
        return setup
    
    def generate_sample_solutions(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate sample feasible solutions for testing"""
        
        print(f"🎯 Generating {n_samples} sample solutions for optimization testing...")
        
        solutions = []
        for i in range(n_samples):
            # Generate random solution within variable bounds
            solution = {}
            for var_name, var_info in self.decision_variables.items():
                if var_name in ['kiln_temperature', 'coal_feed_rate', 'cement_mill_fineness']:
                    # Sample from reasonable operational range
                    solution[var_name] = np.random.uniform(var_info['min'], var_info['max'])
                else:
                    # Chemistry variables - sample near typical values
                    mean_val = var_info['current_mean'] 
                    std_val = (var_info['max'] - var_info['min']) / 6
                    solution[var_name] = np.random.normal(mean_val, std_val)
                    solution[var_name] = np.clip(solution[var_name], var_info['min'], var_info['max'])
            
            # Calculate objective values
            obj_values = {}
            for obj in self.objectives:
                obj_values[f"{obj.name}_value"] = obj.function(solution)
                obj_values[f"{obj.name}_feasible"] = True  # Simplified feasibility
            
            # Check constraint satisfaction
            constraints_satisfied = True
            for const in self.constraints:
                const_value = const.function(solution)
                feasible = const.lower_bound <= const_value <= const.upper_bound
                obj_values[f"{const.name}_constraint_value"] = const_value
                obj_values[f"{const.name}_feasible"] = feasible
                if not feasible:
                    constraints_satisfied = False
            
            # Combine solution variables and objective values
            full_solution = {**solution, **obj_values, 'all_constraints_satisfied': constraints_satisfied}
            solutions.append(full_solution)
        
        solutions_df = pd.DataFrame(solutions)
        
        # Summary statistics
        feasible_count = solutions_df['all_constraints_satisfied'].sum()
        print(f"✅ Generated {n_samples} solutions: {feasible_count} feasible ({feasible_count/n_samples*100:.1f}%)")
        
        return solutions_df

# Initialize the optimization framework
opt_prep = OptimizationDataPrep(cement_dataset)

print(f"\n📋 Optimization Framework Summary:")
print(f"• Decision Variables: {len(opt_prep.decision_variables)} controllable parameters")
print(f"• Ready to add objectives and constraints...")
print(f"• Data prepared for multi-objective optimization algorithms")