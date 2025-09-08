# Add multi-objective functions and constraints to the optimization framework

# Define energy efficiency objective function
def energy_efficiency_objective(variables: Dict[str, float]) -> float:
    """
    Energy efficiency objective - minimize heat consumption
    """
    temp = variables.get('kiln_temperature', 1450)
    coal_rate = variables.get('coal_feed_rate', 3200) 
    lsf = variables.get('LSF', 0.95)
    
    # Heat consumption model based on physical relationships
    base_heat = 750  # kcal/kg clinker baseline
    temp_effect = (temp - 1450) / 50 * 30  # Temperature impact on fuel efficiency  
    coal_effect = (coal_rate - 3200) / 400 * 25  # Coal feed rate impact
    chemistry_effect = abs(lsf - 0.95) * 100 * 20  # Chemistry efficiency impact
    
    heat_consumption = base_heat + temp_effect + coal_effect + chemistry_effect
    return max(heat_consumption, 700)  # Physical minimum

# Define quality objective function  
def quality_objective(variables: Dict[str, float]) -> float:
    """
    Quality objective - maximize cement strength and minimize defects
    """
    c3s = variables.get('C3S', 50)  # Alite content drives early strength
    fineness = variables.get('cement_mill_fineness', 350)
    temp = variables.get('kiln_temperature', 1450)
    lsf = variables.get('LSF', 0.95)
    
    # Quality score components
    strength_potential = c3s * 0.8 + (fineness - 300) * 0.1  
    free_lime_penalty = max(0.1, 3.0 - (temp - 1400) * 0.05) * 10  # Underburning penalty
    chemistry_penalty = abs(lsf - 0.95) * 50  # Non-optimal chemistry penalty
    
    quality_score = strength_potential - free_lime_penalty - chemistry_penalty
    return quality_score

# Define sustainability objective function
def sustainability_objective(variables: Dict[str, float]) -> float:
    """
    Sustainability objective - minimize CO2 emissions and environmental impact
    """
    temp = variables.get('kiln_temperature', 1450)
    coal_rate = variables.get('coal_feed_rate', 3200)
    cao = variables.get('CaO', 65)
    lsf = variables.get('LSF', 0.95)
    
    # CO2 emission components
    process_co2 = (temp - 1400) * 0.2 + (coal_rate - 3000) * 0.0001  # Process emissions
    calcination_co2 = cao * 0.785  # CO2 from limestone calcination (CaCO3 -> CaO + CO2)
    efficiency_penalty = abs(lsf - 0.95) * 20  # Inefficient chemistry increases waste
    
    total_co2 = process_co2 + calcination_co2 + efficiency_penalty
    return max(total_co2, 500)  # kg CO2/tonne cement minimum

# Create optimization objectives
energy_obj = OptimizationObjective(
    name="energy_efficiency",
    function=energy_efficiency_objective,
    minimize=True,
    description="Minimize heat consumption (kcal/kg clinker) for energy efficiency"
)

quality_obj = OptimizationObjective(
    name="cement_quality", 
    function=quality_objective,
    minimize=False,  # Maximize quality
    description="Maximize cement quality (strength potential, minimize defects)"
)

sustainability_obj = OptimizationObjective(
    name="sustainability",
    function=sustainability_objective,
    minimize=True,
    description="Minimize CO2 emissions and environmental impact"
)

# Add objectives to framework
opt_prep.objectives.extend([energy_obj, quality_obj, sustainability_obj])

# Define constraint functions
def temperature_constraint(variables: Dict[str, float]) -> float:
    return variables.get('kiln_temperature', 1450)

def lsf_constraint(variables: Dict[str, float]) -> float:
    return variables.get('LSF', 0.95)

def sm_constraint(variables: Dict[str, float]) -> float:
    return variables.get('SM', 2.5)

def coal_constraint(variables: Dict[str, float]) -> float:
    return variables.get('coal_feed_rate', 3200)

# Create optimization constraints
temp_constraint = OptimizationConstraint(
    name="kiln_temperature",
    function=temperature_constraint,
    lower_bound=1400,
    upper_bound=1480,
    description="Kiln temperature must be between 1400°C and 1480°C for safe operation"
)

lsf_constraint_obj = OptimizationConstraint(
    name="LSF", 
    function=lsf_constraint,
    lower_bound=0.85,
    upper_bound=1.05,
    description="LSF must be between 0.85-1.05 for proper clinker formation"
)

sm_constraint_obj = OptimizationConstraint(
    name="SM",
    function=sm_constraint,
    lower_bound=2.0,
    upper_bound=3.5,
    description="Silica Modulus must be between 2.0-3.5"
)

coal_constraint_obj = OptimizationConstraint(
    name="coal_feed_rate",
    function=coal_constraint,
    lower_bound=2800,
    upper_bound=3600,
    description="Coal feed rate must be between 2800-3600 kg/h"
)

# Add constraints to framework
opt_prep.constraints.extend([temp_constraint, lsf_constraint_obj, sm_constraint_obj, coal_constraint_obj])

print("🎯 Multi-Objective Optimization Setup Complete!")
print(f"✓ Objectives: {len(opt_prep.objectives)} - Energy, Quality, Sustainability")
print(f"✓ Constraints: {len(opt_prep.constraints)} - Temperature, Chemistry, Operational")
print(f"✓ Decision Variables: {len(opt_prep.decision_variables)} controllable parameters")

# Test the objectives with sample values
sample_solution = {
    'kiln_temperature': 1450,
    'coal_feed_rate': 3200,
    'LSF': 0.95,
    'C3S': 55,
    'cement_mill_fineness': 350,
    'CaO': 65,
    'SM': 2.5
}

print(f"\n📊 Sample Solution Evaluation:")
for obj in opt_prep.objectives:
    value = obj.function(sample_solution)
    direction = "minimize" if obj.minimize else "maximize"
    print(f"  {obj.name}: {value:.2f} ({direction})")

print(f"\n🔒 Constraint Validation:")
for const in opt_prep.constraints:
    value = const.function(sample_solution)
    feasible = const.lower_bound <= value <= const.upper_bound
    status = "✓ PASS" if feasible else "✗ FAIL"
    print(f"  {const.name}: {value:.2f} [{const.lower_bound}-{const.upper_bound}] {status}")