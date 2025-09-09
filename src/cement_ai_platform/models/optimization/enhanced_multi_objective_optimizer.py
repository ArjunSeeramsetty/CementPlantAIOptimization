from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    _pymoo_available = True
except ImportError:
    _pymoo_available = False


@dataclass
class OptimizationObjective:
    """Define an optimization objective."""
    name: str
    weight: float
    minimize: bool = True
    target_value: Optional[float] = None


@dataclass
class OptimizationConstraint:
    """Define an optimization constraint."""
    name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    constraint_type: str = "inequality"  # "inequality" or "equality"


class EnhancedMultiObjectiveOptimizer:
    """
    Enhanced multi-objective optimizer for cement plant operations.
    
    Features:
    - Multiple optimization objectives (energy, quality, sustainability)
    - Physics-informed constraints
    - Real-time optimization capabilities
    - Pareto frontier analysis
    """
    
    def __init__(self, n_variables: int = 5):
        self.n_variables = n_variables
        self.objectives: List[OptimizationObjective] = []
        self.constraints: List[OptimizationConstraint] = []
        self.predictors: Dict[str, Any] = {}
        self.bounds: List[Tuple[float, float]] = []
        self.pareto_solutions: Optional[np.ndarray] = None
        
    def add_objective(self, name: str, weight: float, minimize: bool = True, 
                     target_value: Optional[float] = None):
        """Add an optimization objective."""
        objective = OptimizationObjective(name, weight, minimize, target_value)
        self.objectives.append(objective)
    
    def add_constraint(self, name: str, lower_bound: Optional[float] = None,
                      upper_bound: Optional[float] = None, constraint_type: str = "inequality"):
        """Add an optimization constraint."""
        constraint = OptimizationConstraint(name, lower_bound, upper_bound, constraint_type)
        self.constraints.append(constraint)
    
    def set_bounds(self, bounds: List[Tuple[float, float]]):
        """Set variable bounds."""
        self.bounds = bounds
        self.n_variables = len(bounds)
    
    def set_predictors(self, predictors: Dict[str, Any]):
        """Set predictive models for objectives."""
        self.predictors = predictors
    
    def _evaluate_objectives(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all objectives for given decision variables."""
        objectives = np.zeros(len(self.objectives))
        
        for i, obj in enumerate(self.objectives):
            if obj.name == "energy_efficiency":
                objectives[i] = self._calculate_energy_objective(x)
            elif obj.name == "quality_score":
                objectives[i] = self._calculate_quality_objective(x)
            elif obj.name == "sustainability_score":
                objectives[i] = self._calculate_sustainability_objective(x)
            elif obj.name == "production_rate":
                objectives[i] = self._calculate_production_objective(x)
            else:
                objectives[i] = 0.0  # Default
        
        return objectives
    
    def _calculate_energy_objective(self, x: np.ndarray) -> float:
        """Calculate energy efficiency objective."""
        # x = [feed_rate, fuel_rate, kiln_speed, temp, o2]
        feed_rate, fuel_rate, kiln_speed, temp, o2 = x
        
        # Base energy consumption
        base_energy = fuel_rate * 3.1  # MJ/h
        
        # Efficiency factors
        temp_efficiency = 1.0 - abs(temp - 1450) / 1450 * 0.1
        speed_efficiency = 1.0 - abs(kiln_speed - 3.5) / 3.5 * 0.05
        o2_efficiency = 1.0 - abs(o2 - 3.0) / 3.0 * 0.08
        
        # Total energy efficiency
        energy_efficiency = base_energy * temp_efficiency * speed_efficiency * o2_efficiency
        
        return energy_efficiency
    
    def _calculate_quality_objective(self, x: np.ndarray) -> float:
        """Calculate quality score objective."""
        # x = [feed_rate, fuel_rate, kiln_speed, temp, o2]
        feed_rate, fuel_rate, kiln_speed, temp, o2 = x
        
        # Use predictor if available
        if "quality_predictor" in self.predictors:
            try:
                pred = self.predictors["quality_predictor"].predict(x.reshape(1, -1))
                return float(pred[0])
            except:
                pass
        
        # Fallback physics-based calculation
        # Temperature effect on quality
        temp_score = 1.0 - abs(temp - 1450) / 1450 * 0.2
        
        # Residence time effect
        residence_time = 1.0 / kiln_speed
        residence_score = 1.0 - abs(residence_time - 0.3) / 0.3 * 0.15
        
        # Oxygen effect
        o2_score = 1.0 - abs(o2 - 3.0) / 3.0 * 0.1
        
        # Combined quality score
        quality_score = temp_score * residence_score * o2_score
        
        return quality_score
    
    def _calculate_sustainability_objective(self, x: np.ndarray) -> float:
        """Calculate sustainability score objective."""
        # x = [feed_rate, fuel_rate, kiln_speed, temp, o2]
        feed_rate, fuel_rate, kiln_speed, temp, o2 = x
        
        # Alternative fuel usage (simplified)
        alt_fuel_rate = 0.2  # 20% alternative fuel
        
        # CO2 emissions reduction
        co2_reduction = alt_fuel_rate * 0.3  # 30% reduction per 10% alt fuel
        
        # Energy efficiency factor
        energy_factor = self._calculate_energy_objective(x) / 100.0
        
        # Sustainability score
        sustainability_score = co2_reduction + energy_factor * 0.5
        
        return sustainability_score
    
    def _calculate_production_objective(self, x: np.ndarray) -> float:
        """Calculate production rate objective."""
        # x = [feed_rate, fuel_rate, kiln_speed, temp, o2]
        feed_rate, fuel_rate, kiln_speed, temp, o2 = x
        
        # Production efficiency
        temp_efficiency = 1.0 - abs(temp - 1450) / 1450 * 0.1
        speed_efficiency = kiln_speed / 3.5  # Normalized by optimal speed
        
        # Clinker production rate
        production_rate = feed_rate * 0.7 * temp_efficiency * speed_efficiency
        
        return production_rate
    
    def _evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraints for given decision variables."""
        constraints = np.zeros(len(self.constraints))
        
        for i, constraint in enumerate(self.constraints):
            if constraint.name == "temperature_range":
                temp = x[3]  # Temperature is 4th variable
                if constraint.lower_bound is not None and constraint.upper_bound is not None:
                    constraints[i] = temp - constraint.lower_bound  # >= 0
                    constraints[i+1] = constraint.upper_bound - temp  # >= 0
            elif constraint.name == "quality_range":
                quality = self._calculate_quality_objective(x)
                if constraint.lower_bound is not None:
                    constraints[i] = quality - constraint.lower_bound
            elif constraint.name == "energy_limit":
                energy = self._calculate_energy_objective(x)
                if constraint.upper_bound is not None:
                    constraints[i] = constraint.upper_bound - energy
        
        return constraints
    
    def optimize(self, n_generations: int = 100, population_size: int = 50) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        if not _pymoo_available:
            return self._optimize_fallback()
        
        # Define the optimization problem
        class CementOptimizationProblem(Problem):
            def __init__(self, optimizer):
                super().__init__(
                    n_var=optimizer.n_variables,
                    n_obj=len(optimizer.objectives),
                    n_constr=len(optimizer.constraints),
                    xl=np.array([b[0] for b in optimizer.bounds]),
                    xu=np.array([b[1] for b in optimizer.bounds])
                )
                self.optimizer = optimizer
            
            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate objectives
                f = self.optimizer._evaluate_objectives(x)
                
                # Evaluate constraints
                g = self.optimizer._evaluate_constraints(x)
                
                out["F"] = f
                out["G"] = g
        
        # Create and run optimization
        problem = CementOptimizationProblem(self)
        algorithm = NSGA2(pop_size=population_size)
        
        try:
            res = minimize(problem, algorithm, ('n_gen', n_generations), verbose=False)
            
            # Store Pareto solutions
            self.pareto_solutions = res.X
            
            return {
                "status": "success",
                "method": "nsga2",
                "pareto_solutions": res.X,
                "pareto_objectives": res.F,
                "n_solutions": len(res.X),
                "hypervolume": self._calculate_hypervolume(res.F)
            }
        except Exception as e:
            print(f"NSGA2 optimization failed: {e}")
            return self._optimize_fallback()
    
    def _optimize_fallback(self) -> Dict[str, Any]:
        """Fallback optimization when pymoo is not available."""
        from scipy.optimize import minimize
        
        # Single objective optimization (weighted sum)
        def objective(x):
            objectives = self._evaluate_objectives(x)
            weighted_sum = sum(obj.weight * objectives[i] for i, obj in enumerate(self.objectives))
            return weighted_sum
        
        # Constraints
        constraints = []
        for constraint in self.constraints:
            if constraint.name == "temperature_range":
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: x[3] - constraint.lower_bound if constraint.lower_bound else 0
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: constraint.upper_bound - x[3] if constraint.upper_bound else 0
                })
        
        # Initial guess
        x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        
        # Run optimization
        result = minimize(objective, x0, method='SLSQP', bounds=self.bounds, constraints=constraints)
        
        return {
            "status": "success",
            "method": "scipy_fallback",
            "optimal_solution": result.x,
            "optimal_objectives": self._evaluate_objectives(result.x),
            "success": result.success
        }
    
    def _calculate_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Calculate hypervolume metric for Pareto front quality."""
        if len(pareto_front) == 0:
            return 0.0
        
        # Simplified hypervolume calculation
        # In practice, use a proper hypervolume library
        volume = 0.0
        for point in pareto_front:
            point_volume = np.prod(point)
            volume += point_volume
        
        return volume / len(pareto_front)
    
    def get_recommendations(self, solution_idx: int = 0) -> Dict[str, Any]:
        """Get operational recommendations for a specific Pareto solution."""
        if self.pareto_solutions is None:
            raise ValueError("No optimization results available")
        
        solution = self.pareto_solutions[solution_idx]
        objectives = self._evaluate_objectives(solution)
        
        recommendations = {
            "decision_variables": {
                "feed_rate": solution[0],
                "fuel_rate": solution[1],
                "kiln_speed": solution[2],
                "temperature": solution[3],
                "oxygen_content": solution[4]
            },
            "objectives": {
                obj.name: objectives[i] for i, obj in enumerate(self.objectives)
            },
            "recommendations": self._generate_recommendations(solution, objectives)
        }
        
        return recommendations
    
    def _generate_recommendations(self, solution: np.ndarray, objectives: np.ndarray) -> List[str]:
        """Generate operational recommendations based on solution."""
        recommendations = []
        
        feed_rate, fuel_rate, kiln_speed, temp, o2 = solution
        
        # Temperature recommendations
        if temp < 1400:
            recommendations.append("Increase kiln temperature to improve clinker quality")
        elif temp > 1500:
            recommendations.append("Consider reducing temperature to save energy")
        
        # Speed recommendations
        if kiln_speed < 3.0:
            recommendations.append("Increase kiln speed for better mixing")
        elif kiln_speed > 4.0:
            recommendations.append("Reduce kiln speed to improve residence time")
        
        # Oxygen recommendations
        if o2 < 2.0:
            recommendations.append("Increase air flow to improve combustion")
        elif o2 > 4.0:
            recommendations.append("Reduce air flow to optimize fuel efficiency")
        
        # Energy recommendations
        energy_obj = objectives[0] if len(objectives) > 0 else 0
        if energy_obj > 100:
            recommendations.append("High energy consumption - optimize fuel rate")
        
        return recommendations


def create_enhanced_optimizer(n_variables: int = 5) -> EnhancedMultiObjectiveOptimizer:
    """Factory function to create an enhanced multi-objective optimizer."""
    return EnhancedMultiObjectiveOptimizer(n_variables)
