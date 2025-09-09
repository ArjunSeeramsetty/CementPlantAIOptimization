"""
Enhanced Grinding Circuit Simulation for Cement Plant Operations
Implements particle size distribution modeling, circuit optimization, and energy efficiency calculations.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

@dataclass
class MillConfiguration:
    """Mill configuration parameters."""
    mill_type: str  # 'ball_mill', 'vertical_mill', 'hpgr'
    diameter: float  # meters
    length: float    # meters  
    speed: float     # rpm (as fraction of critical speed)
    ball_charge: float  # %
    power_rating: float  # kW
    liner_type: str = 'rubber'  # 'rubber', 'steel', 'lifter'

@dataclass
class SeparatorConfiguration:
    """Separator configuration parameters."""
    separator_type: str  # 'mechanical', 'dynamic', 'high_efficiency'
    cut_size: float      # μm (d50)
    sharpness: float     # separation efficiency curve sharpness
    bypass: float        # % material bypassing separator
    efficiency: float = 0.85  # Overall separator efficiency

class GrindingCircuitSimulator:
    """Advanced grinding circuit simulation with PSD modeling and energy optimization."""

    def __init__(self):
        # Bond Work Indices for different materials (kWh/t)
        self.bond_work_indices = {
            'limestone': 12.0,
            'clay': 8.5,
            'iron_ore': 15.0,
            'sand': 16.0,
            'clinker': 13.5,
            'gypsum': 8.0,
            'fly_ash': 9.0,
            'slag': 11.0
        }

        # Standard sieve sizes (micrometers) for PSD modeling
        self.sieve_sizes = np.array([
            150000, 106000, 75000, 53000, 45000, 38000, 25000, 
            20000, 15000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5
        ])

        print("🔧 Enhanced Grinding Circuit Simulator initialized")
        print(f"📊 Material database: {list(self.bond_work_indices.keys())}")
        print(f"📏 PSD modeling: {len(self.sieve_sizes)} size classes")

    def calculate_bond_work_energy(self, material: str, feed_f80: float, 
                                 product_p80: float, correction_factors: Optional[Dict] = None) -> float:
        """
        Calculate grinding energy using Bond's Third Theory of Comminution.

        Args:
            material: Material type
            feed_f80: Feed 80% passing size (μm)
            product_p80: Product 80% passing size (μm)
            correction_factors: Optional corrections for mill type, etc.

        Returns:
            Specific energy consumption (kWh/t)
        """
        wi = self.bond_work_indices.get(material, 12.0)

        # Basic Bond equation
        energy = wi * (10/np.sqrt(product_p80) - 10/np.sqrt(feed_f80))

        # Apply correction factors if provided
        if correction_factors:
            for factor, value in correction_factors.items():
                if factor == 'mill_diameter':
                    # Mill diameter correction
                    energy *= (2.44 / value) ** 0.2
                elif factor == 'wet_grinding':
                    # Wet grinding correction
                    energy *= 1.3 if value else 1.0
                elif factor == 'open_circuit':
                    # Open vs closed circuit
                    energy *= 1.35 if value else 1.0

        return max(0, energy)

    def simulate_particle_size_distribution(self, feed_psd: np.ndarray,
                                          energy_input: float,
                                          mill_config: MillConfiguration,
                                          residence_time: float = 1.0) -> np.ndarray:
        """
        Simulate PSD evolution during grinding using population balance modeling.

        Args:
            feed_psd: Initial particle size distribution (cumulative % passing)
            energy_input: Specific energy input (kWh/t)
            mill_config: Mill configuration parameters
            residence_time: Material residence time in mill (hours)

        Returns:
            Product PSD after grinding
        """
        # Convert cumulative to differential distribution
        diff_psd = self._cumulative_to_differential(feed_psd)

        # Calculate breakage parameters
        selection_function = self._calculate_selection_function(mill_config)
        breakage_function = self._calculate_breakage_function(mill_config)

        # Time step for numerical integration
        time_step = residence_time / 50  # 50 time steps
        current_psd = diff_psd.copy()

        # Population balance model integration
        for _ in range(50):
            current_psd = self._update_psd_population_balance(
                current_psd, selection_function, breakage_function, 
                time_step, energy_input
            )

        # Convert back to cumulative distribution
        cumulative_psd = self._differential_to_cumulative(current_psd)

        return cumulative_psd

    def simulate_separator_performance(self, mill_product_psd: np.ndarray,
                                     separator_config: SeparatorConfiguration) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate classifier/separator performance with detailed partition curve.

        Args:
            mill_product_psd: Mill product PSD
            separator_config: Separator configuration

        Returns:
            fine_product_psd: Fine product PSD
            coarse_product_psd: Coarse product (recycle) PSD  
            performance_metrics: Separator performance data
        """
        # Calculate partition curve (efficiency vs size)
        partition_curve = self._calculate_partition_curve(separator_config)

        # Apply partition curve to mill product
        diff_mill_product = self._cumulative_to_differential(mill_product_psd)

        fine_diff = diff_mill_product * partition_curve
        coarse_diff = diff_mill_product * (1 - partition_curve)

        # Convert back to cumulative
        fine_product_psd = self._differential_to_cumulative(fine_diff)
        coarse_product_psd = self._differential_to_cumulative(coarse_diff)

        # Calculate performance metrics
        d50 = separator_config.cut_size
        d25 = d50 / (2 ** (1/separator_config.sharpness))
        d75 = d50 * (2 ** (1/separator_config.sharpness))

        imperfection = (d75 - d25) / (2 * d50)

        performance_metrics = {
            'd50_cut_size': d50,
            'd25': d25,
            'd75': d75,
            'imperfection': imperfection,
            'sharpness_index': separator_config.sharpness,
            'bypass': separator_config.bypass,
            'efficiency': separator_config.efficiency
        }

        return fine_product_psd, coarse_product_psd, performance_metrics

    def optimize_grinding_circuit(self, 
                                target_fineness: float,
                                production_rate: float,
                                energy_cost: float,
                                material_properties: Dict,
                                constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize grinding circuit for minimum operating cost while meeting quality targets.

        Args:
            target_fineness: Target Blaine fineness (cm²/g)
            production_rate: Required production rate (t/h)
            energy_cost: Energy cost ($/kWh)
            material_properties: Material properties including grindability
            constraints: Optional operational constraints

        Returns:
            Optimization results with optimal parameters and performance metrics
        """
        from scipy.optimize import minimize

        # Default constraints if not provided
        if constraints is None:
            constraints = {
                'max_mill_speed': 0.8,      # 80% of critical speed
                'min_mill_speed': 0.65,     # 65% of critical speed
                'max_separator_speed': 1500, # rpm
                'min_separator_speed': 500,  # rpm
                'max_recycle_ratio': 4.0,   # 400% recycle
                'min_recycle_ratio': 1.5    # 150% recycle
            }

        def objective_function(params):
            """Objective: minimize total operating cost ($/t)."""
            mill_speed, separator_speed, recycle_ratio = params

            # Calculate energy consumption
            mill_energy = self._calculate_mill_energy(
                mill_speed, production_rate, recycle_ratio, material_properties
            )

            separator_energy = self._calculate_separator_energy(
                separator_speed, production_rate, recycle_ratio
            )

            total_energy = mill_energy + separator_energy
            energy_cost_per_ton = total_energy * energy_cost / production_rate

            # Calculate achieved fineness
            achieved_fineness = self._estimate_circuit_fineness(
                mill_speed, separator_speed, recycle_ratio, material_properties
            )

            # Penalty for not meeting target fineness
            fineness_penalty = abs(achieved_fineness - target_fineness) * 0.1

            # Maintenance cost component
            maintenance_cost = self._estimate_maintenance_cost(
                mill_speed, separator_speed, recycle_ratio
            )

            return energy_cost_per_ton + fineness_penalty + maintenance_cost

        # Optimization constraints
        opt_constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - constraints['min_mill_speed']},
            {'type': 'ineq', 'fun': lambda x: constraints['max_mill_speed'] - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1] - constraints['min_separator_speed']},
            {'type': 'ineq', 'fun': lambda x: constraints['max_separator_speed'] - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2] - constraints['min_recycle_ratio']},
            {'type': 'ineq', 'fun': lambda x: constraints['max_recycle_ratio'] - x[2]}
        ]

        # Initial guess (reasonable starting point)
        x0 = [0.75, 1000, 2.5]  # mill_speed, separator_speed, recycle_ratio

        # Bounds for parameters
        bounds = [
            (constraints['min_mill_speed'], constraints['max_mill_speed']),
            (constraints['min_separator_speed'], constraints['max_separator_speed']),
            (constraints['min_recycle_ratio'], constraints['max_recycle_ratio'])
        ]

        # Run optimization
        result = minimize(
            objective_function, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'ftol': 1e-6, 'maxiter': 100}
        )

        if result.success:
            optimal_mill_speed, optimal_separator_speed, optimal_recycle_ratio = result.x

            # Calculate final performance metrics
            final_energy = self._calculate_mill_energy(
                optimal_mill_speed, production_rate, optimal_recycle_ratio, material_properties
            ) + self._calculate_separator_energy(
                optimal_separator_speed, production_rate, optimal_recycle_ratio
            )

            final_fineness = self._estimate_circuit_fineness(
                optimal_mill_speed, optimal_separator_speed, optimal_recycle_ratio, material_properties
            )

            optimization_results = {
                'optimization_successful': True,
                'optimal_parameters': {
                    'mill_speed': optimal_mill_speed,
                    'separator_speed': optimal_separator_speed,
                    'recycle_ratio': optimal_recycle_ratio
                },
                'performance_metrics': {
                    'achieved_fineness': final_fineness,
                    'specific_energy': final_energy / production_rate,
                    'total_cost_per_ton': result.fun,
                    'energy_cost_per_ton': final_energy * energy_cost / production_rate
                },
                'circuit_analysis': {
                    'mill_load': self._calculate_mill_load(optimal_mill_speed, optimal_recycle_ratio),
                    'separator_efficiency': self._calculate_separator_efficiency(optimal_separator_speed),
                    'circuit_efficiency': final_fineness / target_fineness
                }
            }
        else:
            optimization_results = {
                'optimization_successful': False,
                'error_message': result.message,
                'iterations': result.nit
            }

        return optimization_results

    def simulate_grinding_aid_effects(self, 
                                    base_energy: float,
                                    grinding_aid_type: str,
                                    dosage_rate: float) -> Dict[str, float]:
        """
        Simulate the effects of grinding aids on mill performance.

        Args:
            base_energy: Base energy consumption without aids (kWh/t)
            grinding_aid_type: Type of grinding aid
            dosage_rate: Dosage rate (kg/t cement)

        Returns:
            Effects of grinding aid on performance
        """
        grinding_aids = {
            'triethanolamine': {
                'energy_reduction': 0.15,  # 15% energy reduction
                'fineness_improvement': 1.10,  # 10% fineness improvement
                'optimal_dosage': 0.8,  # kg/t
                'cost_per_kg': 2.5  # $/kg
            },
            'diethylene_glycol': {
                'energy_reduction': 0.12,
                'fineness_improvement': 1.08,
                'optimal_dosage': 0.5,
                'cost_per_kg': 1.8
            },
            'propylene_glycol': {
                'energy_reduction': 0.10,
                'fineness_improvement': 1.06,
                'optimal_dosage': 1.0,
                'cost_per_kg': 1.2
            }
        }

        if grinding_aid_type not in grinding_aids:
            return {'error': f'Unknown grinding aid type: {grinding_aid_type}'}

        aid_properties = grinding_aids[grinding_aid_type]

        # Calculate effectiveness based on dosage
        dosage_effectiveness = min(1.0, dosage_rate / aid_properties['optimal_dosage'])

        # Energy reduction
        energy_reduction = aid_properties['energy_reduction'] * dosage_effectiveness
        new_energy = base_energy * (1 - energy_reduction)

        # Fineness improvement
        fineness_factor = 1 + (aid_properties['fineness_improvement'] - 1) * dosage_effectiveness

        # Cost calculation
        aid_cost_per_ton = dosage_rate * aid_properties['cost_per_kg']

        return {
            'energy_reduction_percent': energy_reduction * 100,
            'new_energy_consumption': new_energy,
            'energy_savings': base_energy - new_energy,
            'fineness_improvement_factor': fineness_factor,
            'grinding_aid_cost_per_ton': aid_cost_per_ton,
            'net_cost_benefit': (base_energy - new_energy) * 0.08 - aid_cost_per_ton  # Assuming $0.08/kWh
        }

    # Helper methods for internal calculations
    def _cumulative_to_differential(self, cumulative_psd: np.ndarray) -> np.ndarray:
        """Convert cumulative PSD to differential PSD."""
        diff_psd = np.zeros_like(cumulative_psd)
        diff_psd[0] = cumulative_psd[0]
        for i in range(1, len(cumulative_psd)):
            diff_psd[i] = cumulative_psd[i] - cumulative_psd[i-1]
        return np.maximum(diff_psd, 0)  # Ensure non-negative

    def _differential_to_cumulative(self, diff_psd: np.ndarray) -> np.ndarray:
        """Convert differential PSD to cumulative PSD."""
        return np.cumsum(diff_psd)

    def _calculate_selection_function(self, mill_config: MillConfiguration) -> np.ndarray:
        """Calculate size-specific breakage rate (selection function)."""
        # Austin-Luckie selection function
        sizes = self.sieve_sizes

        # Parameters depend on mill type and operating conditions
        if mill_config.mill_type == 'ball_mill':
            alpha = 0.5 * mill_config.speed  # Speed effect
            lambda_param = 1.0  # Size exponent
        elif mill_config.mill_type == 'vertical_mill':
            alpha = 0.8 * mill_config.speed
            lambda_param = 0.8
        else:  # HPGR
            alpha = 1.2 * mill_config.speed
            lambda_param = 0.6

        # Selection function: S(x) = alpha * (x/x_ref)^lambda
        x_ref = 1000  # Reference size (μm)
        selection = alpha * (sizes / x_ref) ** lambda_param

        return np.clip(selection, 0, 10)  # Reasonable limits

    def _calculate_breakage_function(self, mill_config: MillConfiguration) -> np.ndarray:
        """Calculate cumulative breakage function."""
        # Simplified breakage function - size reduction distribution
        n_sizes = len(self.sieve_sizes)
        breakage_matrix = np.zeros((n_sizes, n_sizes))

        for i in range(n_sizes):
            for j in range(i, n_sizes):
                # Probability of breaking from size i to size j
                if i == j:
                    breakage_matrix[i, j] = 0  # No self-breakage
                else:
                    # Simple exponential breakage distribution
                    size_ratio = self.sieve_sizes[j] / self.sieve_sizes[i]
                    breakage_matrix[i, j] = np.exp(-2 * (1 - size_ratio))

        return breakage_matrix

    def _update_psd_population_balance(self, current_psd: np.ndarray,
                                     selection: np.ndarray,
                                     breakage_matrix: np.ndarray,
                                     dt: float,
                                     energy_input: float) -> np.ndarray:
        """Update PSD using population balance model."""
        n_sizes = len(current_psd)
        new_psd = current_psd.copy()

        for i in range(n_sizes):
            # Disappearance due to breakage
            disappearance = current_psd[i] * selection[i] * dt * energy_input / 10

            # Generation from larger sizes
            generation = 0
            for j in range(i):
                generation += current_psd[j] * selection[j] * breakage_matrix[j, i] * dt * energy_input / 10

            new_psd[i] = current_psd[i] - disappearance + generation

        return np.maximum(new_psd, 0)  # Ensure non-negative

    def _calculate_partition_curve(self, separator_config: SeparatorConfiguration) -> np.ndarray:
        """Calculate separator partition curve using Plitt model."""
        sizes = self.sieve_sizes
        d50 = separator_config.cut_size
        m = separator_config.sharpness

        # Plitt equation with bypass
        partition_to_coarse = separator_config.bypass + (1 - separator_config.bypass) / (1 + (sizes / d50) ** m)
        partition_to_fine = 1 - partition_to_coarse

        return partition_to_fine

    def _calculate_mill_energy(self, mill_speed: float, production_rate: float,
                             recycle_ratio: float, material_properties: Dict) -> float:
        """Calculate mill energy consumption."""
        # Base energy from Bond's equation
        base_energy = material_properties.get('bond_work_index', 12.0)

        # Speed effect
        speed_factor = (mill_speed / 0.75) ** 1.5

        # Recycle effect
        recycle_factor = 1 + recycle_ratio * 0.3

        total_energy = base_energy * speed_factor * recycle_factor * production_rate
        return total_energy

    def _calculate_separator_energy(self, separator_speed: float, 
                                  production_rate: float, recycle_ratio: float) -> float:
        """Calculate separator energy consumption."""
        base_power = 5.0  # kW per t/h
        speed_factor = (separator_speed / 1000) ** 2
        recycle_factor = 1 + recycle_ratio * 0.1

        return base_power * speed_factor * recycle_factor * production_rate

    def _estimate_circuit_fineness(self, mill_speed: float, separator_speed: float,
                                 recycle_ratio: float, material_properties: Dict) -> float:
        """Estimate circuit product fineness."""
        base_fineness = 3000  # Base Blaine fineness

        # Mill speed effect
        speed_effect = mill_speed * 800

        # Separator speed effect
        separator_effect = separator_speed * 0.3

        # Recycle effect
        recycle_effect = recycle_ratio * 150

        # Material grindability effect
        grindability_effect = (15.0 - material_properties.get('bond_work_index', 12.0)) * 50

        total_fineness = base_fineness + speed_effect + separator_effect + recycle_effect + grindability_effect
        return max(2500, min(5000, total_fineness))  # Reasonable bounds

    def _estimate_maintenance_cost(self, mill_speed: float, separator_speed: float,
                                 recycle_ratio: float) -> float:
        """Estimate maintenance cost per ton."""
        # Mill wear cost
        mill_wear = (mill_speed ** 2) * 0.5  # Higher speeds = more wear

        # Separator wear cost  
        separator_wear = (separator_speed / 1000) * 0.2

        # Recycle impact on wear
        recycle_wear = recycle_ratio * 0.1

        return mill_wear + separator_wear + recycle_wear

    def _calculate_mill_load(self, mill_speed: float, recycle_ratio: float) -> float:
        """Calculate mill load percentage."""
        base_load = 30  # Base load %
        speed_effect = mill_speed * 20
        recycle_effect = recycle_ratio * 5

        return min(45, base_load + speed_effect + recycle_effect)  # Max 45% load

    def _calculate_separator_efficiency(self, separator_speed: float) -> float:
        """Calculate separator efficiency."""
        base_efficiency = 0.75
        speed_effect = (separator_speed - 500) / 1000 * 0.15  # Speed improves efficiency

        return min(0.95, max(0.60, base_efficiency + speed_effect))

# Factory function for easy instantiation
def create_grinding_circuit_simulator() -> GrindingCircuitSimulator:
    """Create a grinding circuit simulator instance."""
    return GrindingCircuitSimulator()
