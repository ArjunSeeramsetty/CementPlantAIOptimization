import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize

@dataclass
class UtilityBenchmarks:
    """Utility consumption benchmarks for cement plants"""
    compressed_air_specific: float = 25.0  # Nm³/t cement
    water_specific: float = 0.3  # m³/t cement
    compressed_air_pressure_optimal: float = 7.5  # bar
    water_temperature_cooling: float = 35.0  # °C
    power_factor_target: float = 0.95

class CompressedAirOptimizer:
    """Optimizes compressed air system efficiency and consumption"""
    
    def __init__(self):
        self.benchmarks = UtilityBenchmarks()
        self.leak_detection_threshold = 0.15  # 15% leakage considered high
        
    def analyze_system_efficiency(self, air_data: Dict) -> Dict:
        """Analyze compressed air system efficiency"""
        
        # Calculate specific consumption
        air_consumption = air_data.get('total_consumption_nm3_h', 1000)
        production_rate = air_data.get('production_rate_tph', 167)
        specific_consumption = air_consumption / production_rate
        
        # Detect leakages
        baseline_consumption = air_data.get('baseline_consumption_nm3_h', 800)
        leakage_rate = max(0, (air_consumption - baseline_consumption) / baseline_consumption)
        
        # Pressure optimization
        system_pressure = air_data.get('system_pressure_bar', 8.0)
        pressure_optimization = self._calculate_pressure_optimization(system_pressure, air_data)
        
        # Compressor efficiency
        power_consumption = air_data.get('compressor_power_kw', 150)
        theoretical_power = air_consumption * 0.125  # kW per Nm³/h (approximation)
        compressor_efficiency = theoretical_power / power_consumption if power_consumption > 0 else 0
        
        return {
            'specific_consumption': specific_consumption,
            'benchmark_comparison': specific_consumption / self.benchmarks.compressed_air_specific,
            'leakage_rate': leakage_rate,
            'leakage_status': 'High' if leakage_rate > self.leak_detection_threshold else 'Acceptable',
            'pressure_optimization': pressure_optimization,
            'compressor_efficiency': compressor_efficiency,
            'annual_energy_cost': self._calculate_annual_energy_cost(power_consumption),
            'potential_savings': self._calculate_potential_savings(air_data)
        }
    
    def _calculate_pressure_optimization(self, current_pressure: float, air_data: Dict) -> Dict:
        """Calculate optimal pressure settings"""
        
        # Get pressure requirements for different applications
        applications = air_data.get('applications', {
            'conveying': 6.0,
            'instrumentation': 7.0,
            'cleaning': 5.5,
            'bag_filter_cleaning': 8.0
        })
        
        # Find optimal pressure (highest requirement + safety margin)
        min_required_pressure = max(applications.values()) + 0.5
        
        # Calculate energy savings from pressure reduction
        if current_pressure > min_required_pressure:
            # Rule of thumb: 1 bar reduction = 7% energy savings
            pressure_reduction = current_pressure - min_required_pressure
            energy_savings_percent = pressure_reduction * 7
            
            power_consumption = air_data.get('compressor_power_kw', 150)
            annual_savings = power_consumption * 8760 * energy_savings_percent / 100 * 0.08  # $0.08/kWh
        else:
            pressure_reduction = 0
            energy_savings_percent = 0
            annual_savings = 0
        
        return {
            'current_pressure': current_pressure,
            'optimal_pressure': min_required_pressure,
            'pressure_reduction_possible': pressure_reduction,
            'energy_savings_percent': energy_savings_percent,
            'annual_savings_usd': annual_savings
        }
    
    def _calculate_annual_energy_cost(self, power_kw: float, hours_per_year: int = 8760) -> float:
        """Calculate annual energy cost for compressed air"""
        return power_kw * hours_per_year * 0.08  # Assuming $0.08/kWh
    
    def _calculate_potential_savings(self, air_data: Dict) -> Dict:
        """Calculate potential savings from various optimizations"""
        
        savings = {
            'leak_repair': 0,
            'pressure_optimization': 0,
            'compressor_sizing': 0,
            'heat_recovery': 0,
            'total': 0
        }
        
        power_consumption = air_data.get('compressor_power_kw', 150)
        annual_energy_cost = self._calculate_annual_energy_cost(power_consumption)
        
        # Leak repair savings
        leakage_rate = max(0, (air_data.get('total_consumption_nm3_h', 1000) - 
                              air_data.get('baseline_consumption_nm3_h', 800)) / 
                              air_data.get('baseline_consumption_nm3_h', 800))
        if leakage_rate > 0.05:  # 5% threshold
            savings['leak_repair'] = annual_energy_cost * min(leakage_rate, 0.3)
        
        # Pressure optimization (already calculated above)
        current_pressure = air_data.get('system_pressure_bar', 8.0)
        if current_pressure > 7.5:
            pressure_savings_percent = (current_pressure - 7.5) * 7
            savings['pressure_optimization'] = annual_energy_cost * pressure_savings_percent / 100
        
        # Compressor right-sizing
        consumption_variation = air_data.get('consumption_variation_percent', 20)
        if consumption_variation > 30:  # High variation suggests oversizing
            savings['compressor_sizing'] = annual_energy_cost * 0.1
        
        # Heat recovery potential
        waste_heat_recoverable = power_consumption * 0.9  # 90% of power becomes heat
        if waste_heat_recoverable > 50:  # kW
            # Assume 30% of waste heat can be recovered and used
            savings['heat_recovery'] = waste_heat_recoverable * 0.3 * 8760 * 0.08 * 0.7
        
        savings['total'] = sum(savings.values())
        
        return savings
    
    def optimize_compressor_schedule(self, demand_profile: Dict) -> Dict:
        """Optimize compressor operation schedule based on demand"""
        
        hourly_demand = demand_profile.get('hourly_demand', [100] * 24)
        compressor_capacities = demand_profile.get('compressor_capacities', [50, 75, 100])
        
        optimized_schedule = []
        total_energy = 0
        
        for hour in range(24):
            demand = hourly_demand[hour]
            
            # Find optimal compressor combination
            best_combination = self._find_optimal_compressor_combination(
                demand, compressor_capacities
            )
            
            optimized_schedule.append(best_combination)
            total_energy += best_combination['energy_consumption']
        
        return {
            'optimized_schedule': optimized_schedule,
            'total_daily_energy': total_energy,
            'energy_savings_percent': self._calculate_schedule_savings(demand_profile, optimized_schedule)
        }
    
    def _find_optimal_compressor_combination(self, demand: float, capacities: List[float]) -> Dict:
        """Find optimal combination of compressors for given demand"""
        
        # Simple greedy approach - can be improved with dynamic programming
        running_compressors = []
        total_capacity = 0
        total_energy = 0
        
        sorted_capacities = sorted(capacities, reverse=True)
        
        for capacity in sorted_capacities:
            if total_capacity < demand:
                running_compressors.append(capacity)
                total_capacity += capacity
                # Energy consumption model: quadratic with capacity
                total_energy += capacity * 0.8 + (capacity ** 2) * 0.002
        
        return {
            'running_compressors': running_compressors,
            'total_capacity': total_capacity,
            'energy_consumption': total_energy,
            'efficiency': demand / total_capacity if total_capacity > 0 else 0
        }
    
    def _calculate_schedule_savings(self, original_profile: Dict, optimized_schedule: List) -> float:
        """Calculate energy savings from optimized schedule"""
        
        # Assume original schedule runs all compressors at 80% efficiency
        original_energy = sum(original_profile.get('hourly_demand', [100] * 24)) * 1.25
        optimized_energy = sum(hour['energy_consumption'] for hour in optimized_schedule)
        
        return max(0, (original_energy - optimized_energy) / original_energy * 100)

class WaterSystemOptimizer:
    """Optimizes water consumption and cooling systems"""
    
    def __init__(self):
        self.benchmarks = UtilityBenchmarks()
    
    def analyze_water_consumption(self, water_data: Dict) -> Dict:
        """Analyze water consumption patterns and efficiency"""
        
        total_consumption = water_data.get('total_consumption_m3_h', 50)
        production_rate = water_data.get('production_rate_tph', 167)
        specific_consumption = total_consumption / production_rate
        
        # Breakdown by usage
        usage_breakdown = water_data.get('usage_breakdown', {
            'cooling': 0.6,
            'dust_suppression': 0.2,
            'equipment_cooling': 0.15,
            'other': 0.05
        })
        
        # Cooling system efficiency
        cooling_efficiency = self._analyze_cooling_efficiency(water_data)
        
        # Recycling potential
        recycling_analysis = self._analyze_recycling_potential(water_data)
        
        return {
            'specific_consumption': specific_consumption,
            'benchmark_comparison': specific_consumption / self.benchmarks.water_specific,
            'usage_breakdown': usage_breakdown,
            'cooling_efficiency': cooling_efficiency,
            'recycling_potential': recycling_analysis,
            'optimization_opportunities': self._identify_water_optimization_opportunities(water_data)
        }
    
    def _analyze_cooling_efficiency(self, water_data: Dict) -> Dict:
        """Analyze cooling system efficiency"""
        
        inlet_temp = water_data.get('cooling_inlet_temp_c', 30)
        outlet_temp = water_data.get('cooling_outlet_temp_c', 40)
        ambient_temp = water_data.get('ambient_temp_c', 25)
        
        # Cooling tower efficiency
        cooling_range = outlet_temp - inlet_temp
        approach = inlet_temp - ambient_temp
        
        efficiency = cooling_range / (cooling_range + approach) if (cooling_range + approach) > 0 else 0
        
        # Water consumption for cooling
        cooling_flow = water_data.get('cooling_water_flow_m3_h', 30)
        evaporation_rate = cooling_flow * 0.001 * cooling_range  # Approximate evaporation
        
        return {
            'cooling_efficiency': efficiency,
            'cooling_range': cooling_range,
            'approach_temperature': approach,
            'evaporation_rate': evaporation_rate,
            'makeup_water_requirement': evaporation_rate * 1.3  # Including blowdown
        }
    
    def _analyze_recycling_potential(self, water_data: Dict) -> Dict:
        """Analyze water recycling and reuse potential"""
        
        total_consumption = water_data.get('total_consumption_m3_h', 50)
        
        # Identify recyclable streams
        recyclable_streams = {
            'cooling_blowdown': total_consumption * 0.1,  # 10% of total
            'process_water': total_consumption * 0.05,    # 5% of total
            'rainwater_harvesting': water_data.get('rainfall_potential_m3_year', 1000) / 8760
        }
        
        total_recyclable = sum(recyclable_streams.values())
        recycling_percentage = total_recyclable / total_consumption * 100
        
        # Treatment requirements and costs
        treatment_cost = self._estimate_treatment_cost(recyclable_streams)
        
        return {
            'recyclable_streams': recyclable_streams,
            'total_recyclable_m3_h': total_recyclable,
            'recycling_percentage': recycling_percentage,
            'treatment_cost_usd_m3': treatment_cost,
            'annual_savings_potential': total_recyclable * 8760 * 2.0 - treatment_cost * total_recyclable * 8760
        }
    
    def _estimate_treatment_cost(self, streams: Dict) -> float:
        """Estimate water treatment cost"""
        
        # Simple cost model based on treatment complexity
        treatment_costs = {
            'cooling_blowdown': 0.50,  # $/m³
            'process_water': 1.20,     # $/m³
            'rainwater_harvesting': 0.20  # $/m³
        }
        
        weighted_cost = sum(streams[stream] * treatment_costs.get(stream, 1.0)
                          for stream in streams) / sum(streams.values())
        
        return weighted_cost
    
    def _identify_water_optimization_opportunities(self, water_data: Dict) -> List[Dict]:
        """Identify specific optimization opportunities"""
        
        opportunities = []
        
        # High consumption opportunity
        specific_consumption = (water_data.get('total_consumption_m3_h', 50) / 
                              water_data.get('production_rate_tph', 167))
        if specific_consumption > self.benchmarks.water_specific * 1.2:
            opportunities.append({
                'opportunity': 'Reduce overall water consumption',
                'potential_reduction': '15-25%',
                'investment_required': 'Medium',
                'payback_months': 18
            })
        
        # Cooling system optimization
        cooling_efficiency = self._analyze_cooling_efficiency(water_data)
        if cooling_efficiency['cooling_efficiency'] < 0.7:
            opportunities.append({
                'opportunity': 'Optimize cooling tower performance',
                'potential_reduction': '10-15%',
                'investment_required': 'Low',
                'payback_months': 12
            })
        
        # Recycling opportunity
        recycling_potential = self._analyze_recycling_potential(water_data)
        if recycling_potential['recycling_percentage'] < 20:
            opportunities.append({
                'opportunity': 'Implement water recycling system',
                'potential_reduction': '20-30%',
                'investment_required': 'High',
                'payback_months': 36
            })
        
        return opportunities

class MaterialHandlingOptimizer:
    """Optimizes internal material handling systems"""
    
    def analyze_conveyor_efficiency(self, conveyor_data: Dict) -> Dict:
        """Analyze conveyor system efficiency"""
        
        total_power = conveyor_data.get('total_power_kw', 200)
        material_throughput = conveyor_data.get('throughput_tph', 200)
        specific_energy = total_power / material_throughput if material_throughput > 0 else 0
        
        # Belt efficiency analysis
        belt_speeds = conveyor_data.get('belt_speeds_mps', [1.5, 2.0, 1.8])
        optimal_speeds = self._calculate_optimal_speeds(conveyor_data)
        
        # Motor efficiency
        motor_loads = conveyor_data.get('motor_loads_percent', [75, 85, 60])
        motor_efficiency = self._analyze_motor_efficiency(motor_loads)
        
        return {
            'specific_energy': specific_energy,
            'belt_optimization': {
                'current_speeds': belt_speeds,
                'optimal_speeds': optimal_speeds,
                'energy_savings_potential': self._calculate_speed_optimization_savings(belt_speeds, optimal_speeds)
            },
            'motor_efficiency': motor_efficiency,
            'maintenance_optimization': self._analyze_maintenance_opportunities(conveyor_data)
        }
    
    def _calculate_optimal_speeds(self, conveyor_data: Dict) -> List[float]:
        """Calculate optimal belt speeds for minimum energy consumption"""
        
        # Simplified optimization - in practice would consider:
        # - Material properties, belt loading, power consumption curves
        conveyor_lengths = conveyor_data.get('conveyor_lengths_m', [100, 150, 80])
        material_densities = conveyor_data.get('material_densities_kg_m3', [1500, 1600, 1400])
        
        optimal_speeds = []
        for i, length in enumerate(conveyor_lengths):
            density = material_densities[i] if i < len(material_densities) else 1500
            
            # Optimal speed typically balances throughput vs. power consumption
            # Rule of thumb: 1.5-2.5 m/s for cement plant applications
            optimal_speed = min(2.5, max(1.5, 150 / length * 1.8))
            optimal_speeds.append(optimal_speed)
        
        return optimal_speeds
    
    def _calculate_speed_optimization_savings(self, current_speeds: List, optimal_speeds: List) -> float:
        """Calculate energy savings from speed optimization"""
        
        total_savings = 0
        for i, (current, optimal) in enumerate(zip(current_speeds, optimal_speeds)):
            # Power consumption is roughly cubic with speed for belt conveyors
            if current > 0:
                power_ratio = (optimal / current) ** 3
                energy_reduction = 1 - power_ratio
                total_savings += max(0, energy_reduction)
        
        return (total_savings / len(current_speeds)) * 100 if current_speeds else 0
    
    def _analyze_motor_efficiency(self, motor_loads: List[float]) -> Dict:
        """Analyze motor loading and efficiency"""
        
        # Motor efficiency curves (typical for industrial motors)
        efficiency_curve = {
            20: 0.85, 30: 0.88, 50: 0.92, 75: 0.94, 100: 0.92
        }
        
        motor_efficiencies = []
        for load in motor_loads:
            # Interpolate efficiency
            if load <= 20:
                eff = 0.85
            elif load >= 100:
                eff = 0.92
            else:
                # Linear interpolation between points
                lower_load = max([k for k in efficiency_curve.keys() if k <= load])
                upper_load = min([k for k in efficiency_curve.keys() if k >= load])
                
                if lower_load == upper_load:
                    eff = efficiency_curve[lower_load]
                else:
                    ratio = (load - lower_load) / (upper_load - lower_load)
                    eff = efficiency_curve[lower_load] + ratio * (efficiency_curve[upper_load] - efficiency_curve[lower_load])
            
            motor_efficiencies.append(eff)
        
        avg_efficiency = sum(motor_efficiencies) / len(motor_efficiencies)
        
        return {
            'individual_efficiencies': motor_efficiencies,
            'average_efficiency': avg_efficiency,
            'underloaded_motors': [i for i, load in enumerate(motor_loads) if load < 40],
            'overloaded_motors': [i for i, load in enumerate(motor_loads) if load > 90],
            'optimization_potential': max(0, 0.94 - avg_efficiency) * 100
        }
    
    def _analyze_maintenance_opportunities(self, conveyor_data: Dict) -> List[Dict]:
        """Identify maintenance-based optimization opportunities"""
        
        opportunities = []
        
        # Belt condition analysis
        belt_ages = conveyor_data.get('belt_ages_months', [24, 18, 36])
        for i, age in enumerate(belt_ages):
            if age > 30:  # Assume 30 months is typical replacement time
                opportunities.append({
                    'conveyor': f'Conveyor_{i+1}',
                    'issue': 'Belt replacement due',
                    'energy_impact': '5-15% increase in power consumption',
                    'recommended_action': 'Schedule belt replacement'
                })
        
        # Alignment issues
        power_variations = conveyor_data.get('power_variations_percent', [5, 15, 8])
        for i, variation in enumerate(power_variations):
            if variation > 10:  # High variation suggests alignment issues
                opportunities.append({
                    'conveyor': f'Conveyor_{i+1}',
                    'issue': 'Possible alignment problems',
                    'energy_impact': '10-20% increase in power consumption',
                    'recommended_action': 'Check belt alignment and roller condition'
                })
        
        return opportunities

class UtilityOptimizer:
    """
    Main utility optimization coordinator that combines all utility systems
    """
    
    def __init__(self):
        self.compressed_air = CompressedAirOptimizer()
        self.water_system = WaterSystemOptimizer()
        self.material_handling = MaterialHandlingOptimizer()
    
    def optimize_all_utilities(self, utility_data: Dict) -> Dict:
        """Comprehensive utility optimization analysis"""
        
        # Analyze each utility system
        compressed_air_analysis = self.compressed_air.analyze_system_efficiency(
            utility_data.get('compressed_air', {})
        )
        
        water_analysis = self.water_system.analyze_water_consumption(
            utility_data.get('water_system', {})
        )
        
        material_handling_analysis = self.material_handling.analyze_conveyor_efficiency(
            utility_data.get('material_handling', {})
        )
        
        # Calculate overall savings potential
        total_savings = self._calculate_total_savings(
            compressed_air_analysis, water_analysis, material_handling_analysis
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            compressed_air_analysis, water_analysis, material_handling_analysis
        )
        
        return {
            'compressed_air_analysis': compressed_air_analysis,
            'water_system_analysis': water_analysis,
            'material_handling_analysis': material_handling_analysis,
            'total_savings_potential': total_savings,
            'optimization_recommendations': recommendations,
            'implementation_priority': self._prioritize_recommendations(recommendations)
        }
    
    def _calculate_total_savings(self, air_analysis: Dict, water_analysis: Dict, 
                               handling_analysis: Dict) -> Dict:
        """Calculate total utility optimization savings"""
        
        air_savings = air_analysis.get('potential_savings', {}).get('total', 0)
        water_savings = water_analysis.get('recycling_potential', {}).get('annual_savings_potential', 0)
        
        # Estimate material handling savings (simplified)
        handling_power = 200  # kW assumption
        handling_efficiency_gain = handling_analysis.get('motor_efficiency', {}).get('optimization_potential', 0) / 100
        handling_savings = handling_power * 8760 * 0.08 * handling_efficiency_gain
        
        total_annual_savings = air_savings + water_savings + handling_savings
        
        return {
            'compressed_air_savings': air_savings,
            'water_system_savings': water_savings,
            'material_handling_savings': handling_savings,
            'total_annual_savings': total_annual_savings,
            'roi_months': 24  # Typical ROI for utility optimizations
        }
    
    def _generate_optimization_recommendations(self, air_analysis: Dict, 
                                             water_analysis: Dict, 
                                             handling_analysis: Dict) -> List[Dict]:
        """Generate prioritized optimization recommendations"""
        
        recommendations = []
        
        # Compressed air recommendations
        air_savings = air_analysis.get('potential_savings', {})
        if air_savings.get('leak_repair', 0) > 1000:  # $1000+ savings
            recommendations.append({
                'system': 'Compressed Air',
                'recommendation': 'Implement leak detection and repair program',
                'annual_savings': air_savings['leak_repair'],
                'investment_required': 5000,
                'payback_months': 60,
                'priority': 'High'
            })
        
        if air_savings.get('pressure_optimization', 0) > 2000:
            recommendations.append({
                'system': 'Compressed Air',
                'recommendation': 'Optimize system pressure settings',
                'annual_savings': air_savings['pressure_optimization'],
                'investment_required': 2000,
                'payback_months': 12,
                'priority': 'High'
            })
        
        # Water system recommendations
        water_opportunities = water_analysis.get('optimization_opportunities', [])
        for opportunity in water_opportunities:
            if opportunity.get('payback_months', 50) < 36:
                recommendations.append({
                    'system': 'Water System',
                    'recommendation': opportunity['opportunity'],
                    'potential_reduction': opportunity['potential_reduction'],
                    'investment_required': opportunity['investment_required'],
                    'payback_months': opportunity['payback_months'],
                    'priority': 'High' if opportunity['payback_months'] < 18 else 'Medium'
                })
        
        # Material handling recommendations
        handling_opportunities = handling_analysis.get('maintenance_optimization', [])
        for opportunity in handling_opportunities:
            recommendations.append({
                'system': 'Material Handling',
                'recommendation': opportunity['recommended_action'],
                'energy_impact': opportunity['energy_impact'],
                'priority': 'Medium'
            })
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize recommendations based on ROI and impact"""
        
        def priority_score(rec):
            # Calculate priority score based on payback and savings
            payback = rec.get('payback_months', 36)
            savings = rec.get('annual_savings', 1000)
            
            # Lower payback and higher savings = higher priority
            score = (savings / 1000) / (payback / 12)
            return score
        
        return sorted(recommendations, key=priority_score, reverse=True)
