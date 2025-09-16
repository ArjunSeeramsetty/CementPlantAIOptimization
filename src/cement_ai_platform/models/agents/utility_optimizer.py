"""
Utility Optimization Module for JK Cement Requirements
Optimizes compressed air, water consumption, and internal material handling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

@dataclass
class UtilityTargets:
    """Utility optimization targets"""
    air_efficiency: float = 0.85  # Target compressed air efficiency
    water_efficiency: float = 0.80  # Target water consumption efficiency
    material_handling_efficiency: float = 0.90  # Target material handling efficiency
    power_reduction_pct: float = 5.0  # Target power reduction percentage

@dataclass
class CompressedAirSystem:
    """Compressed air system parameters"""
    compressor_count: int = 3
    compressor_capacity_nm3_min: float = 100.0
    operating_pressure_bar: float = 7.0
    base_efficiency: float = 0.75
    leak_rate_pct: float = 15.0  # Typical leak rate

@dataclass
class WaterSystem:
    """Water system parameters"""
    cooling_water_flow_m3_h: float = 500.0
    process_water_flow_m3_h: float = 200.0
    lubrication_water_flow_m3_h: float = 50.0
    base_efficiency: float = 0.70

@dataclass
class MaterialHandlingSystem:
    """Material handling system parameters"""
    conveyor_count: int = 15
    bucket_elevator_count: int = 8
    pneumatic_conveyor_count: int = 5
    base_efficiency: float = 0.80

class CompressedAirOptimizer:
    """Optimize compressed air system efficiency"""
    
    def __init__(self, system_params: CompressedAirSystem):
        self.system = system_params
        self.leak_detection_threshold = 0.1  # 10% pressure drop threshold
        
    def optimize_compressed_air(self, pressure_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize compressed air system
        
        Args:
            pressure_data: Pressure readings from different points
            
        Returns:
            Optimization recommendations and expected savings
        """
        logger.info("🔄 Optimizing compressed air system...")
        
        # Analyze pressure profile
        pressure_analysis = self._analyze_pressure_profile(pressure_data)
        
        # Detect leaks
        leak_analysis = self._detect_leaks(pressure_data)
        
        # Calculate optimization opportunities
        optimization_opportunities = self._calculate_optimization_opportunities(
            pressure_analysis, leak_analysis
        )
        
        # Generate recommendations
        recommendations = self._generate_air_recommendations(
            pressure_analysis, leak_analysis, optimization_opportunities
        )
        
        return {
            'pressure_analysis': pressure_analysis,
            'leak_analysis': leak_analysis,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'expected_savings': {
                'power_reduction_kw': optimization_opportunities['power_savings'],
                'cost_savings_usd_year': optimization_opportunities['cost_savings'],
                'efficiency_improvement_pct': optimization_opportunities['efficiency_gain']
            }
        }
    
    def _analyze_pressure_profile(self, pressure_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze pressure profile across the system"""
        
        inlet_pressure = pressure_data.get('inlet_pressure_bar', 7.0)
        outlet_pressure = pressure_data.get('outlet_pressure_bar', 6.5)
        critical_points = pressure_data.get('critical_points', {})
        
        # Calculate pressure drop
        pressure_drop = inlet_pressure - outlet_pressure
        pressure_drop_pct = (pressure_drop / inlet_pressure) * 100
        
        # Analyze critical points
        critical_analysis = {}
        for point, pressure in critical_points.items():
            critical_analysis[point] = {
                'pressure_bar': pressure,
                'pressure_drop_from_inlet': inlet_pressure - pressure,
                'status': 'Normal' if pressure > inlet_pressure * 0.9 else 'Low Pressure'
            }
        
        return {
            'inlet_pressure_bar': inlet_pressure,
            'outlet_pressure_bar': outlet_pressure,
            'pressure_drop_bar': pressure_drop,
            'pressure_drop_pct': pressure_drop_pct,
            'critical_points_analysis': critical_analysis,
            'system_status': 'Normal' if pressure_drop_pct < 10 else 'Needs Attention'
        }
    
    def _detect_leaks(self, pressure_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect potential leaks in the system"""
        
        inlet_pressure = pressure_data.get('inlet_pressure_bar', 7.0)
        outlet_pressure = pressure_data.get('outlet_pressure_bar', 6.5)
        
        # Calculate leak rate based on pressure drop
        pressure_drop = inlet_pressure - outlet_pressure
        estimated_leak_rate = (pressure_drop / inlet_pressure) * 100
        
        # Determine leak severity
        if estimated_leak_rate < 5:
            leak_severity = 'Low'
            leak_status = 'Normal'
        elif estimated_leak_rate < 15:
            leak_severity = 'Medium'
            leak_status = 'Attention Required'
        else:
            leak_severity = 'High'
            leak_status = 'Immediate Action Required'
        
        return {
            'estimated_leak_rate_pct': estimated_leak_rate,
            'leak_severity': leak_severity,
            'leak_status': leak_status,
            'pressure_drop_bar': pressure_drop,
            'recommended_action': self._get_leak_action_recommendation(leak_severity)
        }
    
    def _get_leak_action_recommendation(self, leak_severity: str) -> str:
        """Get action recommendation based on leak severity"""
        
        recommendations = {
            'Low': 'Continue monitoring, schedule routine inspection',
            'Medium': 'Schedule leak detection survey within 2 weeks',
            'High': 'Immediate leak detection survey required'
        }
        
        return recommendations.get(leak_severity, 'Schedule inspection')
    
    def _calculate_optimization_opportunities(self, 
                                           pressure_analysis: Dict[str, Any],
                                           leak_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization opportunities and potential savings"""
        
        # Power savings from leak reduction
        leak_rate = leak_analysis['estimated_leak_rate_pct']
        target_leak_rate = 5.0  # Target leak rate
        
        leak_reduction_potential = max(0, leak_rate - target_leak_rate)
        power_savings_from_leaks = leak_reduction_potential * 2.0  # 2 kW per % leak reduction
        
        # Power savings from pressure optimization
        pressure_drop = pressure_analysis['pressure_drop_pct']
        pressure_optimization_savings = max(0, pressure_drop - 5.0) * 1.5  # 1.5 kW per % pressure optimization
        
        # Total power savings
        total_power_savings = power_savings_from_leaks + pressure_optimization_savings
        
        # Cost savings (assuming $0.10/kWh)
        cost_per_kwh = 0.10
        annual_hours = 8760
        annual_cost_savings = total_power_savings * annual_hours * cost_per_kwh
        
        # Efficiency improvement
        current_efficiency = self.system.base_efficiency
        target_efficiency = self.system.base_efficiency + (leak_reduction_potential + pressure_drop) * 0.01
        efficiency_gain = min(0.15, target_efficiency - current_efficiency)  # Cap at 15% improvement
        
        return {
            'power_savings': total_power_savings,
            'cost_savings': annual_cost_savings,
            'efficiency_gain': efficiency_gain * 100,
            'leak_reduction_potential': leak_reduction_potential,
            'pressure_optimization_potential': max(0, pressure_drop - 5.0)
        }
    
    def _generate_air_recommendations(self, 
                                    pressure_analysis: Dict[str, Any],
                                    leak_analysis: Dict[str, Any],
                                    optimization_opportunities: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific recommendations for compressed air optimization"""
        
        recommendations = []
        
        # Leak reduction recommendations
        if leak_analysis['leak_severity'] != 'Low':
            recommendations.append({
                'category': 'Leak Reduction',
                'priority': 'High' if leak_analysis['leak_severity'] == 'High' else 'Medium',
                'action': leak_analysis['recommended_action'],
                'expected_savings': f"{optimization_opportunities['leak_reduction_potential']:.1f}% leak reduction"
            })
        
        # Pressure optimization recommendations
        if pressure_analysis['pressure_drop_pct'] > 8:
            recommendations.append({
                'category': 'Pressure Optimization',
                'priority': 'Medium',
                'action': 'Optimize pressure setpoints and control valves',
                'expected_savings': f"{optimization_opportunities['pressure_optimization_potential']:.1f}% pressure improvement"
            })
        
        # General efficiency recommendations
        recommendations.append({
            'category': 'System Efficiency',
            'priority': 'Low',
            'action': 'Implement variable frequency drives for compressors',
            'expected_savings': '5-10% power reduction'
        })
        
        return recommendations

class WaterConsumptionOptimizer:
    """Optimize water consumption across plant operations"""
    
    def __init__(self, system_params: WaterSystem):
        self.system = system_params
        
    def optimize_water(self, flow_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize water consumption
        
        Args:
            flow_data: Water flow readings from different systems
            
        Returns:
            Optimization recommendations and expected savings
        """
        logger.info("🔄 Optimizing water consumption...")
        
        # Analyze water usage patterns
        usage_analysis = self._analyze_water_usage(flow_data)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_water_optimization_opportunities(usage_analysis)
        
        # Generate recommendations
        recommendations = self._generate_water_recommendations(optimization_opportunities)
        
        return {
            'usage_analysis': usage_analysis,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'expected_savings': {
                'water_reduction_m3_h': optimization_opportunities['water_savings'],
                'cost_savings_usd_year': optimization_opportunities['cost_savings'],
                'efficiency_improvement_pct': optimization_opportunities['efficiency_gain']
            }
        }
    
    def _analyze_water_usage(self, flow_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze water usage patterns"""
        
        cooling_flow = flow_data.get('cooling_water_flow_m3_h', self.system.cooling_water_flow_m3_h)
        process_flow = flow_data.get('process_water_flow_m3_h', self.system.process_water_flow_m3_h)
        lubrication_flow = flow_data.get('lubrication_water_flow_m3_h', self.system.lubrication_water_flow_m3_h)
        
        total_flow = cooling_flow + process_flow + lubrication_flow
        
        # Calculate usage efficiency
        cooling_efficiency = min(1.0, cooling_flow / self.system.cooling_water_flow_m3_h)
        process_efficiency = min(1.0, process_flow / self.system.process_water_flow_m3_h)
        lubrication_efficiency = min(1.0, lubrication_flow / self.system.lubrication_water_flow_m3_h)
        
        overall_efficiency = (cooling_efficiency + process_efficiency + lubrication_efficiency) / 3
        
        return {
            'total_flow_m3_h': total_flow,
            'cooling_flow_m3_h': cooling_flow,
            'process_flow_m3_h': process_flow,
            'lubrication_flow_m3_h': lubrication_flow,
            'cooling_efficiency': cooling_efficiency,
            'process_efficiency': process_efficiency,
            'lubrication_efficiency': lubrication_efficiency,
            'overall_efficiency': overall_efficiency,
            'usage_status': 'Optimal' if overall_efficiency > 0.8 else 'Needs Optimization'
        }
    
    def _identify_water_optimization_opportunities(self, usage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify water optimization opportunities"""
        
        # Calculate potential savings
        cooling_savings = max(0, usage_analysis['cooling_flow_m3_h'] * 0.1)  # 10% reduction potential
        process_savings = max(0, usage_analysis['process_flow_m3_h'] * 0.15)  # 15% reduction potential
        lubrication_savings = max(0, usage_analysis['lubrication_flow_m3_h'] * 0.05)  # 5% reduction potential
        
        total_water_savings = cooling_savings + process_savings + lubrication_savings
        
        # Cost savings (assuming $2/m³ water cost)
        water_cost_per_m3 = 2.0
        annual_hours = 8760
        annual_cost_savings = total_water_savings * annual_hours * water_cost_per_m3
        
        # Efficiency improvement
        current_efficiency = usage_analysis['overall_efficiency']
        target_efficiency = min(0.95, current_efficiency + 0.1)  # 10% improvement potential
        efficiency_gain = target_efficiency - current_efficiency
        
        return {
            'water_savings': total_water_savings,
            'cost_savings': annual_cost_savings,
            'efficiency_gain': efficiency_gain * 100,
            'cooling_savings_potential': cooling_savings,
            'process_savings_potential': process_savings,
            'lubrication_savings_potential': lubrication_savings
        }
    
    def _generate_water_recommendations(self, optimization_opportunities: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate water optimization recommendations"""
        
        recommendations = []
        
        # Cooling water optimization
        if optimization_opportunities['cooling_savings_potential'] > 10:
            recommendations.append({
                'category': 'Cooling Water',
                'priority': 'Medium',
                'action': 'Implement closed-loop cooling system with heat recovery',
                'expected_savings': f"{optimization_opportunities['cooling_savings_potential']:.1f} m³/h reduction"
            })
        
        # Process water optimization
        if optimization_opportunities['process_savings_potential'] > 15:
            recommendations.append({
                'category': 'Process Water',
                'priority': 'High',
                'action': 'Optimize water usage in grinding and mixing processes',
                'expected_savings': f"{optimization_opportunities['process_savings_potential']:.1f} m³/h reduction"
            })
        
        # General recommendations
        recommendations.append({
            'category': 'Water Management',
            'priority': 'Low',
            'action': 'Implement water recycling and treatment system',
            'expected_savings': '20-30% overall water reduction'
        })
        
        return recommendations

class MaterialHandlingOptimizer:
    """Optimize internal material handling systems"""
    
    def __init__(self, system_params: MaterialHandlingSystem):
        self.system = system_params
        
    def optimize_material_handling(self, handling_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize material handling systems
        
        Args:
            handling_data: Material handling system data
            
        Returns:
            Optimization recommendations and expected savings
        """
        logger.info("🔄 Optimizing material handling systems...")
        
        # Analyze handling efficiency
        efficiency_analysis = self._analyze_handling_efficiency(handling_data)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_handling_optimization_opportunities(efficiency_analysis)
        
        # Generate recommendations
        recommendations = self._generate_handling_recommendations(optimization_opportunities)
        
        return {
            'efficiency_analysis': efficiency_analysis,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'expected_savings': {
                'power_reduction_kw': optimization_opportunities['power_savings'],
                'cost_savings_usd_year': optimization_opportunities['cost_savings'],
                'efficiency_improvement_pct': optimization_opportunities['efficiency_gain']
            }
        }
    
    def _analyze_handling_efficiency(self, handling_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze material handling efficiency"""
        
        conveyor_efficiency = handling_data.get('conveyor_efficiency', 0.85)
        elevator_efficiency = handling_data.get('elevator_efficiency', 0.80)
        pneumatic_efficiency = handling_data.get('pneumatic_efficiency', 0.75)
        
        # Calculate overall efficiency
        overall_efficiency = (conveyor_efficiency + elevator_efficiency + pneumatic_efficiency) / 3
        
        # Analyze power consumption
        conveyor_power = handling_data.get('conveyor_power_kw', 200.0)
        elevator_power = handling_data.get('elevator_power_kw', 150.0)
        pneumatic_power = handling_data.get('pneumatic_power_kw', 100.0)
        
        total_power = conveyor_power + elevator_power + pneumatic_power
        
        return {
            'conveyor_efficiency': conveyor_efficiency,
            'elevator_efficiency': elevator_efficiency,
            'pneumatic_efficiency': pneumatic_efficiency,
            'overall_efficiency': overall_efficiency,
            'conveyor_power_kw': conveyor_power,
            'elevator_power_kw': elevator_power,
            'pneumatic_power_kw': pneumatic_power,
            'total_power_kw': total_power,
            'efficiency_status': 'Good' if overall_efficiency > 0.8 else 'Needs Improvement'
        }
    
    def _identify_handling_optimization_opportunities(self, efficiency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify material handling optimization opportunities"""
        
        # Power savings potential
        conveyor_savings = efficiency_analysis['conveyor_power_kw'] * 0.1  # 10% savings
        elevator_savings = efficiency_analysis['elevator_power_kw'] * 0.15  # 15% savings
        pneumatic_savings = efficiency_analysis['pneumatic_power_kw'] * 0.2  # 20% savings
        
        total_power_savings = conveyor_savings + elevator_savings + pneumatic_savings
        
        # Cost savings
        cost_per_kwh = 0.10
        annual_hours = 8760
        annual_cost_savings = total_power_savings * annual_hours * cost_per_kwh
        
        # Efficiency improvement
        current_efficiency = efficiency_analysis['overall_efficiency']
        target_efficiency = min(0.95, current_efficiency + 0.1)
        efficiency_gain = target_efficiency - current_efficiency
        
        return {
            'power_savings': total_power_savings,
            'cost_savings': annual_cost_savings,
            'efficiency_gain': efficiency_gain * 100,
            'conveyor_savings_potential': conveyor_savings,
            'elevator_savings_potential': elevator_savings,
            'pneumatic_savings_potential': pneumatic_savings
        }
    
    def _generate_handling_recommendations(self, optimization_opportunities: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate material handling optimization recommendations"""
        
        recommendations = []
        
        # Conveyor optimization
        if optimization_opportunities['conveyor_savings_potential'] > 15:
            recommendations.append({
                'category': 'Conveyor Systems',
                'priority': 'Medium',
                'action': 'Implement variable speed drives and load-based control',
                'expected_savings': f"{optimization_opportunities['conveyor_savings_potential']:.1f} kW reduction"
            })
        
        # Elevator optimization
        if optimization_opportunities['elevator_savings_potential'] > 10:
            recommendations.append({
                'category': 'Bucket Elevators',
                'priority': 'High',
                'action': 'Optimize bucket filling and discharge mechanisms',
                'expected_savings': f"{optimization_opportunities['elevator_savings_potential']:.1f} kW reduction"
            })
        
        # Pneumatic optimization
        if optimization_opportunities['pneumatic_savings_potential'] > 5:
            recommendations.append({
                'category': 'Pneumatic Conveying',
                'priority': 'Medium',
                'action': 'Implement pressure-based control and leak detection',
                'expected_savings': f"{optimization_opportunities['pneumatic_savings_potential']:.1f} kW reduction"
            })
        
        return recommendations

class UtilityOptimizer:
    """
    Main utility optimization class integrating all utility systems.
    Implements JK Cement's requirement for utility optimization.
    """

    def __init__(self, config_path: str = "config/plant_config.yml"):
        # Initialize utility systems
        self.air_system = CompressedAirSystem()
        self.water_system = WaterSystem()
        self.material_system = MaterialHandlingSystem()
        
        # Initialize optimizers
        self.air_optimizer = CompressedAirOptimizer(self.air_system)
        self.water_optimizer = WaterConsumptionOptimizer(self.water_system)
        self.material_optimizer = MaterialHandlingOptimizer(self.material_system)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.targets = UtilityTargets()
        
        # Optimization history
        self.optimization_history = []
        
        logger.info("✅ Utility Optimizer initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load plant configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def optimize_all_utilities(self, 
                             pressure_data: Dict[str, float],
                             flow_data: Dict[str, float],
                             handling_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize all utility systems
        
        Args:
            pressure_data: Compressed air pressure data
            flow_data: Water flow data
            handling_data: Material handling data
            
        Returns:
            Comprehensive optimization results
        """
        logger.info("🔄 Starting comprehensive utility optimization...")
        
        # Optimize compressed air
        air_results = self.air_optimizer.optimize_compressed_air(pressure_data)
        
        # Optimize water consumption
        water_results = self.water_optimizer.optimize_water(flow_data)
        
        # Optimize material handling
        material_results = self.material_optimizer.optimize_material_handling(handling_data)
        
        # Calculate total savings
        total_savings = self._calculate_total_savings(air_results, water_results, material_results)
        
        # Generate priority recommendations
        priority_recommendations = self._generate_priority_recommendations(
            air_results, water_results, material_results
        )
        
        # Store optimization results
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'air_results': air_results,
            'water_results': water_results,
            'material_results': material_results,
            'total_savings': total_savings,
            'priority_recommendations': priority_recommendations
        }
        self.optimization_history.append(optimization_record)
        
        # Keep only last 50 records
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
        
        logger.info("✅ Comprehensive utility optimization completed")
        
        return {
            'air_optimization': air_results,
            'water_optimization': water_results,
            'material_handling_optimization': material_results,
            'total_savings': total_savings,
            'priority_recommendations': priority_recommendations,
            'optimization_summary': self._generate_optimization_summary(total_savings)
        }

    def _calculate_total_savings(self, 
                               air_results: Dict[str, Any],
                               water_results: Dict[str, Any],
                               material_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total savings across all utility systems"""
        
        # Power savings
        total_power_savings = (
            air_results['expected_savings']['power_reduction_kw'] +
            material_results['expected_savings']['power_reduction_kw']
        )
        
        # Cost savings
        total_cost_savings = (
            air_results['expected_savings']['cost_savings_usd_year'] +
            water_results['expected_savings']['cost_savings_usd_year'] +
            material_results['expected_savings']['cost_savings_usd_year']
        )
        
        # Efficiency improvement
        avg_efficiency_gain = (
            air_results['expected_savings']['efficiency_improvement_pct'] +
            water_results['expected_savings']['efficiency_improvement_pct'] +
            material_results['expected_savings']['efficiency_improvement_pct']
        ) / 3
        
        return {
            'total_power_savings_kw': total_power_savings,
            'total_cost_savings_usd_year': total_cost_savings,
            'average_efficiency_gain_pct': avg_efficiency_gain,
            'roi_period_years': self._calculate_roi_period(total_cost_savings),
            'carbon_footprint_reduction_tco2_year': total_power_savings * 0.5  # 0.5 tCO2 per kW
        }

    def _calculate_roi_period(self, annual_savings: float) -> float:
        """Calculate return on investment period"""
        
        # Assume average implementation cost of $100,000
        implementation_cost = 100000.0
        
        if annual_savings > 0:
            roi_period = implementation_cost / annual_savings
        else:
            roi_period = float('inf')
        
        return roi_period

    def _generate_priority_recommendations(self, 
                                         air_results: Dict[str, Any],
                                         water_results: Dict[str, Any],
                                         material_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate priority-based recommendations"""
        
        recommendations = []
        
        # High priority recommendations
        if air_results['leak_analysis']['leak_severity'] == 'High':
            recommendations.append({
                'priority': 'High',
                'system': 'Compressed Air',
                'action': 'Immediate leak detection and repair',
                'expected_impact': 'High power savings'
            })
        
        if water_results['optimization_opportunities']['process_savings_potential'] > 20:
            recommendations.append({
                'priority': 'High',
                'system': 'Water',
                'action': 'Process water optimization',
                'expected_impact': 'Significant water savings'
            })
        
        # Medium priority recommendations
        if material_results['optimization_opportunities']['elevator_savings_potential'] > 15:
            recommendations.append({
                'priority': 'Medium',
                'system': 'Material Handling',
                'action': 'Elevator optimization',
                'expected_impact': 'Moderate power savings'
            })
        
        return recommendations

    def _generate_optimization_summary(self, total_savings: Dict[str, Any]) -> str:
        """Generate optimization summary"""
        
        power_savings = total_savings['total_power_savings_kw']
        cost_savings = total_savings['total_cost_savings_usd_year']
        efficiency_gain = total_savings['average_efficiency_gain_pct']
        
        summary = f"""
        Utility Optimization Summary:
        - Total Power Savings: {power_savings:.1f} kW
        - Annual Cost Savings: ${cost_savings:,.0f}
        - Average Efficiency Gain: {efficiency_gain:.1f}%
        - ROI Period: {total_savings['roi_period_years']:.1f} years
        - Carbon Footprint Reduction: {total_savings['carbon_footprint_reduction_tco2_year']:.1f} tCO2/year
        
        Status: {'Excellent' if efficiency_gain > 10 else 'Good' if efficiency_gain > 5 else 'Needs Improvement'}
        """
        
        return summary.strip()

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history

    def export_optimization_data(self) -> Dict[str, Any]:
        """Export optimization data for analysis"""
        
        return {
            'optimization_history': self.optimization_history[-20:],  # Last 20 records
            'system_configurations': {
                'air_system': self.air_system,
                'water_system': self.water_system,
                'material_system': self.material_system
            },
            'targets': self.targets,
            'export_timestamp': datetime.now().isoformat()
        }
