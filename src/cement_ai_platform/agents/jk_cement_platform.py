"""
JK Cement Digital Twin Platform - Unified Agent Integration
Integrates all five critical agents for comprehensive cement plant optimization
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import all agent modules
from .alternative_fuel_optimizer import AlternativeFuelOptimizer
from .cement_plant_gpt import CementPlantGPT
from .unified_kiln_cooler_controller import UnifiedKilnCoolerController
from .utility_optimizer import UtilityOptimizer
from .plant_anomaly_detector import PlantAnomalyDetector

logger = logging.getLogger(__name__)

class JKCementDigitalTwinPlatform:
    """
    Unified platform integrating all JK Cement requirements:
    1. Alternative Fuel Optimization
    2. Cement Plant GPT Interface
    3. Unified Kiln-Cooler Controller
    4. Utility Optimization
    5. Plant Anomaly Detection
    """
    
    def __init__(self, config_file: str = "config/plant_config.yml"):
        self.config_file = config_file
        self.plant_config = self._load_plant_config()
        
        # Initialize all agents
        self.alternative_fuel_optimizer = AlternativeFuelOptimizer()
        self.cement_plant_gpt = CementPlantGPT()
        self.kiln_cooler_controller = UnifiedKilnCoolerController()
        self.utility_optimizer = UtilityOptimizer()
        self.anomaly_detector = PlantAnomalyDetector()
        
        # Platform state
        self.current_plant_data = {}
        self.optimization_history = []
        self.performance_metrics = {}
        
        logger.info("🚀 JK Cement Digital Twin Platform initialized successfully")
    
    def _load_plant_config(self) -> Dict:
        """Load plant configuration"""
        try:
            import yaml
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load plant config: {e}")
            return {}
    
    def process_plant_data(self, plant_data: Dict) -> Dict:
        """
        Main processing function that orchestrates all agents
        
        Args:
            plant_data: Current plant sensor data and KPIs
            
        Returns:
            Comprehensive analysis and recommendations
        """
        
        logger.info("🔄 Processing plant data through unified platform...")
        
        # Store current data
        self.current_plant_data = plant_data
        timestamp = datetime.now().isoformat()
        
        # 1. Alternative Fuel Optimization
        fuel_optimization_results = self._optimize_alternative_fuels(plant_data)
        
        # 2. Unified Kiln-Cooler Control
        control_results = self._compute_unified_control_setpoints(plant_data)
        
        # 3. Utility Optimization
        utility_optimization_results = self._optimize_utilities(plant_data)
        
        # 4. Plant Anomaly Detection
        anomaly_results = self._detect_anomalies(plant_data)
        
        # 5. GPT Analysis and Recommendations
        gpt_results = self._generate_gpt_analysis(plant_data, {
            'fuel_optimization': fuel_optimization_results,
            'control_setpoints': control_results,
            'utility_optimization': utility_optimization_results,
            'anomaly_detection': anomaly_results
        })
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': timestamp,
            'plant_data': plant_data,
            'fuel_optimization': fuel_optimization_results,
            'control_setpoints': control_results,
            'utility_optimization': utility_optimization_results,
            'anomaly_detection': anomaly_results,
            'gpt_analysis': gpt_results,
            'overall_plant_status': self._assess_overall_plant_status(
                fuel_optimization_results, control_results, 
                utility_optimization_results, anomaly_results
            ),
            'recommendations': self._generate_unified_recommendations(
                fuel_optimization_results, control_results,
                utility_optimization_results, anomaly_results
            )
        }
        
        # Store in history
        self.optimization_history.append(comprehensive_results)
        
        # Update performance metrics
        self._update_performance_metrics(comprehensive_results)
        
        logger.info("✅ Plant data processing completed successfully")
        
        return comprehensive_results
    
    def _optimize_alternative_fuels(self, plant_data: Dict) -> Dict:
        """Optimize alternative fuel blend for maximum TSR"""
        
        logger.info("🔄 Optimizing alternative fuel blend...")
        
        # Extract fuel-related data
        available_fuels = plant_data.get('available_fuels', {
            'coal': 1000,  # kg/h
            'petcoke': 200,
            'rdf': 150,
            'biomass': 100,
            'tire_chips': 50
        })
        
        fuel_costs = plant_data.get('fuel_costs', {
            'coal': 0.08,  # $/kg
            'petcoke': 0.12,
            'rdf': 0.05,
            'biomass': 0.06,
            'tire_chips': 0.04
        })
        
        quality_constraints = plant_data.get('quality_constraints', {
            'max_chlorine': 0.15,
            'max_sulfur': 2.0,
            'max_alkali': 3.0
        })
        
        production_rate = plant_data.get('production_rate_tph', 167)
        
        # Run optimization
        optimization_result = self.alternative_fuel_optimizer.optimize_fuel_blend(
            available_fuels=available_fuels,
            fuel_costs=fuel_costs,
            quality_constraints=quality_constraints,
            production_rate=production_rate
        )
        
        # Generate RDF scenarios
        rdf_scenarios = self.alternative_fuel_optimizer.generate_rdf_scenarios({
            'base_production': production_rate,
            'current_tsr': optimization_result.get('tsr_achieved', 0.15)
        })
        
        return {
            'optimization_result': optimization_result,
            'rdf_scenarios': rdf_scenarios,
            'tsr_achieved': optimization_result.get('tsr_achieved', 0.15),
            'quality_impact': optimization_result.get('quality_impact', {}),
            'cost_analysis': optimization_result.get('total_cost', 0)
        }
    
    def _compute_unified_control_setpoints(self, plant_data: Dict) -> Dict:
        """Compute unified kiln-cooler control setpoints"""
        
        logger.info("🔄 Computing unified control setpoints...")
        
        # Extract sensor data for control
        sensor_data = {
            'free_lime_percent': plant_data.get('free_lime_percent', 1.0),
            'burning_zone_temp_c': plant_data.get('burning_zone_temp_c', 1450),
            'cooler_outlet_temp_c': plant_data.get('cooler_outlet_temp_c', 100),
            'secondary_air_temp_c': plant_data.get('secondary_air_temp_c', 950),
            'nox_mg_nm3': plant_data.get('nox_mg_nm3', 500),
            'feed_rate_tph': plant_data.get('feed_rate_tph', 167),
            'fuel_rate_tph': plant_data.get('fuel_rate_tph', 16.3),
            'kiln_speed_rpm': plant_data.get('kiln_speed_rpm', 3.2),
            'o2_percent': plant_data.get('o2_percent', 3.0),
            'kiln_torque_percent': plant_data.get('kiln_torque_percent', 70),
            'cooler_speed_rpm': plant_data.get('cooler_speed_rpm', 10)
        }
        
        # Compute unified setpoints
        control_results = self.kiln_cooler_controller.compute_unified_setpoints(
            sensor_data=sensor_data,
            dt=1.0  # 1 minute time step
        )
        
        return {
            'setpoints': control_results['setpoints'],
            'performance_prediction': control_results['performance_prediction'],
            'coordination_factors': control_results['coordination_factors'],
            'control_health': control_results['control_health']
        }
    
    def _optimize_utilities(self, plant_data: Dict) -> Dict:
        """Optimize utility systems (compressed air, water, material handling)"""
        
        logger.info("🔄 Optimizing utility systems...")
        
        # Prepare utility data
        utility_data = {
            'compressed_air': {
                'total_consumption_nm3_h': plant_data.get('air_consumption_nm3_h', 1000),
                'baseline_consumption_nm3_h': plant_data.get('baseline_air_consumption_nm3_h', 800),
                'system_pressure_bar': plant_data.get('air_pressure_bar', 8.0),
                'compressor_power_kw': plant_data.get('compressor_power_kw', 150),
                'production_rate_tph': plant_data.get('production_rate_tph', 167)
            },
            'water_system': {
                'total_consumption_m3_h': plant_data.get('water_consumption_m3_h', 50),
                'production_rate_tph': plant_data.get('production_rate_tph', 167),
                'cooling_inlet_temp_c': plant_data.get('cooling_inlet_temp_c', 30),
                'cooling_outlet_temp_c': plant_data.get('cooling_outlet_temp_c', 40),
                'ambient_temp_c': plant_data.get('ambient_temp_c', 25)
            },
            'material_handling': {
                'total_power_kw': plant_data.get('conveyor_power_kw', 200),
                'throughput_tph': plant_data.get('material_throughput_tph', 200),
                'belt_speeds_mps': plant_data.get('belt_speeds_mps', [1.5, 2.0, 1.8]),
                'motor_loads_percent': plant_data.get('motor_loads_percent', [75, 85, 60])
            }
        }
        
        # Run utility optimization
        optimization_results = self.utility_optimizer.optimize_all_utilities(utility_data)
        
        return optimization_results
    
    def _detect_anomalies(self, plant_data: Dict) -> Dict:
        """Detect plant anomalies and equipment health issues"""
        
        logger.info("🔄 Detecting plant anomalies...")
        
        # Prepare plant data for anomaly detection
        plant_monitoring_data = {
            'equipment_list': [
                {'equipment_name': 'Raw_Mill_01', 'equipment_type': 'mill'},
                {'equipment_name': 'Kiln_01', 'equipment_type': 'kiln'},
                {'equipment_name': 'Cement_Mill_01', 'equipment_type': 'mill'},
                {'equipment_name': 'ID_Fan_01', 'equipment_type': 'fan'}
            ],
            'equipment_data': {
                'Raw_Mill_01': {
                    'vibration_x_mm_s': plant_data.get('raw_mill_vibration_x', 3.0),
                    'vibration_y_mm_s': plant_data.get('raw_mill_vibration_y', 3.0),
                    'vibration_z_mm_s': plant_data.get('raw_mill_vibration_z', 2.5),
                    'bearing_temperature_c': plant_data.get('raw_mill_bearing_temp', 70),
                    'motor_temperature_c': plant_data.get('raw_mill_motor_temp', 80),
                    'power_kw': plant_data.get('raw_mill_power_kw', 2000),
                    'throughput_tph': plant_data.get('raw_mill_throughput_tph', 80)
                },
                'Kiln_01': {
                    'fuel_rate_tph': plant_data.get('fuel_rate_tph', 16),
                    'production_rate_tph': plant_data.get('production_rate_tph', 167),
                    'free_lime_percent': plant_data.get('free_lime_percent', 1.0),
                    'burning_zone_temp_c': plant_data.get('burning_zone_temp_c', 1450)
                },
                'Cement_Mill_01': {
                    'vibration_x_mm_s': plant_data.get('cement_mill_vibration_x', 3.5),
                    'vibration_y_mm_s': plant_data.get('cement_mill_vibration_y', 3.5),
                    'vibration_z_mm_s': plant_data.get('cement_mill_vibration_z', 3.0),
                    'power_kw': plant_data.get('cement_mill_power_kw', 3000),
                    'throughput_tph': plant_data.get('cement_mill_throughput_tph', 120)
                },
                'ID_Fan_01': {
                    'flow_nm3_h': plant_data.get('id_fan_flow_nm3_h', 50000),
                    'pressure_pa': plant_data.get('id_fan_pressure_pa', -150),
                    'current_phase_a': plant_data.get('id_fan_current_a', 100),
                    'current_phase_b': plant_data.get('id_fan_current_b', 100),
                    'current_phase_c': plant_data.get('id_fan_current_c', 100)
                }
            },
            'process_data': {
                'kiln_speed_rpm': plant_data.get('kiln_speed_rpm', 3.2),
                'feed_rate_tph': plant_data.get('feed_rate_tph', 167),
                'fuel_rate_tph': plant_data.get('fuel_rate_tph', 16),
                'burning_zone_temp_c': plant_data.get('burning_zone_temp_c', 1450),
                'free_lime_percent': plant_data.get('free_lime_percent', 1.0),
                'o2_percent': plant_data.get('o2_percent', 3.0),
                'nox_mg_nm3': plant_data.get('nox_mg_nm3', 500),
                'co_mg_nm3': plant_data.get('co_mg_nm3', 100)
            }
        }
        
        # Run anomaly detection
        anomaly_results = self.anomaly_detector.monitor_plant_status(plant_monitoring_data)
        
        return anomaly_results
    
    def _generate_gpt_analysis(self, plant_data: Dict, analysis_context: Dict) -> Dict:
        """Generate GPT analysis and recommendations"""
        
        logger.info("🔄 Generating GPT analysis...")
        
        # Prepare context for GPT
        gpt_context = {
            'current_plant_data': plant_data,
            'fuel_optimization_results': analysis_context['fuel_optimization'],
            'control_results': analysis_context['control_setpoints'],
            'utility_optimization_results': analysis_context['utility_optimization'],
            'anomaly_results': analysis_context['anomaly_detection']
        }
        
        # Generate comprehensive analysis
        analysis_query = """
        Based on the current plant data and optimization results, provide:
        1. Overall plant performance assessment
        2. Key optimization opportunities
        3. Critical issues requiring attention
        4. Recommendations for next shift
        5. Long-term improvement strategies
        """
        
        gpt_response = self.cement_plant_gpt.query(analysis_query, gpt_context)
        
        # Generate shift report
        shift_report = self.cement_plant_gpt.generate_shift_report(plant_data)
        
        return {
            'gpt_response': gpt_response,
            'shift_report': shift_report,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _assess_overall_plant_status(self, fuel_results: Dict, control_results: Dict,
                                   utility_results: Dict, anomaly_results: Dict) -> Dict:
        """Assess overall plant status based on all agent results"""
        
        # Extract key metrics
        tsr_achieved = fuel_results.get('tsr_achieved', 0.15)
        control_health = control_results.get('control_health', {}).get('health_score', 0.8)
        plant_health_score = anomaly_results.get('plant_health_score', 0.8)
        total_savings = utility_results.get('total_savings_potential', {}).get('total_annual_savings', 0)
        
        # Calculate overall performance score
        performance_score = (
            min(tsr_achieved / 0.15, 1.0) * 0.25 +  # TSR achievement (25%)
            control_health * 0.25 +                  # Control health (25%)
            plant_health_score * 0.25 +             # Plant health (25%)
            min(total_savings / 1000000, 1.0) * 0.25  # Savings potential (25%)
        )
        
        # Categorize status
        if performance_score >= 0.8:
            status = 'Excellent'
        elif performance_score >= 0.6:
            status = 'Good'
        elif performance_score >= 0.4:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'overall_performance_score': performance_score,
            'status': status,
            'tsr_achievement': tsr_achieved,
            'control_health': control_health,
            'plant_health': plant_health_score,
            'utility_savings_potential': total_savings,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _generate_unified_recommendations(self, fuel_results: Dict, control_results: Dict,
                                        utility_results: Dict, anomaly_results: Dict) -> List[Dict]:
        """Generate unified recommendations from all agents"""
        
        recommendations = []
        
        # Fuel optimization recommendations
        if fuel_results.get('tsr_achieved', 0) < 0.15:
            recommendations.append({
                'category': 'Alternative Fuels',
                'priority': 'High',
                'recommendation': 'Increase alternative fuel usage to achieve 15% TSR target',
                'expected_benefit': 'Cost savings and environmental compliance',
                'implementation_time': '1-3 months'
            })
        
        # Control optimization recommendations
        control_health = control_results.get('control_health', {}).get('health_score', 0.8)
        if control_health < 0.7:
            recommendations.append({
                'category': 'Process Control',
                'priority': 'Medium',
                'recommendation': 'Tune PID controllers for better performance',
                'expected_benefit': 'Improved process stability and quality',
                'implementation_time': '1-2 weeks'
            })
        
        # Utility optimization recommendations
        utility_recs = utility_results.get('optimization_recommendations', [])
        for rec in utility_recs[:3]:  # Top 3 utility recommendations
            recommendations.append({
                'category': 'Utility Optimization',
                'priority': rec.get('priority', 'Medium'),
                'recommendation': rec.get('recommendation', 'Optimize utility system'),
                'expected_benefit': f"Annual savings: ${rec.get('annual_savings', 0):,.0f}",
                'implementation_time': f"{rec.get('payback_months', 24)} months"
            })
        
        # Anomaly-based recommendations
        anomaly_recs = anomaly_results.get('recommendations', [])
        for rec in anomaly_recs[:2]:  # Top 2 anomaly recommendations
            recommendations.append({
                'category': 'Equipment Maintenance',
                'priority': 'High' if rec.get('urgency') in ['Emergency', 'Critical'] else 'Medium',
                'recommendation': rec.get('recommendation', 'Equipment maintenance required'),
                'expected_benefit': 'Prevent equipment failure and downtime',
                'implementation_time': 'Immediate' if rec.get('urgency') == 'Emergency' else '1-4 weeks'
            })
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _update_performance_metrics(self, results: Dict):
        """Update platform performance metrics"""
        
        timestamp = results['timestamp']
        
        # Key performance indicators
        kpis = {
            'tsr_achieved': results['fuel_optimization'].get('tsr_achieved', 0.15),
            'plant_health_score': results['anomaly_detection'].get('plant_health_score', 0.8),
            'control_health_score': results['control_setpoints'].get('control_health', {}).get('health_score', 0.8),
            'utility_savings_potential': results['utility_optimization'].get('total_savings_potential', {}).get('total_annual_savings', 0),
            'overall_performance_score': results['overall_plant_status'].get('overall_performance_score', 0.8)
        }
        
        self.performance_metrics[timestamp] = kpis
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary over specified period"""
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
        
        recent_metrics = {
            timestamp: metrics for timestamp, metrics in self.performance_metrics.items()
            if datetime.fromisoformat(timestamp).timestamp() > cutoff_date
        }
        
        if not recent_metrics:
            return {'message': 'No recent performance data available'}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['tsr_achieved', 'plant_health_score', 'control_health_score', 
                      'utility_savings_potential', 'overall_performance_score']:
            values = [metrics[metric] for metrics in recent_metrics.values() if metric in metrics]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        return {
            'period_days': days,
            'data_points': len(recent_metrics),
            'average_metrics': avg_metrics,
            'trend_analysis': self._analyze_performance_trends(recent_metrics)
        }
    
    def _analyze_performance_trends(self, metrics: Dict) -> Dict:
        """Analyze performance trends"""
        
        if len(metrics) < 2:
            return {'trend': 'Insufficient data'}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics.items())
        
        # Calculate trends
        first_performance = sorted_metrics[0][1].get('overall_performance_score', 0.8)
        last_performance = sorted_metrics[-1][1].get('overall_performance_score', 0.8)
        
        performance_trend = 'Improving' if last_performance > first_performance else 'Declining'
        
        return {
            'performance_trend': performance_trend,
            'trend_magnitude': abs(last_performance - first_performance),
            'recommendation': 'Continue current optimization strategies' if performance_trend == 'Improving' else 'Review and adjust optimization parameters'
        }
    
    def export_results(self, results: Dict, format: str = 'json') -> str:
        """Export results in specified format"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'json':
            filename = f"jk_cement_platform_results_{timestamp}.json"
            
            # Convert pandas Timestamps and other non-serializable objects
            def convert_for_json(obj):
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                else:
                    return obj
            
            results_serializable = convert_for_json(results)
            
            with open(filename, 'w') as f:
                json.dump(results_serializable, f, indent=2)
        
        elif format.lower() == 'csv':
            # Export key metrics to CSV
            filename = f"jk_cement_platform_metrics_{timestamp}.csv"
            
            metrics_data = []
            for timestamp_key, metrics in self.performance_metrics.items():
                row = {'timestamp': timestamp_key}
                row.update(metrics)
                metrics_data.append(row)
            
            df = pd.DataFrame(metrics_data)
            df.to_csv(filename, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to {filename}")
        return filename
    
    def train_anomaly_models(self, historical_data: pd.DataFrame):
        """Train anomaly detection models on historical data"""
        
        logger.info("🔄 Training anomaly detection models...")
        
        try:
            self.anomaly_detector.train_process_models(historical_data)
            logger.info("✅ Anomaly detection models trained successfully")
        except Exception as e:
            logger.error(f"❌ Failed to train anomaly models: {e}")
    
    def get_platform_status(self) -> Dict:
        """Get current platform status and health"""
        
        return {
            'platform_status': 'Operational',
            'agents_initialized': {
                'alternative_fuel_optimizer': True,
                'cement_plant_gpt': True,
                'kiln_cooler_controller': True,
                'utility_optimizer': True,
                'anomaly_detector': True
            },
            'optimization_history_count': len(self.optimization_history),
            'performance_metrics_count': len(self.performance_metrics),
            'last_processing_time': self.optimization_history[-1]['timestamp'] if self.optimization_history else None,
            'platform_version': '1.0.0',
            'jk_cement_requirements_coverage': {
                'alternative_fuel_optimization': '✅ Implemented',
                'cement_plant_gpt_interface': '✅ Implemented',
                'unified_kiln_cooler_controller': '✅ Implemented',
                'utility_optimization': '✅ Implemented',
                'plant_anomaly_detection': '✅ Implemented'
            }
        }

def create_unified_platform(config_file: str = "config/plant_config.yml") -> JKCementDigitalTwinPlatform:
    """
    Factory function to create and initialize the unified JK Cement platform
    
    Args:
        config_file: Path to plant configuration file
        
    Returns:
        Initialized JKCementDigitalTwinPlatform instance
    """
    
    logger.info("🚀 Creating JK Cement Digital Twin Platform...")
    
    platform = JKCementDigitalTwinPlatform(config_file)
    
    logger.info("✅ JK Cement Digital Twin Platform created successfully")
    logger.info("📋 Platform covers all JK Cement requirements:")
    logger.info("   ✅ Alternative Fuel Optimization")
    logger.info("   ✅ Cement Plant GPT Interface")
    logger.info("   ✅ Unified Kiln-Cooler Controller")
    logger.info("   ✅ Utility Optimization")
    logger.info("   ✅ Plant Anomaly Detection")
    
    return platform
