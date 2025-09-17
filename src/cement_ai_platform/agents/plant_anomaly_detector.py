import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnomalyThresholds:
    """Anomaly detection thresholds for different equipment"""
    vibration_high: float = 8.0  # mm/s
    vibration_critical: float = 12.0  # mm/s
    temperature_deviation: float = 50.0  # °C from normal
    pressure_deviation: float = 100.0  # Pa from normal
    current_deviation: float = 20.0  # % from normal
    efficiency_low: float = 0.80  # 80% of normal efficiency

@dataclass
class EquipmentNormals:
    """Normal operating ranges for different equipment"""
    kiln_speed: Tuple[float, float] = (2.8, 4.2)  # rpm
    burning_zone_temp: Tuple[float, float] = (1420, 1480)  # °C
    feed_rate: Tuple[float, float] = (150, 200)  # t/h
    fuel_rate: Tuple[float, float] = (14, 20)  # t/h
    cooler_outlet_temp: Tuple[float, float] = (80, 120)  # °C
    raw_mill_power: Tuple[float, float] = (1800, 2400)  # kW
    cement_mill_power: Tuple[float, float] = (3000, 4000)  # kW

class EquipmentHealthMonitor:
    """Monitor health of critical equipment using multiple indicators"""
    
    def __init__(self):
        self.thresholds = AnomalyThresholds()
        self.normals = EquipmentNormals()
        self.health_history = {}
        
    def analyze_equipment_health(self, equipment_data: Dict) -> Dict:
        """Comprehensive equipment health analysis"""
        
        equipment_name = equipment_data.get('equipment_name', 'Unknown')
        
        # Multi-parameter health assessment
        health_indicators = {
            'vibration_health': self._assess_vibration_health(equipment_data),
            'thermal_health': self._assess_thermal_health(equipment_data),
            'electrical_health': self._assess_electrical_health(equipment_data),
            'performance_health': self._assess_performance_health(equipment_data)
        }
        
        # Overall health score (0-1 scale)
        health_weights = {'vibration_health': 0.3, 'thermal_health': 0.25, 
                         'electrical_health': 0.25, 'performance_health': 0.2}
        
        overall_health = sum(health_indicators[indicator] * health_weights[indicator]
                           for indicator in health_indicators)
        
        # Failure prediction
        failure_prediction = self._predict_failure_risk(equipment_data, health_indicators)
        
        # Maintenance recommendations
        maintenance_recommendations = self._generate_maintenance_recommendations(
            equipment_name, health_indicators, failure_prediction
        )
        
        # Store in history
        self.health_history[equipment_name] = {
            'timestamp': pd.Timestamp.now(),
            'health_score': overall_health,
            'indicators': health_indicators
        }
        
        return {
            'equipment_name': equipment_name,
            'overall_health_score': overall_health,
            'health_status': self._categorize_health_status(overall_health),
            'health_indicators': health_indicators,
            'failure_prediction': failure_prediction,
            'maintenance_recommendations': maintenance_recommendations
        }
    
    def _assess_vibration_health(self, equipment_data: Dict) -> float:
        """Assess equipment health based on vibration measurements"""
        
        vibration_x = equipment_data.get('vibration_x_mm_s', 3.0)
        vibration_y = equipment_data.get('vibration_y_mm_s', 3.0)
        vibration_z = equipment_data.get('vibration_z_mm_s', 2.5)
        
        # Overall vibration level
        overall_vibration = np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2)
        
        # Health assessment based on ISO 10816 standards
        if overall_vibration <= 2.8:
            health_score = 1.0  # Good
        elif overall_vibration <= 7.1:
            health_score = 0.8  # Satisfactory
        elif overall_vibration <= 18.0:
            health_score = 0.4  # Unsatisfactory
        else:
            health_score = 0.1  # Unacceptable
        
        return health_score
    
    def _assess_thermal_health(self, equipment_data: Dict) -> float:
        """Assess equipment health based on temperature measurements"""
        
        # Get temperature measurements
        bearing_temp = equipment_data.get('bearing_temperature_c', 70)
        motor_temp = equipment_data.get('motor_temperature_c', 80)
        ambient_temp = equipment_data.get('ambient_temperature_c', 35)
        
        # Normal temperature rises above ambient
        normal_bearing_rise = 40  # °C
        normal_motor_rise = 50    # °C
        
        # Calculate temperature rises
        bearing_rise = bearing_temp - ambient_temp
        motor_rise = motor_temp - ambient_temp
        
        # Health assessment
        bearing_health = max(0, 1 - max(0, bearing_rise - normal_bearing_rise) / normal_bearing_rise)
        motor_health = max(0, 1 - max(0, motor_rise - normal_motor_rise) / normal_motor_rise)
        
        # Combined thermal health
        thermal_health = (bearing_health + motor_health) / 2
        
        return thermal_health
    
    def _assess_electrical_health(self, equipment_data: Dict) -> float:
        """Assess equipment health based on electrical parameters"""
        
        # Current measurements
        current_a = equipment_data.get('current_phase_a', 100)
        current_b = equipment_data.get('current_phase_b', 100) 
        current_c = equipment_data.get('current_phase_c', 100)
        
        # Voltage measurements  
        voltage_a = equipment_data.get('voltage_phase_a', 400)
        voltage_b = equipment_data.get('voltage_phase_b', 400)
        voltage_c = equipment_data.get('voltage_phase_c', 400)
        
        # Power factor
        power_factor = equipment_data.get('power_factor', 0.85)
        
        # Current imbalance assessment
        avg_current = (current_a + current_b + current_c) / 3
        current_imbalance = max(abs(current_a - avg_current), 
                               abs(current_b - avg_current),
                               abs(current_c - avg_current)) / avg_current * 100
        
        # Voltage imbalance assessment
        avg_voltage = (voltage_a + voltage_b + voltage_c) / 3
        voltage_imbalance = max(abs(voltage_a - avg_voltage),
                               abs(voltage_b - avg_voltage), 
                               abs(voltage_c - avg_voltage)) / avg_voltage * 100
        
        # Health scoring
        current_health = max(0, 1 - current_imbalance / 10)  # 10% imbalance = 0 health
        voltage_health = max(0, 1 - voltage_imbalance / 5)   # 5% imbalance = 0 health  
        pf_health = max(0, (power_factor - 0.7) / 0.25)      # 0.7-0.95 range
        
        electrical_health = (current_health + voltage_health + pf_health) / 3
        
        return electrical_health
    
    def _assess_performance_health(self, equipment_data: Dict) -> float:
        """Assess equipment health based on performance indicators"""
        
        equipment_type = equipment_data.get('equipment_type', 'general')
        
        if equipment_type == 'mill':
            return self._assess_mill_performance(equipment_data)
        elif equipment_type == 'kiln':
            return self._assess_kiln_performance(equipment_data)
        elif equipment_type == 'fan':
            return self._assess_fan_performance(equipment_data)
        else:
            return self._assess_general_performance(equipment_data)
    
    def _assess_mill_performance(self, equipment_data: Dict) -> float:
        """Assess mill-specific performance health"""
        
        # Power consumption efficiency
        power_consumption = equipment_data.get('power_kw', 2000)
        throughput = equipment_data.get('throughput_tph', 80)
        specific_power = power_consumption / throughput if throughput > 0 else float('inf')
        
        # Typical specific power for raw mills: 15-25 kWh/t
        target_specific_power = 20.0
        power_efficiency = target_specific_power / specific_power if specific_power > 0 else 0
        power_efficiency = min(1.0, power_efficiency)
        
        # Product fineness consistency
        fineness_variation = equipment_data.get('fineness_variation_percent', 5)
        fineness_health = max(0, 1 - fineness_variation / 10)
        
        return (power_efficiency + fineness_health) / 2
    
    def _assess_kiln_performance(self, equipment_data: Dict) -> float:
        """Assess kiln-specific performance health"""
        
        # Thermal efficiency
        fuel_rate = equipment_data.get('fuel_rate_tph', 16)
        production_rate = equipment_data.get('production_rate_tph', 167)
        specific_energy = fuel_rate * 6500 / production_rate if production_rate > 0 else float('inf')
        
        # Target: 690 kcal/kg
        target_energy = 690
        energy_efficiency = target_energy / specific_energy if specific_energy > 0 else 0
        energy_efficiency = min(1.0, energy_efficiency)
        
        # Quality consistency (free lime)
        free_lime = equipment_data.get('free_lime_percent', 1.0)
        free_lime_health = max(0, 1 - abs(free_lime - 1.0) / 2.0)
        
        return (energy_efficiency + free_lime_health) / 2
    
    def _assess_fan_performance(self, equipment_data: Dict) -> float:
        """Assess fan-specific performance health"""
        
        # Flow efficiency
        actual_flow = equipment_data.get('flow_nm3_h', 50000)
        design_flow = equipment_data.get('design_flow_nm3_h', 55000)
        flow_efficiency = actual_flow / design_flow if design_flow > 0 else 0
        
        # Pressure efficiency  
        actual_pressure = equipment_data.get('pressure_pa', -150)
        design_pressure = equipment_data.get('design_pressure_pa', -160)
        pressure_efficiency = abs(actual_pressure) / abs(design_pressure) if design_pressure != 0 else 0
        
        return (min(1.0, flow_efficiency) + min(1.0, pressure_efficiency)) / 2
    
    def _assess_general_performance(self, equipment_data: Dict) -> float:
        """Assess general equipment performance"""
        
        # Availability
        availability = equipment_data.get('availability_percent', 95) / 100
        
        # Power consumption stability
        power_variation = equipment_data.get('power_variation_percent', 5)
        power_stability = max(0, 1 - power_variation / 20)
        
        return (availability + power_stability) / 2
    
    def _predict_failure_risk(self, equipment_data: Dict, health_indicators: Dict) -> Dict:
        """Predict equipment failure risk and timeline"""
        
        overall_health = sum(health_indicators.values()) / len(health_indicators)
        
        # Simple rule-based prediction (could be enhanced with ML models)
        if overall_health > 0.8:
            risk_level = 'Low'
            predicted_failure_days = None
            maintenance_urgency = 'Scheduled'
        elif overall_health > 0.6:
            risk_level = 'Medium'
            predicted_failure_days = 90
            maintenance_urgency = 'Plan Soon'
        elif overall_health > 0.4:
            risk_level = 'High'
            predicted_failure_days = 30
            maintenance_urgency = 'Schedule Immediately'
        else:
            risk_level = 'Critical'
            predicted_failure_days = 7
            maintenance_urgency = 'Emergency'
        
        # Trend analysis if historical data available
        equipment_name = equipment_data.get('equipment_name', 'Unknown')
        trend = self._analyze_health_trend(equipment_name, overall_health)
        
        return {
            'risk_level': risk_level,
            'predicted_failure_days': predicted_failure_days,
            'maintenance_urgency': maintenance_urgency,
            'health_trend': trend,
            'confidence': 0.7  # Could be improved with more sophisticated models
        }
    
    def _analyze_health_trend(self, equipment_name: str, current_health: float) -> str:
        """Analyze health trend based on historical data"""
        
        if equipment_name not in self.health_history:
            return 'No History'
        
        historical_health = self.health_history[equipment_name]['health_score']
        
        if current_health > historical_health + 0.05:
            return 'Improving'
        elif current_health < historical_health - 0.05:
            return 'Deteriorating'
        else:
            return 'Stable'
    
    def _generate_maintenance_recommendations(self, equipment_name: str, 
                                            health_indicators: Dict, 
                                            failure_prediction: Dict) -> List[Dict]:
        """Generate specific maintenance recommendations"""
        
        recommendations = []
        
        # Vibration-based recommendations
        if health_indicators['vibration_health'] < 0.6:
            recommendations.append({
                'category': 'Vibration',
                'recommendation': 'Check bearing condition and alignment',
                'urgency': failure_prediction['maintenance_urgency'],
                'estimated_cost': 5000
            })
        
        # Thermal-based recommendations
        if health_indicators['thermal_health'] < 0.6:
            recommendations.append({
                'category': 'Thermal',
                'recommendation': 'Inspect cooling systems and lubrication',
                'urgency': failure_prediction['maintenance_urgency'],
                'estimated_cost': 3000
            })
        
        # Electrical-based recommendations
        if health_indicators['electrical_health'] < 0.6:
            recommendations.append({
                'category': 'Electrical',
                'recommendation': 'Check electrical connections and motor condition',
                'urgency': failure_prediction['maintenance_urgency'],
                'estimated_cost': 8000
            })
        
        # Performance-based recommendations
        if health_indicators['performance_health'] < 0.6:
            recommendations.append({
                'category': 'Performance',
                'recommendation': 'Review operating parameters and optimize settings',
                'urgency': failure_prediction['maintenance_urgency'],
                'estimated_cost': 2000
            })
        
        return recommendations
    
    def _categorize_health_status(self, health_score: float) -> str:
        """Categorize overall health status"""
        
        if health_score >= 0.8:
            return 'Excellent'
        elif health_score >= 0.6:
            return 'Good'  
        elif health_score >= 0.4:
            return 'Fair'
        elif health_score >= 0.2:
            return 'Poor'
        else:
            return 'Critical'

class ProcessAnomalyDetector:
    """Detect process anomalies using statistical and ML methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.is_fitted = False
        self.process_limits = self._initialize_process_limits()
    
    def _initialize_process_limits(self) -> Dict:
        """Initialize process parameter limits"""
        return {
            'kiln_speed_rpm': (2.8, 4.2),
            'feed_rate_tph': (150, 200),
            'fuel_rate_tph': (14, 20),
            'burning_zone_temp_c': (1420, 1480),
            'free_lime_percent': (0.5, 2.0),
            'o2_percent': (2.0, 4.5),
            'nox_mg_nm3': (300, 800),
            'co_mg_nm3': (50, 250)
        }
    
    def fit(self, training_data: pd.DataFrame):
        """Train anomaly detection models on historical data"""
        
        # Select relevant columns
        feature_columns = [col for col in training_data.columns 
                          if col in self.process_limits.keys()]
        
        if not feature_columns:
            raise ValueError("No relevant process parameters found in training data")
        
        X = training_data[feature_columns].dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        
        # Store feature columns
        self.feature_columns = feature_columns
        self.is_fitted = True
        
        print(f"Anomaly detector trained on {len(X)} samples with {len(feature_columns)} features")
    
    def detect_anomalies(self, current_data: Dict) -> Dict:
        """Detect anomalies in current process data"""
        
        # Rule-based anomaly detection
        rule_based_anomalies = self._detect_rule_based_anomalies(current_data)
        
        # Statistical anomaly detection (if model is fitted)
        statistical_anomalies = {}
        if self.is_fitted:
            statistical_anomalies = self._detect_statistical_anomalies(current_data)
        
        # Combine results
        all_anomalies = {**rule_based_anomalies, **statistical_anomalies}
        
        # Calculate overall anomaly score
        anomaly_score = len([a for a in all_anomalies.values() if a['is_anomaly']]) / len(all_anomalies) if all_anomalies else 0
        
        return {
            'overall_anomaly_score': anomaly_score,
            'anomaly_status': self._categorize_anomaly_status(anomaly_score),
            'detected_anomalies': all_anomalies,
            'summary': self._generate_anomaly_summary(all_anomalies)
        }
    
    def _detect_rule_based_anomalies(self, current_data: Dict) -> Dict:
        """Detect anomalies using predefined rules"""
        
        anomalies = {}
        
        for parameter, (min_val, max_val) in self.process_limits.items():
            if parameter in current_data:
                value = current_data[parameter]
                is_anomaly = value < min_val or value > max_val
                
                if is_anomaly:
                    severity = 'High' if (value < min_val * 0.8 or value > max_val * 1.2) else 'Medium'
                else:
                    severity = 'Low'
                
                anomalies[parameter] = {
                    'is_anomaly': is_anomaly,
                    'value': value,
                    'normal_range': (min_val, max_val),
                    'severity': severity,
                    'deviation_percent': self._calculate_deviation_percent(value, min_val, max_val),
                    'detection_method': 'Rule-based'
                }
        
        return anomalies
    
    def _detect_statistical_anomalies(self, current_data: Dict) -> Dict:
        """Detect anomalies using statistical ML models"""
        
        if not self.is_fitted:
            return {}
        
        # Prepare data for prediction
        feature_data = {col: current_data.get(col, 0) for col in self.feature_columns}
        X = pd.DataFrame([feature_data])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict using isolation forest
        anomaly_prediction = self.isolation_forest.predict(X_scaled)[0]
        anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
        
        # Determine anomalous features
        statistical_anomalies = {}
        
        if anomaly_prediction == -1:  # Anomaly detected
            # Calculate individual feature contributions to anomaly
            for i, feature in enumerate(self.feature_columns):
                feature_score = abs(X_scaled[0, i])  # Simplified contribution
                
                statistical_anomalies[f"{feature}_statistical"] = {
                    'is_anomaly': feature_score > 2.0,  # 2 standard deviations
                    'value': feature_data[feature],
                    'anomaly_score': float(feature_score),
                    'severity': 'High' if feature_score > 3.0 else 'Medium' if feature_score > 2.0 else 'Low',
                    'detection_method': 'Statistical'
                }
        
        return statistical_anomalies
    
    def _calculate_deviation_percent(self, value: float, min_val: float, max_val: float) -> float:
        """Calculate percentage deviation from normal range"""
        
        if min_val <= value <= max_val:
            return 0.0
        elif value < min_val:
            return (min_val - value) / min_val * 100
        else:
            return (value - max_val) / max_val * 100
    
    def _categorize_anomaly_status(self, anomaly_score: float) -> str:
        """Categorize overall anomaly status"""
        
        if anomaly_score < 0.1:
            return 'Normal'
        elif anomaly_score < 0.3:
            return 'Minor Anomalies'
        elif anomaly_score < 0.5:
            return 'Moderate Anomalies'
        else:
            return 'Severe Anomalies'
    
    def _generate_anomaly_summary(self, anomalies: Dict) -> Dict:
        """Generate summary of detected anomalies"""
        
        total_anomalies = len([a for a in anomalies.values() if a['is_anomaly']])
        high_severity = len([a for a in anomalies.values() if a.get('severity') == 'High'])
        medium_severity = len([a for a in anomalies.values() if a.get('severity') == 'Medium'])
        
        # Most critical anomaly
        critical_anomaly = None
        max_deviation = 0
        
        for param, anomaly in anomalies.items():
            if anomaly['is_anomaly']:
                deviation = anomaly.get('deviation_percent', 0)
                if deviation > max_deviation:
                    max_deviation = deviation
                    critical_anomaly = param
        
        return {
            'total_anomalies': total_anomalies,
            'high_severity_count': high_severity,
            'medium_severity_count': medium_severity,
            'most_critical_parameter': critical_anomaly,
            'max_deviation_percent': max_deviation
        }

class PlantAnomalyDetector:
    """
    Main plant anomaly detection system combining equipment and process monitoring
    """
    
    def __init__(self):
        self.equipment_monitor = EquipmentHealthMonitor()
        self.process_monitor = ProcessAnomalyDetector()
        self.alert_history = []
    
    def monitor_plant_status(self, plant_data: Dict) -> Dict:
        """Comprehensive plant status monitoring"""
        
        # Equipment health monitoring
        equipment_results = {}
        equipment_list = plant_data.get('equipment_list', [
            {'equipment_name': 'Raw_Mill_01', 'equipment_type': 'mill'},
            {'equipment_name': 'Kiln_01', 'equipment_type': 'kiln'},
            {'equipment_name': 'Cement_Mill_01', 'equipment_type': 'mill'},
            {'equipment_name': 'ID_Fan_01', 'equipment_type': 'fan'}
        ])
        
        for equipment in equipment_list:
            equipment_name = equipment['equipment_name']
            equipment_data = plant_data.get('equipment_data', {}).get(equipment_name, {})
            equipment_data.update(equipment)  # Add equipment type info
            
            equipment_results[equipment_name] = self.equipment_monitor.analyze_equipment_health(equipment_data)
        
        # Process anomaly detection
        process_data = plant_data.get('process_data', {})
        process_results = self.process_monitor.detect_anomalies(process_data)
        
        # Overall plant health assessment
        plant_health = self._assess_overall_plant_health(equipment_results, process_results)
        
        # Generate alerts
        alerts = self._generate_alerts(equipment_results, process_results, plant_health)
        
        # Store alert history
        self.alert_history.extend(alerts)
        
        return {
            'plant_health_score': plant_health['overall_score'],
            'plant_status': plant_health['status'],
            'equipment_health': equipment_results,
            'process_anomalies': process_results,
            'active_alerts': alerts,
            'recommendations': self._generate_plant_recommendations(equipment_results, process_results),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _assess_overall_plant_health(self, equipment_results: Dict, process_results: Dict) -> Dict:
        """Assess overall plant health"""
        
        # Equipment health component
        equipment_scores = [result['overall_health_score'] for result in equipment_results.values()]
        avg_equipment_health = sum(equipment_scores) / len(equipment_scores) if equipment_scores else 1.0
        
        # Process anomaly component  
        process_anomaly_score = process_results.get('overall_anomaly_score', 0)
        process_health_score = 1.0 - process_anomaly_score
        
        # Weighted overall score
        overall_score = avg_equipment_health * 0.7 + process_health_score * 0.3
        
        # Categorize status
        if overall_score >= 0.8:
            status = 'Excellent'
        elif overall_score >= 0.6:
            status = 'Good'
        elif overall_score >= 0.4:
            status = 'Fair'
        else:
            status = 'Poor'
        
        return {
            'overall_score': overall_score,
            'status': status,
            'equipment_contribution': avg_equipment_health,
            'process_contribution': process_health_score
        }
    
    def _generate_alerts(self, equipment_results: Dict, process_results: Dict, 
                        plant_health: Dict) -> List[Dict]:
        """Generate alerts based on monitoring results"""
        
        alerts = []
        
        # Equipment-based alerts
        for equipment_name, result in equipment_results.items():
            if result['overall_health_score'] < 0.4:
                alerts.append({
                    'type': 'Equipment Health',
                    'equipment': equipment_name,
                    'severity': 'High' if result['overall_health_score'] < 0.2 else 'Medium',
                    'message': f"{equipment_name} health score is {result['overall_health_score']:.2f}",
                    'recommended_action': result['failure_prediction']['maintenance_urgency'],
                    'timestamp': pd.Timestamp.now()
                })
        
        # Process-based alerts
        for param, anomaly in process_results.get('detected_anomalies', {}).items():
            if anomaly['is_anomaly'] and anomaly['severity'] in ['High', 'Medium']:
                alerts.append({
                    'type': 'Process Anomaly',
                    'parameter': param,
                    'severity': anomaly['severity'],
                    'message': f"{param} is {anomaly['value']:.2f}, outside normal range {anomaly.get('normal_range', 'N/A')}",
                    'recommended_action': 'Investigate process conditions',
                    'timestamp': pd.Timestamp.now()
                })
        
        # Plant-level alerts
        if plant_health['overall_score'] < 0.3:
            alerts.append({
                'type': 'Plant Health',
                'severity': 'Critical',
                'message': f"Overall plant health is poor ({plant_health['overall_score']:.2f})",
                'recommended_action': 'Immediate management attention required',
                'timestamp': pd.Timestamp.now()
            })
        
        return alerts
    
    def _generate_plant_recommendations(self, equipment_results: Dict, 
                                      process_results: Dict) -> List[Dict]:
        """Generate plant-wide recommendations"""
        
        recommendations = []
        
        # Equipment recommendations
        for equipment_name, result in equipment_results.items():
            for rec in result['maintenance_recommendations']:
                recommendations.append({
                    'category': 'Equipment Maintenance',
                    'equipment': equipment_name,
                    'recommendation': rec['recommendation'],
                    'urgency': rec['urgency'],
                    'estimated_cost': rec['estimated_cost']
                })
        
        # Process optimization recommendations
        critical_anomalies = process_results.get('detected_anomalies', {})
        for param, anomaly in critical_anomalies.items():
            if anomaly['is_anomaly'] and anomaly['severity'] == 'High':
                recommendations.append({
                    'category': 'Process Optimization',
                    'parameter': param,
                    'recommendation': f"Adjust {param} to normal range {anomaly.get('normal_range', 'N/A')}",
                    'urgency': 'Immediate',
                    'estimated_cost': 1000  # Operational adjustment cost
                })
        
        # Sort by urgency
        urgency_order = {'Emergency': 0, 'Critical': 1, 'Immediate': 2, 'Plan Soon': 3, 'Scheduled': 4}
        recommendations.sort(key=lambda x: urgency_order.get(x['urgency'], 5))
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def train_process_models(self, historical_data: pd.DataFrame):
        """Train process anomaly detection models"""
        self.process_monitor.fit(historical_data)
    
    def get_alert_summary(self, days: int = 7) -> Dict:
        """Get summary of alerts over specified period"""
        
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['timestamp'] > cutoff_date]
        
        # Count by type and severity
        alert_counts = {'Equipment Health': 0, 'Process Anomaly': 0, 'Plant Health': 0}
        severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for alert in recent_alerts:
            alert_counts[alert['type']] = alert_counts.get(alert['type'], 0) + 1
            severity_counts[alert['severity']] = severity_counts.get(alert['severity'], 0) + 1
        
        return {
            'period_days': days,
            'total_alerts': len(recent_alerts),
            'alert_by_type': alert_counts,
            'alert_by_severity': severity_counts,
            'avg_alerts_per_day': len(recent_alerts) / days
        }
