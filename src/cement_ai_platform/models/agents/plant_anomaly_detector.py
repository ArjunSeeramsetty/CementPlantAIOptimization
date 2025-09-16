"""
Plant Anomaly Detection System for JK Cement Requirements
Real-time monitoring and anomaly detection for critical equipment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import yaml
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """Types of anomalies"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    FLOW = "flow"
    POWER = "power"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"

@dataclass
class AnomalyThreshold:
    """Anomaly detection thresholds"""
    parameter: str
    normal_min: float
    normal_max: float
    warning_min: float
    warning_max: float
    critical_min: float
    critical_max: float
    unit: str

@dataclass
class EquipmentConfig:
    """Equipment configuration for anomaly detection"""
    equipment_id: str
    equipment_name: str
    equipment_type: str
    critical_tags: List[str]
    thresholds: Dict[str, AnomalyThreshold]
    sampling_rate_seconds: int = 60
    history_window_hours: int = 24

class StatisticalAnomalyDetector:
    """Statistical anomaly detection using Z-score and moving averages"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_history = {}
        self.statistics = {}
        
    def detect_statistical_anomalies(self, 
                                   sensor_data: Dict[str, float],
                                   equipment_id: str) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in sensor data"""
        
        anomalies = []
        
        for tag, value in sensor_data.items():
            if not isinstance(value, (int, float)):
                continue
                
            # Initialize history if needed
            if tag not in self.data_history:
                self.data_history[tag] = deque(maxlen=self.window_size)
                self.statistics[tag] = {'mean': 0, 'std': 0}
            
            # Add new data point
            self.data_history[tag].append(value)
            
            # Update statistics if we have enough data
            if len(self.data_history[tag]) >= 10:
                self._update_statistics(tag)
                
                # Detect anomalies
                anomaly_score = self._calculate_anomaly_score(tag, value)
                
                if abs(anomaly_score) > 2.0:  # Z-score > 2
                    anomalies.append({
                        'equipment_id': equipment_id,
                        'tag': tag,
                        'value': value,
                        'anomaly_score': anomaly_score,
                        'severity': self._determine_severity(anomaly_score),
                        'type': self._determine_anomaly_type(tag),
                        'timestamp': datetime.now().isoformat(),
                        'detection_method': 'statistical'
                    })
        
        return anomalies
    
    def _update_statistics(self, tag: str):
        """Update statistical parameters for a tag"""
        
        if len(self.data_history[tag]) >= 10:
            values = list(self.data_history[tag])
            self.statistics[tag]['mean'] = statistics.mean(values)
            self.statistics[tag]['std'] = statistics.stdev(values) if len(values) > 1 else 0
    
    def _calculate_anomaly_score(self, tag: str, value: float) -> float:
        """Calculate Z-score anomaly score"""
        
        if tag not in self.statistics or self.statistics[tag]['std'] == 0:
            return 0.0
        
        mean = self.statistics[tag]['mean']
        std = self.statistics[tag]['std']
        
        return (value - mean) / std
    
    def _determine_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Determine anomaly severity based on score"""
        
        abs_score = abs(anomaly_score)
        
        if abs_score >= 4.0:
            return AnomalySeverity.CRITICAL
        elif abs_score >= 3.0:
            return AnomalySeverity.HIGH
        elif abs_score >= 2.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _determine_anomaly_type(self, tag: str) -> AnomalyType:
        """Determine anomaly type based on tag name"""
        
        tag_lower = tag.lower()
        
        if 'temp' in tag_lower or 'temperature' in tag_lower:
            return AnomalyType.TEMPERATURE
        elif 'pressure' in tag_lower or 'press' in tag_lower:
            return AnomalyType.PRESSURE
        elif 'vibration' in tag_lower or 'vib' in tag_lower:
            return AnomalyType.VIBRATION
        elif 'flow' in tag_lower:
            return AnomalyType.FLOW
        elif 'power' in tag_lower or 'kw' in tag_lower:
            return AnomalyType.POWER
        elif 'quality' in tag_lower or 'strength' in tag_lower:
            return AnomalyType.QUALITY
        else:
            return AnomalyType.EFFICIENCY

class ThresholdAnomalyDetector:
    """Threshold-based anomaly detection"""
    
    def __init__(self, equipment_configs: Dict[str, EquipmentConfig]):
        self.equipment_configs = equipment_configs
        
    def detect_threshold_anomalies(self, 
                                 sensor_data: Dict[str, float],
                                 equipment_id: str) -> List[Dict[str, Any]]:
        """Detect anomalies based on predefined thresholds"""
        
        anomalies = []
        
        if equipment_id not in self.equipment_configs:
            return anomalies
        
        config = self.equipment_configs[equipment_id]
        
        for tag, value in sensor_data.items():
            if tag not in config.thresholds:
                continue
                
            if not isinstance(value, (int, float)):
                continue
            
            threshold = config.thresholds[tag]
            anomaly = self._check_threshold_anomaly(tag, value, threshold, equipment_id)
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _check_threshold_anomaly(self, 
                               tag: str,
                               value: float,
                               threshold: AnomalyThreshold,
                               equipment_id: str) -> Optional[Dict[str, Any]]:
        """Check if value exceeds threshold limits"""
        
        severity = None
        message = ""
        
        # Check critical limits
        if value < threshold.critical_min or value > threshold.critical_max:
            severity = AnomalySeverity.CRITICAL
            if value < threshold.critical_min:
                message = f"{tag} critically low: {value:.2f} {threshold.unit} (min: {threshold.critical_min:.2f})"
            else:
                message = f"{tag} critically high: {value:.2f} {threshold.unit} (max: {threshold.critical_max:.2f})"
        
        # Check warning limits
        elif value < threshold.warning_min or value > threshold.warning_max:
            severity = AnomalySeverity.HIGH
            if value < threshold.warning_min:
                message = f"{tag} high warning: {value:.2f} {threshold.unit} (min: {threshold.warning_min:.2f})"
            else:
                message = f"{tag} high warning: {value:.2f} {threshold.unit} (max: {threshold.warning_max:.2f})"
        
        # Check normal limits
        elif value < threshold.normal_min or value > threshold.normal_max:
            severity = AnomalySeverity.MEDIUM
            if value < threshold.normal_min:
                message = f"{tag} outside normal range: {value:.2f} {threshold.unit} (min: {threshold.normal_min:.2f})"
            else:
                message = f"{tag} outside normal range: {value:.2f} {threshold.unit} (max: {threshold.normal_max:.2f})"
        
        if severity:
            return {
                'equipment_id': equipment_id,
                'tag': tag,
                'value': value,
                'threshold_min': threshold.normal_min,
                'threshold_max': threshold.normal_max,
                'severity': severity,
                'type': self._determine_anomaly_type(tag),
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'detection_method': 'threshold'
            }
        
        return None
    
    def _determine_anomaly_type(self, tag: str) -> AnomalyType:
        """Determine anomaly type based on tag name"""
        
        tag_lower = tag.lower()
        
        if 'temp' in tag_lower or 'temperature' in tag_lower:
            return AnomalyType.TEMPERATURE
        elif 'pressure' in tag_lower or 'press' in tag_lower:
            return AnomalyType.PRESSURE
        elif 'vibration' in tag_lower or 'vib' in tag_lower:
            return AnomalyType.VIBRATION
        elif 'flow' in tag_lower:
            return AnomalyType.FLOW
        elif 'power' in tag_lower or 'kw' in tag_lower:
            return AnomalyType.POWER
        elif 'quality' in tag_lower or 'strength' in tag_lower:
            return AnomalyType.QUALITY
        else:
            return AnomalyType.EFFICIENCY

class TrendAnomalyDetector:
    """Trend-based anomaly detection"""
    
    def __init__(self, trend_window: int = 20):
        self.trend_window = trend_window
        self.trend_history = {}
        
    def detect_trend_anomalies(self, 
                              sensor_data: Dict[str, float],
                              equipment_id: str) -> List[Dict[str, Any]]:
        """Detect trend-based anomalies"""
        
        anomalies = []
        
        for tag, value in sensor_data.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Initialize trend history
            if tag not in self.trend_history:
                self.trend_history[tag] = deque(maxlen=self.trend_window)
            
            # Add new data point
            self.trend_history[tag].append(value)
            
            # Detect trends if we have enough data
            if len(self.trend_history[tag]) >= 10:
                trend_anomaly = self._analyze_trend(tag, value, equipment_id)
                if trend_anomaly:
                    anomalies.append(trend_anomaly)
        
        return anomalies
    
    def _analyze_trend(self, tag: str, value: float, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Analyze trend patterns for anomalies"""
        
        values = list(self.trend_history[tag])
        
        # Calculate trend slope
        if len(values) >= 5:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Detect rapid changes
            if abs(slope) > self._get_trend_threshold(tag):
                severity = self._determine_trend_severity(slope, tag)
                
                return {
                    'equipment_id': equipment_id,
                    'tag': tag,
                    'value': value,
                    'trend_slope': slope,
                    'severity': severity,
                    'type': self._determine_anomaly_type(tag),
                    'message': f"{tag} showing rapid {'increase' if slope > 0 else 'decrease'}: {slope:.3f} per sample",
                    'timestamp': datetime.now().isoformat(),
                    'detection_method': 'trend'
                }
        
        return None
    
    def _get_trend_threshold(self, tag: str) -> float:
        """Get trend threshold for specific tag"""
        
        tag_lower = tag.lower()
        
        # Different thresholds for different parameter types
        if 'temp' in tag_lower:
            return 2.0  # °C per sample
        elif 'pressure' in tag_lower:
            return 0.5  # bar per sample
        elif 'vibration' in tag_lower:
            return 0.2  # mm/s per sample
        elif 'flow' in tag_lower:
            return 5.0  # tph per sample
        else:
            return 1.0  # Default threshold
    
    def _determine_trend_severity(self, slope: float, tag: str) -> AnomalySeverity:
        """Determine severity based on trend slope"""
        
        abs_slope = abs(slope)
        threshold = self._get_trend_threshold(tag)
        
        if abs_slope > threshold * 3:
            return AnomalySeverity.CRITICAL
        elif abs_slope > threshold * 2:
            return AnomalySeverity.HIGH
        elif abs_slope > threshold * 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _determine_anomaly_type(self, tag: str) -> AnomalyType:
        """Determine anomaly type based on tag name"""
        
        tag_lower = tag.lower()
        
        if 'temp' in tag_lower or 'temperature' in tag_lower:
            return AnomalyType.TEMPERATURE
        elif 'pressure' in tag_lower or 'press' in tag_lower:
            return AnomalyType.PRESSURE
        elif 'vibration' in tag_lower or 'vib' in tag_lower:
            return AnomalyType.VIBRATION
        elif 'flow' in tag_lower:
            return AnomalyType.FLOW
        elif 'power' in tag_lower or 'kw' in tag_lower:
            return AnomalyType.POWER
        elif 'quality' in tag_lower or 'strength' in tag_lower:
            return AnomalyType.QUALITY
        else:
            return AnomalyType.EFFICIENCY

class PlantAnomalyDetector:
    """
    Main plant anomaly detection system.
    Implements JK Cement's requirement for real-time anomaly warning system.
    """

    def __init__(self, config_path: str = "config/plant_config.yml"):
        self.config_path = config_path
        self.equipment_configs = self._load_equipment_configs()
        
        # Initialize detectors
        self.statistical_detector = StatisticalAnomalyDetector()
        self.threshold_detector = ThresholdAnomalyDetector(self.equipment_configs)
        self.trend_detector = TrendAnomalyDetector()
        
        # Anomaly history
        self.anomaly_history = []
        self.active_alerts = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            AnomalySeverity.CRITICAL: 0,  # Immediate alert
            AnomalySeverity.HIGH: 1,     # Alert after 1 occurrence
            AnomalySeverity.MEDIUM: 3,    # Alert after 3 occurrences
            AnomalySeverity.LOW: 5        # Alert after 5 occurrences
        }
        
        logger.info("✅ Plant Anomaly Detector initialized")

    def _load_equipment_configs(self) -> Dict[str, EquipmentConfig]:
        """Load equipment configurations for anomaly detection"""
        
        # Default equipment configurations
        equipment_configs = {}
        
        # Kiln configuration
        kiln_thresholds = {
            'burning_zone_temp_c': AnomalyThreshold(
                parameter='burning_zone_temp_c',
                normal_min=1400, normal_max=1500,
                warning_min=1350, warning_max=1550,
                critical_min=1300, critical_max=1600,
                unit='°C'
            ),
            'kiln_speed_rpm': AnomalyThreshold(
                parameter='kiln_speed_rpm',
                normal_min=2.5, normal_max=3.5,
                warning_min=2.0, warning_max=4.0,
                critical_min=1.5, critical_max=4.5,
                unit='rpm'
            ),
            'fuel_rate_tph': AnomalyThreshold(
                parameter='fuel_rate_tph',
                normal_min=12, normal_max=18,
                warning_min=10, warning_max=20,
                critical_min=8, critical_max=25,
                unit='tph'
            )
        }
        
        equipment_configs['kiln_01'] = EquipmentConfig(
            equipment_id='kiln_01',
            equipment_name='Main Kiln',
            equipment_type='kiln',
            critical_tags=['burning_zone_temp_c', 'kiln_speed_rpm', 'fuel_rate_tph'],
            thresholds=kiln_thresholds
        )
        
        # Raw Mill configuration
        raw_mill_thresholds = {
            'mill_vibration_mm_s': AnomalyThreshold(
                parameter='mill_vibration_mm_s',
                normal_min=2.0, normal_max=6.0,
                warning_min=1.5, warning_max=8.0,
                critical_min=1.0, critical_max=10.0,
                unit='mm/s'
            ),
            'mill_power_kw': AnomalyThreshold(
                parameter='mill_power_kw',
                normal_min=2000, normal_max=3000,
                warning_min=1800, warning_max=3200,
                critical_min=1500, critical_max=3500,
                unit='kW'
            ),
            'mill_outlet_temp_c': AnomalyThreshold(
                parameter='mill_outlet_temp_c',
                normal_min=80, normal_max=120,
                warning_min=70, warning_max=130,
                critical_min=60, critical_max=150,
                unit='°C'
            )
        }
        
        equipment_configs['raw_mill_01'] = EquipmentConfig(
            equipment_id='raw_mill_01',
            equipment_name='Raw Mill',
            equipment_type='mill',
            critical_tags=['mill_vibration_mm_s', 'mill_power_kw', 'mill_outlet_temp_c'],
            thresholds=raw_mill_thresholds
        )
        
        # Cement Mill configuration
        cement_mill_thresholds = {
            'mill_vibration_mm_s': AnomalyThreshold(
                parameter='mill_vibration_mm_s',
                normal_min=1.5, normal_max=5.0,
                warning_min=1.0, warning_max=7.0,
                critical_min=0.5, critical_max=9.0,
                unit='mm/s'
            ),
            'mill_power_kw': AnomalyThreshold(
                parameter='mill_power_kw',
                normal_min=1500, normal_max=2500,
                warning_min=1300, warning_max=2700,
                critical_min=1100, critical_max=3000,
                unit='kW'
            ),
            'fineness_blaine_cm2_g': AnomalyThreshold(
                parameter='fineness_blaine_cm2_g',
                normal_min=3200, normal_max=3800,
                warning_min=3000, warning_max=4000,
                critical_min=2800, critical_max=4200,
                unit='cm²/g'
            )
        }
        
        equipment_configs['cement_mill_01'] = EquipmentConfig(
            equipment_id='cement_mill_01',
            equipment_name='Cement Mill',
            equipment_type='mill',
            critical_tags=['mill_vibration_mm_s', 'mill_power_kw', 'fineness_blaine_cm2_g'],
            thresholds=cement_mill_thresholds
        )
        
        # ID Fan configuration
        id_fan_thresholds = {
            'fan_vibration_mm_s': AnomalyThreshold(
                parameter='fan_vibration_mm_s',
                normal_min=3.0, normal_max=7.0,
                warning_min=2.0, warning_max=9.0,
                critical_min=1.0, critical_max=12.0,
                unit='mm/s'
            ),
            'fan_power_kw': AnomalyThreshold(
                parameter='fan_power_kw',
                normal_min=800, normal_max=1200,
                warning_min=700, warning_max=1300,
                critical_min=600, critical_max=1500,
                unit='kW'
            ),
            'fan_pressure_pa': AnomalyThreshold(
                parameter='fan_pressure_pa',
                normal_min=-200, normal_max=-100,
                warning_min=-250, warning_max=-80,
                critical_min=-300, critical_max=-50,
                unit='Pa'
            )
        }
        
        equipment_configs['id_fan_01'] = EquipmentConfig(
            equipment_id='id_fan_01',
            equipment_name='ID Fan',
            equipment_type='fan',
            critical_tags=['fan_vibration_mm_s', 'fan_power_kw', 'fan_pressure_pa'],
            thresholds=id_fan_thresholds
        )
        
        return equipment_configs

    def detect_anomalies(self, 
                        sensor_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Detect anomalies across all equipment
        
        Args:
            sensor_data: Dictionary with equipment_id as key and sensor readings as value
            
        Returns:
            Comprehensive anomaly detection results
        """
        logger.info("🔄 Starting comprehensive anomaly detection...")
        
        all_anomalies = []
        equipment_status = {}
        
        for equipment_id, readings in sensor_data.items():
            # Detect anomalies using all methods
            statistical_anomalies = self.statistical_detector.detect_statistical_anomalies(
                readings, equipment_id
            )
            
            threshold_anomalies = self.threshold_detector.detect_threshold_anomalies(
                readings, equipment_id
            )
            
            trend_anomalies = self.trend_detector.detect_trend_anomalies(
                readings, equipment_id
            )
            
            # Combine all anomalies
            equipment_anomalies = statistical_anomalies + threshold_anomalies + trend_anomalies
            all_anomalies.extend(equipment_anomalies)
            
            # Determine equipment status
            equipment_status[equipment_id] = self._determine_equipment_status(
                equipment_anomalies, equipment_id
            )
        
        # Process alerts
        alerts = self._process_alerts(all_anomalies)
        
        # Update anomaly history
        self._update_anomaly_history(all_anomalies)
        
        # Generate summary
        summary = self._generate_anomaly_summary(all_anomalies, equipment_status)
        
        logger.info(f"✅ Anomaly detection completed: {len(all_anomalies)} anomalies found")
        
        return {
            'anomalies': all_anomalies,
            'equipment_status': equipment_status,
            'alerts': alerts,
            'summary': summary,
            'detection_timestamp': datetime.now().isoformat()
        }

    def _determine_equipment_status(self, 
                                  anomalies: List[Dict[str, Any]],
                                  equipment_id: str) -> Dict[str, Any]:
        """Determine equipment status based on anomalies"""
        
        if not anomalies:
            return {
                'status': 'Normal',
                'severity': 'None',
                'anomaly_count': 0,
                'last_anomaly': None
            }
        
        # Find highest severity anomaly
        severities = [anomaly['severity'] for anomaly in anomalies]
        highest_severity = max(severities, key=lambda s: s.value)
        
        # Determine overall status
        if highest_severity == AnomalySeverity.CRITICAL:
            status = 'Critical'
        elif highest_severity == AnomalySeverity.HIGH:
            status = 'Warning'
        elif highest_severity == AnomalySeverity.MEDIUM:
            status = 'Attention'
        else:
            status = 'Normal'
        
        return {
            'status': status,
            'severity': highest_severity.value,
            'anomaly_count': len(anomalies),
            'last_anomaly': anomalies[-1]['timestamp'] if anomalies else None,
            'critical_anomalies': [a for a in anomalies if a['severity'] == AnomalySeverity.CRITICAL]
        }

    def _process_alerts(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process anomalies and generate alerts"""
        
        alerts = []
        
        for anomaly in anomalies:
            equipment_id = anomaly['equipment_id']
            severity = anomaly['severity']
            
            # Check if alert should be generated
            if self._should_generate_alert(equipment_id, severity):
                alert = {
                    'alert_id': f"{equipment_id}_{severity.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'equipment_id': equipment_id,
                    'equipment_name': self.equipment_configs.get(equipment_id, {}).get('equipment_name', equipment_id),
                    'severity': severity.value,
                    'anomaly': anomaly,
                    'recommended_actions': self._get_recommended_actions(anomaly),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'Active'
                }
                
                alerts.append(alert)
                self.active_alerts[alert['alert_id']] = alert
        
        return alerts

    def _should_generate_alert(self, equipment_id: str, severity: AnomalySeverity) -> bool:
        """Determine if alert should be generated based on severity and history"""
        
        threshold = self.alert_thresholds[severity]
        
        # Count recent anomalies of this severity for this equipment
        recent_anomalies = [
            a for a in self.anomaly_history[-100:]  # Last 100 anomalies
            if (a['equipment_id'] == equipment_id and 
                a['severity'] == severity and
                datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1))
        ]
        
        return len(recent_anomalies) >= threshold

    def _get_recommended_actions(self, anomaly: Dict[str, Any]) -> List[str]:
        """Get recommended actions for an anomaly"""
        
        equipment_type = self.equipment_configs.get(anomaly['equipment_id'], {}).get('equipment_type', 'unknown')
        anomaly_type = anomaly['type'].value
        severity = anomaly['severity'].value
        
        actions = []
        
        # General actions based on severity
        if severity == 'critical':
            actions.append("Immediate shutdown and inspection required")
            actions.append("Notify maintenance team immediately")
        elif severity == 'high':
            actions.append("Schedule maintenance within 24 hours")
            actions.append("Increase monitoring frequency")
        elif severity == 'medium':
            actions.append("Schedule maintenance within 1 week")
            actions.append("Continue monitoring")
        
        # Specific actions based on equipment and anomaly type
        if equipment_type == 'kiln' and anomaly_type == 'temperature':
            actions.append("Check fuel flow and air supply")
            actions.append("Inspect refractory condition")
        elif equipment_type == 'mill' and anomaly_type == 'vibration':
            actions.append("Check grinding media condition")
            actions.append("Inspect mill bearings")
        elif equipment_type == 'fan' and anomaly_type == 'pressure':
            actions.append("Check fan blades and casing")
            actions.append("Inspect ductwork for blockages")
        
        return actions

    def _update_anomaly_history(self, anomalies: List[Dict[str, Any]]):
        """Update anomaly history"""
        
        self.anomaly_history.extend(anomalies)
        
        # Keep only last 1000 anomalies
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]

    def _generate_anomaly_summary(self, 
                                anomalies: List[Dict[str, Any]],
                                equipment_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate anomaly detection summary"""
        
        # Count anomalies by severity
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = len([
                a for a in anomalies if a['severity'] == severity
            ])
        
        # Count equipment by status
        status_counts = {}
        for status_info in equipment_status.values():
            status = status_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate overall plant health
        total_equipment = len(equipment_status)
        normal_equipment = status_counts.get('Normal', 0)
        plant_health = (normal_equipment / total_equipment) * 100 if total_equipment > 0 else 100
        
        return {
            'total_anomalies': len(anomalies),
            'severity_distribution': severity_counts,
            'equipment_status_distribution': status_counts,
            'plant_health_percentage': plant_health,
            'critical_equipment': [
                eq_id for eq_id, status in equipment_status.items() 
                if status['status'] in ['Critical', 'Warning']
            ],
            'active_alerts_count': len(self.active_alerts)
        }

    def get_equipment_health_report(self) -> Dict[str, Any]:
        """Get comprehensive equipment health report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'equipment_summary': {},
            'anomaly_trends': self._analyze_anomaly_trends(),
            'maintenance_recommendations': self._generate_maintenance_recommendations()
        }
        
        # Generate equipment summary
        for equipment_id, config in self.equipment_configs.items():
            recent_anomalies = [
                a for a in self.anomaly_history[-100:]
                if a['equipment_id'] == equipment_id
            ]
            
            report['equipment_summary'][equipment_id] = {
                'equipment_name': config.equipment_name,
                'equipment_type': config.equipment_type,
                'recent_anomalies_count': len(recent_anomalies),
                'last_anomaly': recent_anomalies[-1]['timestamp'] if recent_anomalies else None,
                'health_status': 'Good' if len(recent_anomalies) < 5 else 'Needs Attention'
            }
        
        return report

    def _analyze_anomaly_trends(self) -> Dict[str, Any]:
        """Analyze anomaly trends over time"""
        
        if len(self.anomaly_history) < 10:
            return {'status': 'Insufficient data'}
        
        # Analyze last 24 hours
        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        # Count by hour
        hourly_counts = {}
        for anomaly in recent_anomalies:
            hour = datetime.fromisoformat(anomaly['timestamp']).hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        return {
            'anomalies_last_24h': len(recent_anomalies),
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
            'trend': 'Increasing' if len(recent_anomalies) > 20 else 'Stable'
        }

    def _generate_maintenance_recommendations(self) -> List[Dict[str, str]]:
        """Generate maintenance recommendations based on anomaly patterns"""
        
        recommendations = []
        
        # Analyze equipment with most anomalies
        equipment_anomaly_counts = {}
        for anomaly in self.anomaly_history[-200:]:  # Last 200 anomalies
            eq_id = anomaly['equipment_id']
            equipment_anomaly_counts[eq_id] = equipment_anomaly_counts.get(eq_id, 0) + 1
        
        # Get top 3 equipment with most anomalies
        top_equipment = sorted(equipment_anomaly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for equipment_id, count in top_equipment:
            if count > 10:  # Threshold for maintenance recommendation
                equipment_name = self.equipment_configs.get(equipment_id, {}).get('equipment_name', equipment_id)
                recommendations.append({
                    'equipment': equipment_name,
                    'priority': 'High' if count > 20 else 'Medium',
                    'recommendation': f'Schedule preventive maintenance - {count} anomalies in recent history',
                    'expected_benefit': 'Reduce anomaly frequency and improve reliability'
                })
        
        return recommendations

    def export_anomaly_data(self) -> Dict[str, Any]:
        """Export anomaly detection data"""
        
        return {
            'anomaly_history': self.anomaly_history[-100:],  # Last 100 anomalies
            'active_alerts': list(self.active_alerts.values()),
            'equipment_configs': {
                eq_id: {
                    'equipment_name': config.equipment_name,
                    'equipment_type': config.equipment_type,
                    'critical_tags': config.critical_tags
                }
                for eq_id, config in self.equipment_configs.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
