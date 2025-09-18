import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

@dataclass
class MaintenanceRecommendation:
    equipment_id: str
    equipment_name: str
    failure_probability: float
    predicted_failure_date: datetime
    time_to_failure_hours: float
    confidence: float
    maintenance_type: str
    priority: str
    estimated_cost: float
    impact_description: str
    recommended_actions: List[str]

class PredictiveMaintenanceEngine:
    """
    Advanced predictive maintenance system with time-to-failure models
    and CMMS integration for cement plant equipment
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        
        # Models for different equipment types
        self.failure_models = {}
        self.scalers = {}
        
        # Maintenance thresholds
        self.thresholds = {
            'critical': 0.8,    # 80% failure probability
            'high': 0.6,        # 60% failure probability  
            'medium': 0.4,      # 40% failure probability
            'low': 0.2          # 20% failure probability
        }
        
        # Equipment-specific parameters
        self.equipment_configs = {
            'kiln': {
                'sensors': ['temperature', 'vibration', 'current', 'torque'],
                'critical_components': ['drive_motor', 'refractory', 'tire_pads'],
                'maintenance_intervals': {'preventive': 168, 'major': 8760}  # hours
            },
            'raw_mill': {
                'sensors': ['vibration', 'power', 'temperature', 'oil_analysis'],
                'critical_components': ['grinding_media', 'liners', 'bearings'],
                'maintenance_intervals': {'preventive': 336, 'major': 4380}
            },
            'cement_mill': {
                'sensors': ['vibration', 'power', 'temperature', 'fineness'],
                'critical_components': ['grinding_media', 'liners', 'separator'],
                'maintenance_intervals': {'preventive': 336, 'major': 4380}
            },
            'id_fan': {
                'sensors': ['vibration', 'temperature', 'current', 'flow'],
                'critical_components': ['impeller', 'bearings', 'shaft'],
                'maintenance_intervals': {'preventive': 720, 'major': 8760}
            },
            'cooler': {
                'sensors': ['temperature', 'pressure', 'vibration'],
                'critical_components': ['grate_plates', 'drive_system', 'air_ducts'],
                'maintenance_intervals': {'preventive': 168, 'major': 2190}
            }
        }
        
        # Initialize maintenance system
        self._initialize_predictive_maintenance()
    
    def _initialize_predictive_maintenance(self):
        """Initialize predictive maintenance system"""
        
        # Load or train failure prediction models
        self._load_or_train_models()
        
        print("✅ Predictive maintenance system initialized")
    
    def _load_or_train_models(self):
        """Load existing models or train new ones"""
        
        for equipment_type in self.equipment_configs:
            try:
                # Train a new model with synthetic data for demo
                self._train_failure_model(equipment_type)
                
            except Exception as e:
                print(f"❌ Error training model for {equipment_type}: {e}")
    
    def _train_failure_model(self, equipment_type: str):
        """Train failure prediction model for specific equipment type"""
        
        # Generate synthetic training data for demo
        training_data = self._generate_synthetic_maintenance_data(equipment_type, n_samples=2000)
        
        # Prepare features and target
        feature_cols = ['operating_hours', 'maintenance_age', 'load_factor']
        
        # Add equipment-specific sensor features
        config = self.equipment_configs[equipment_type]
        for sensor in config['sensors']:
            feature_cols.append(f'{equipment_type}_{sensor}')
        
        X = training_data[feature_cols].fillna(0)
        y_failure_prob = training_data['failure_probability']
        y_ttf = training_data['time_to_failure_hours']
        
        # Train failure probability model
        failure_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train time-to-failure model  
        ttf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        failure_model.fit(X_scaled, y_failure_prob)
        ttf_model.fit(X_scaled, y_ttf)
        
        # Store models
        self.failure_models[equipment_type] = {
            'failure_probability': failure_model,
            'time_to_failure': ttf_model,
            'feature_names': feature_cols
        }
        self.scalers[equipment_type] = scaler
        
        # Calculate model performance
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_failure_prob, test_size=0.2)
        test_score = failure_model.score(X_test, y_test)
        
        print(f"✅ Trained {equipment_type} failure model - R²: {test_score:.3f}")
    
    def _generate_synthetic_maintenance_data(self, equipment_type: str, n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic maintenance data for training"""
        
        np.random.seed(42)
        
        # Base data
        data = {
            'equipment_id': [f"{equipment_type}_{i//100 + 1:03d}" for i in range(n_samples)],
            'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='1H'),
            'operating_hours': np.random.exponential(8760, n_samples),
            'maintenance_age': np.random.uniform(0, 8760, n_samples),
            'load_factor': np.random.beta(8, 2, n_samples)
        }
        
        # Equipment-specific sensors
        config = self.equipment_configs[equipment_type]
        
        for sensor in config['sensors']:
            if sensor == 'vibration':
                data[f'{equipment_type}_vibration'] = np.random.lognormal(1.5, 0.5, n_samples)
            elif sensor == 'temperature':
                data[f'{equipment_type}_temperature'] = np.random.normal(75, 15, n_samples)
            elif sensor == 'current':
                data[f'{equipment_type}_current'] = np.random.normal(100, 20, n_samples)
            elif sensor == 'power':
                data[f'{equipment_type}_power'] = np.random.normal(2000, 400, n_samples)
            elif sensor == 'oil_analysis':
                data[f'{equipment_type}_oil_analysis'] = np.random.exponential(0.1, n_samples)
            else:
                data[f'{equipment_type}_{sensor}'] = np.random.normal(50, 10, n_samples)
        
        # Create failure indicators based on sensor conditions
        df = pd.DataFrame(data)
        
        # Calculate failure probability based on multiple factors
        failure_prob = np.zeros(n_samples)
        
        # Age factor
        age_factor = np.clip(df['operating_hours'] / 50000, 0, 1)
        failure_prob += age_factor * 0.3
        
        # Maintenance age factor
        maint_factor = np.clip(df['maintenance_age'] / config['maintenance_intervals']['major'], 0, 1)
        failure_prob += maint_factor * 0.4
        
        # Sensor-based factors
        if 'vibration' in config['sensors']:
            vib_factor = np.clip((df[f'{equipment_type}_vibration'] - 3) / 10, 0, 1)
            failure_prob += vib_factor * 0.2
        
        if 'temperature' in config['sensors']:
            temp_factor = np.clip((df[f'{equipment_type}_temperature'] - 80) / 50, 0, 1)
            failure_prob += temp_factor * 0.1
        
        # Add noise and clip
        failure_prob += np.random.normal(0, 0.05, n_samples)
        failure_prob = np.clip(failure_prob, 0, 1)
        
        # Calculate time to failure
        time_to_failure = np.where(
            failure_prob > 0.1,
            np.random.exponential(1000, n_samples) * (1 - failure_prob) + 24,
            np.random.uniform(8760, 17520, n_samples)
        )
        
        df['failure_probability'] = failure_prob
        df['time_to_failure_hours'] = time_to_failure
        
        return df
    
    def predict_equipment_failure(self, equipment_data: Dict) -> MaintenanceRecommendation:
        """Predict failure for specific equipment"""
        
        equipment_id = equipment_data['equipment_id']
        equipment_type = equipment_data.get('equipment_type', 'kiln')
        
        if equipment_type not in self.failure_models:
            print(f"❌ No model available for equipment type: {equipment_type}")
            return None
        
        # Prepare features
        features = self._prepare_features_for_prediction(equipment_data, equipment_type)
        
        # Scale features
        scaler = self.scalers[equipment_type]
        features_scaled = scaler.transform([features])
        
        # Predict failure probability and time to failure
        failure_prob = self.failure_models[equipment_type]['failure_probability'].predict(features_scaled)[0]
        ttf_hours = self.failure_models[equipment_type]['time_to_failure'].predict(features_scaled)[0]
        
        # Calculate confidence
        confidence = min(0.95, max(0.6, 1 - abs(failure_prob - 0.5)))
        
        # Determine priority and maintenance type
        priority, maintenance_type = self._determine_maintenance_priority(failure_prob, ttf_hours)
        
        # Estimate costs
        estimated_cost = self._estimate_maintenance_cost(equipment_type, maintenance_type, failure_prob)
        
        # Generate recommendations
        recommendations = self._generate_maintenance_recommendations(
            equipment_type, failure_prob, ttf_hours, maintenance_type
        )
        
        return MaintenanceRecommendation(
            equipment_id=equipment_id,
            equipment_name=equipment_data.get('equipment_name', equipment_id),
            failure_probability=failure_prob,
            predicted_failure_date=datetime.now() + timedelta(hours=ttf_hours),
            time_to_failure_hours=ttf_hours,
            confidence=confidence,
            maintenance_type=maintenance_type,
            priority=priority,
            estimated_cost=estimated_cost,
            impact_description=self._generate_impact_description(equipment_type, failure_prob),
            recommended_actions=recommendations
        )
    
    def _prepare_features_for_prediction(self, equipment_data: Dict, equipment_type: str) -> List[float]:
        """Prepare features for model prediction"""
        
        config = self.equipment_configs[equipment_type]
        features = []
        
        # Operating characteristics
        features.append(equipment_data.get('operating_hours', 8760))
        features.append(equipment_data.get('maintenance_age', 1000))
        features.append(equipment_data.get('load_factor', 0.85))
        
        # Sensor readings
        for sensor in config['sensors']:
            if sensor == 'vibration':
                features.append(equipment_data.get(f'{equipment_type}_vibration', 4.0))
            elif sensor == 'temperature':
                features.append(equipment_data.get(f'{equipment_type}_temperature', 75.0))
            elif sensor == 'current':
                features.append(equipment_data.get(f'{equipment_type}_current', 100.0))
            elif sensor == 'power':
                features.append(equipment_data.get(f'{equipment_type}_power', 2000.0))
            elif sensor == 'oil_analysis':
                features.append(equipment_data.get(f'{equipment_type}_oil_analysis', 0.05))
            else:
                features.append(equipment_data.get(f'{equipment_type}_{sensor}', 50.0))
        
        return features
    
    def _determine_maintenance_priority(self, failure_prob: float, ttf_hours: float) -> Tuple[str, str]:
        """Determine maintenance priority and type"""
        
        if failure_prob >= self.thresholds['critical'] or ttf_hours < 168:
            return "Critical", "Emergency"
        elif failure_prob >= self.thresholds['high'] or ttf_hours < 720:
            return "High", "Corrective"
        elif failure_prob >= self.thresholds['medium'] or ttf_hours < 2160:
            return "Medium", "Preventive"
        else:
            return "Low", "Routine"
    
    def _estimate_maintenance_cost(self, equipment_type: str, maintenance_type: str, failure_prob: float) -> float:
        """Estimate maintenance cost"""
        
        base_costs = {
            'kiln': {'Routine': 5000, 'Preventive': 15000, 'Corrective': 50000, 'Emergency': 200000},
            'raw_mill': {'Routine': 3000, 'Preventive': 10000, 'Corrective': 35000, 'Emergency': 150000},
            'cement_mill': {'Routine': 3000, 'Preventive': 12000, 'Corrective': 40000, 'Emergency': 180000},
            'id_fan': {'Routine': 2000, 'Preventive': 8000, 'Corrective': 25000, 'Emergency': 100000},
            'cooler': {'Routine': 4000, 'Preventive': 12000, 'Corrective': 30000, 'Emergency': 120000}
        }
        
        base_cost = base_costs.get(equipment_type, {}).get(maintenance_type, 10000)
        cost_multiplier = 1 + (failure_prob * 0.5)
        
        return base_cost * cost_multiplier
    
    def _generate_maintenance_recommendations(self, equipment_type: str, failure_prob: float, 
                                           ttf_hours: float, maintenance_type: str) -> List[str]:
        """Generate specific maintenance recommendations"""
        
        recommendations = []
        
        if equipment_type == 'kiln':
            if failure_prob > 0.6:
                recommendations.extend([
                    "Inspect kiln drive motor bearings and lubrication system",
                    "Check refractory condition and thermal imaging scan",
                    "Verify tire pad alignment and wear patterns"
                ])
            else:
                recommendations.extend([
                    "Schedule routine kiln inspection",
                    "Check kiln alignment and shell ovality"
                ])
        
        elif equipment_type == 'raw_mill':
            if failure_prob > 0.6:
                recommendations.extend([
                    "Inspect mill bearings and oil system",
                    "Check grinding media charge and liner wear",
                    "Verify separator performance and air flow"
                ])
            else:
                recommendations.extend([
                    "Monitor vibration trends",
                    "Schedule oil analysis"
                ])
        
        # Add time-based and priority-based recommendations
        if ttf_hours < 168:
            recommendations.append("Schedule immediate shutdown for inspection")
        elif ttf_hours < 720:
            recommendations.append("Plan maintenance during next scheduled shutdown")
        
        if maintenance_type == "Emergency":
            recommendations.append("Prepare emergency response team and spare parts")
        
        return recommendations[:5]
    
    def _generate_impact_description(self, equipment_type: str, failure_prob: float) -> str:
        """Generate impact description for potential failure"""
        
        impact_templates = {
            'kiln': {
                'high': "Critical impact - Complete production shutdown, potential refractory damage",
                'medium': "High impact - Reduced production capacity, quality issues",
                'low': "Medium impact - Minor efficiency loss, increased maintenance needs"
            },
            'raw_mill': {
                'high': "High impact - Raw material preparation disruption, production delay",
                'medium': "Medium impact - Reduced grinding efficiency, quality variation", 
                'low': "Low impact - Minor performance degradation"
            }
        }
        
        if failure_prob > 0.6:
            impact_level = 'high'
        elif failure_prob > 0.3:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        return impact_templates.get(equipment_type, {}).get(impact_level, "Standard maintenance impact")
    
    def generate_maintenance_report(self, plant_id: str, days_ahead: int = 30) -> Dict:
        """Generate comprehensive maintenance report"""
        
        # Simulate equipment list for plant
        equipment_list = [
            {'equipment_id': 'KILN_001', 'equipment_type': 'kiln', 'equipment_name': 'Kiln #1'},
            {'equipment_id': 'RMILL_001', 'equipment_type': 'raw_mill', 'equipment_name': 'Raw Mill #1'},
            {'equipment_id': 'CMILL_001', 'equipment_type': 'cement_mill', 'equipment_name': 'Cement Mill #1'},
            {'equipment_id': 'IDFAN_001', 'equipment_type': 'id_fan', 'equipment_name': 'ID Fan #1'},
            {'equipment_id': 'COOLER_001', 'equipment_type': 'cooler', 'equipment_name': 'Cooler #1'}
        ]
        
        recommendations = []
        total_cost = 0
        
        for equipment in equipment_list:
            # Simulate equipment data
            equipment_data = self._simulate_equipment_data(equipment)
            
            # Get maintenance recommendation
            recommendation = self.predict_equipment_failure(equipment_data)
            
            if recommendation and recommendation.time_to_failure_hours < (days_ahead * 24):
                recommendations.append(recommendation)
                total_cost += recommendation.estimated_cost
        
        # Sort by priority and time to failure
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: (priority_order[x.priority], x.time_to_failure_hours))
        
        # Generate summary statistics
        summary = {
            'total_recommendations': len(recommendations),
            'critical_count': len([r for r in recommendations if r.priority == 'Critical']),
            'high_count': len([r for r in recommendations if r.priority == 'High']),
            'medium_count': len([r for r in recommendations if r.priority == 'Medium']),
            'low_count': len([r for r in recommendations if r.priority == 'Low']),
            'total_estimated_cost': total_cost,
            'avg_failure_probability': sum(r.failure_probability for r in recommendations) / len(recommendations) if recommendations else 0,
            'report_generated_at': datetime.now().isoformat()
        }
        
        return {
            'plant_id': plant_id,
            'report_period_days': days_ahead,
            'summary': summary,
            'recommendations': recommendations
        }
    
    def _simulate_equipment_data(self, equipment_info: Dict) -> Dict:
        """Simulate equipment sensor data for demo"""
        
        equipment_type = equipment_info['equipment_type']
        
        equipment_data = {
            'equipment_id': equipment_info['equipment_id'],
            'equipment_type': equipment_type,
            'equipment_name': equipment_info['equipment_name'],
            'operating_hours': random.uniform(5000, 45000),
            'maintenance_age': random.uniform(100, 8000),
            'load_factor': random.uniform(0.7, 0.95)
        }
        
        # Add equipment-specific sensor data
        config = self.equipment_configs.get(equipment_type, {})
        
        for sensor in config.get('sensors', []):
            if sensor == 'vibration':
                equipment_data[f'{equipment_type}_vibration'] = random.uniform(2.0, 9.0)
            elif sensor == 'temperature':
                equipment_data[f'{equipment_type}_temperature'] = random.uniform(65, 95)
            elif sensor == 'current':
                equipment_data[f'{equipment_type}_current'] = random.uniform(80, 120)
            elif sensor == 'power':
                equipment_data[f'{equipment_type}_power'] = random.uniform(1800, 2200)
            elif sensor == 'oil_analysis':
                equipment_data[f'{equipment_type}_oil_analysis'] = random.uniform(0.01, 0.2)
            else:
                equipment_data[f'{equipment_type}_{sensor}'] = random.uniform(40, 60)
        
        return equipment_data
