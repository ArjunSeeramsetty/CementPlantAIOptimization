import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import ks_2samp, chi2_contingency
from google.cloud import bigquery
from google.cloud import monitoring_v3
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class DataDriftDetector:
    """
    Statistical drift detection and data validation system
    with Google Cloud integration for cement plant data
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        
        try:
            self.monitoring_client = monitoring_v3.MetricServiceClient()
        except Exception as e:
            print(f"⚠️ Monitoring client warning: {e}")
            self.monitoring_client = None
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'ks_test_pvalue': 0.05,      # KS test p-value threshold
            'mean_shift_threshold': 0.15, # 15% mean shift threshold
            'std_shift_threshold': 0.20,  # 20% std deviation shift threshold
            'outlier_rate_threshold': 0.10 # 10% outlier rate threshold
        }
        
        # Process variable categories for validation
        self.process_variables = {
            'quality': ['free_lime_percent', 'compressive_strength_28d_mpa', 'blaine_fineness_cm2_g'],
            'energy': ['thermal_energy_kcal_kg', 'electrical_energy_kwh_t', 'coal_consumption_kg_t'],
            'process': ['feed_rate_tph', 'fuel_rate_tph', 'kiln_speed_rpm', 'burning_zone_temp_c'],
            'emissions': ['nox_mg_nm3', 'so2_mg_nm3', 'dust_mg_nm3', 'co2_kg_per_ton']
        }
        
        # Expected ranges for process variables (for anomaly detection)
        self.expected_ranges = {
            'free_lime_percent': (0.5, 2.5),
            'thermal_energy_kcal_kg': (600, 800),
            'feed_rate_tph': (140, 190),
            'fuel_rate_tph': (12, 22),
            'burning_zone_temp_c': (1400, 1500),
            'nox_mg_nm3': (300, 700)
        }
        
        print("✅ Data drift detection system initialized")
    
    def create_reference_snapshot(self, data: pd.DataFrame, snapshot_name: str = "baseline") -> bool:
        """Create reference snapshot for drift detection"""
        
        try:
            # Calculate statistical summary for each variable
            reference_stats = {}
            
            for category, variables in self.process_variables.items():
                category_stats = {}
                
                for var in variables:
                    if var in data.columns:
                        var_data = data[var].dropna()
                        
                        if len(var_data) > 0:
                            category_stats[var] = {
                                'mean': float(var_data.mean()),
                                'std': float(var_data.std()),
                                'median': float(var_data.median()),
                                'q25': float(var_data.quantile(0.25)),
                                'q75': float(var_data.quantile(0.75)),
                                'min': float(var_data.min()),
                                'max': float(var_data.max()),
                                'count': int(len(var_data)),
                                'snapshot_date': datetime.now().isoformat()
                            }
                
                reference_stats[category] = category_stats
            
            # Store reference snapshot (in production, this would go to BigQuery)
            if not hasattr(self, 'reference_snapshots'):
                self.reference_snapshots = {}
            
            self.reference_snapshots[snapshot_name] = reference_stats
            
            print(f"✅ Created reference snapshot: {snapshot_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating reference snapshot: {e}")
            return False
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         reference_snapshot: str = "baseline") -> Dict:
        """Detect data drift between current data and reference snapshot"""
        
        if not hasattr(self, 'reference_snapshots') or reference_snapshot not in self.reference_snapshots:
            return {
                'drift_detected': False,
                'error': f"Reference snapshot '{reference_snapshot}' not found"
            }
        
        reference_stats = self.reference_snapshots[reference_snapshot]
        drift_results = {}
        overall_drift_detected = False
        
        # Analyze each category
        for category, variables in self.process_variables.items():
            if category not in reference_stats:
                continue
                
            category_results = {}
            category_drift = False
            
            for var in variables:
                if var in current_data.columns and var in reference_stats[category]:
                    
                    # Get current and reference data
                    current_var_data = current_data[var].dropna()
                    ref_stats = reference_stats[category][var]
                    
                    if len(current_var_data) > 10:  # Minimum sample size
                        
                        # Calculate drift metrics
                        drift_metrics = self._calculate_drift_metrics(current_var_data, ref_stats)
                        
                        # Determine if drift detected
                        var_drift_detected = self._evaluate_drift_significance(drift_metrics)
                        
                        category_results[var] = {
                            **drift_metrics,
                            'drift_detected': var_drift_detected,
                            'severity': self._assess_drift_severity(drift_metrics)
                        }
                        
                        if var_drift_detected:
                            category_drift = True
                            overall_drift_detected = True
            
            drift_results[category] = {
                'variables': category_results,
                'category_drift_detected': category_drift
            }
        
        # Generate drift summary
        drift_summary = self._generate_drift_summary(drift_results)
        
        # Send monitoring alerts if drift detected
        if overall_drift_detected and self.monitoring_client:
            self._send_drift_alert(drift_summary)
        
        return {
            'drift_detected': overall_drift_detected,
            'drift_results': drift_results,
            'drift_summary': drift_summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'reference_snapshot': reference_snapshot
        }
    
    def _calculate_drift_metrics(self, current_data: pd.Series, ref_stats: Dict) -> Dict:
        """Calculate drift metrics between current data and reference statistics"""
        
        # Current data statistics
        current_mean = current_data.mean()
        current_std = current_data.std()
        current_median = current_data.median()
        
        # Reference statistics
        ref_mean = ref_stats['mean']
        ref_std = ref_stats['std']
        ref_median = ref_stats['median']
        
        # Calculate drift metrics
        metrics = {
            'current_mean': current_mean,
            'reference_mean': ref_mean,
            'current_std': current_std,
            'reference_std': ref_std,
            'current_median': current_median,
            'reference_median': ref_median,
            'sample_size': len(current_data)
        }
        
        # Mean shift (relative)
        mean_shift = abs(current_mean - ref_mean) / (ref_std + 1e-8)
        metrics['mean_shift'] = mean_shift
        
        # Standard deviation shift (relative)
        std_shift = abs(current_std - ref_std) / (ref_std + 1e-8)
        metrics['std_shift'] = std_shift
        
        # Generate synthetic reference data for KS test (in production, use actual reference data)
        ref_synthetic = np.random.normal(ref_mean, ref_std, len(current_data))
        ks_statistic, ks_pvalue = ks_2samp(current_data, ref_synthetic)
        
        metrics['ks_statistic'] = ks_statistic
        metrics['ks_pvalue'] = ks_pvalue
        
        # Outlier detection
        outlier_threshold = 3 * ref_std
        outliers = np.abs(current_data - ref_mean) > outlier_threshold
        outlier_rate = outliers.sum() / len(current_data)
        metrics['outlier_rate'] = outlier_rate
        
        return metrics
    
    def _evaluate_drift_significance(self, metrics: Dict) -> bool:
        """Evaluate if drift is statistically significant"""
        
        # Check multiple drift indicators
        drift_indicators = []
        
        # KS test significance
        if metrics['ks_pvalue'] < self.drift_thresholds['ks_test_pvalue']:
            drift_indicators.append('ks_test')
        
        # Mean shift significance
        if metrics['mean_shift'] > self.drift_thresholds['mean_shift_threshold']:
            drift_indicators.append('mean_shift')
        
        # Standard deviation shift significance
        if metrics['std_shift'] > self.drift_thresholds['std_shift_threshold']:
            drift_indicators.append('std_shift')
        
        # Outlier rate significance
        if metrics['outlier_rate'] > self.drift_thresholds['outlier_rate_threshold']:
            drift_indicators.append('outlier_rate')
        
        # Drift detected if any significant indicator
        return len(drift_indicators) > 0
    
    def _assess_drift_severity(self, metrics: Dict) -> str:
        """Assess the severity of detected drift"""
        
        severity_score = 0
        
        # KS test contribution
        if metrics['ks_pvalue'] < 0.001:
            severity_score += 3
        elif metrics['ks_pvalue'] < 0.01:
            severity_score += 2
        elif metrics['ks_pvalue'] < 0.05:
            severity_score += 1
        
        # Mean shift contribution
        if metrics['mean_shift'] > 0.5:
            severity_score += 3
        elif metrics['mean_shift'] > 0.3:
            severity_score += 2
        elif metrics['mean_shift'] > 0.15:
            severity_score += 1
        
        # Outlier rate contribution
        if metrics['outlier_rate'] > 0.2:
            severity_score += 2
        elif metrics['outlier_rate'] > 0.1:
            severity_score += 1
        
        # Classify severity
        if severity_score >= 5:
            return "Critical"
        elif severity_score >= 3:
            return "High"
        elif severity_score >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_drift_summary(self, drift_results: Dict) -> Dict:
        """Generate summary of drift detection results"""
        
        summary = {
            'total_variables_analyzed': 0,
            'variables_with_drift': 0,
            'categories_with_drift': 0,
            'max_severity': "Low",
            'critical_variables': [],
            'recommendations': []
        }
        
        severity_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        max_severity_score = 0
        
        for category, results in drift_results.items():
            category_has_drift = results['category_drift_detected']
            
            if category_has_drift:
                summary['categories_with_drift'] += 1
            
            for var, var_results in results['variables'].items():
                summary['total_variables_analyzed'] += 1
                
                if var_results['drift_detected']:
                    summary['variables_with_drift'] += 1
                    
                    # Check severity
                    severity = var_results['severity']
                    severity_score = severity_order[severity]
                    
                    if severity_score > max_severity_score:
                        max_severity_score = severity_score
                        summary['max_severity'] = severity
                    
                    if severity == "Critical":
                        summary['critical_variables'].append(var)
        
        # Generate recommendations
        if summary['variables_with_drift'] > 0:
            summary['recommendations'].extend([
                "Investigate data collection processes",
                "Check sensor calibration and maintenance",
                "Validate data preprocessing pipelines"
            ])
            
            if summary['max_severity'] in ["High", "Critical"]:
                summary['recommendations'].extend([
                    "Consider model retraining",
                    "Review process control parameters",
                    "Implement additional quality checks"
                ])
        
        return summary
    
    def _send_drift_alert(self, drift_summary: Dict):
        """Send drift detection alert to Cloud Monitoring"""
        
        try:
            # Create custom metric for drift detection
            project_name = f"projects/{self.project_id}"
            
            # Send drift detection metric
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/cement_plant/data_drift_score"
            series.resource.type = "generic_task"
            series.resource.labels["project_id"] = self.project_id
            series.resource.labels["location"] = "global"
            series.resource.labels["namespace"] = "data_validation"
            series.resource.labels["task_id"] = "drift_detection"
            
            # Calculate drift score
            drift_score = drift_summary['variables_with_drift'] / max(drift_summary['total_variables_analyzed'], 1)
            
            # Create data point
            import time
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": seconds, "nanos": nanos}
            })
            
            point = monitoring_v3.Point({
                "interval": interval, 
                "value": {"double_value": drift_score}
            })
            
            series.points = [point]
            
            # Send to monitoring
            self.monitoring_client.create_time_series(
                name=project_name, 
                time_series=[series]
            )
            
            print(f"✅ Drift alert sent to Cloud Monitoring (score: {drift_score:.3f})")
            
        except Exception as e:
            print(f"❌ Error sending drift alert: {e}")
    
    def trigger_model_retraining(self, drift_summary: Dict) -> Dict:
        """Trigger model retraining pipeline based on drift detection"""
        
        # Simulate retraining pipeline trigger
        if drift_summary['max_severity'] in ["High", "Critical"]:
            
            retraining_config = {
                'trigger_reason': 'data_drift_detected',
                'drift_severity': drift_summary['max_severity'],
                'affected_variables': drift_summary.get('critical_variables', []),
                'retraining_priority': 'high' if drift_summary['max_severity'] == "Critical" else 'medium',
                'estimated_duration': '2-4 hours',
                'pipeline_id': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            print(f"🔄 Triggering model retraining pipeline: {retraining_config['pipeline_id']}")
            
            return {
                'retraining_triggered': True,
                'retraining_config': retraining_config,
                'estimated_completion': datetime.now() + timedelta(hours=3)
            }
        
        else:
            return {
                'retraining_triggered': False,
                'reason': 'Drift severity below retraining threshold'
            }
