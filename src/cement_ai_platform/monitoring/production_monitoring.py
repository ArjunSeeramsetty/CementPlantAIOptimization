"""
Production monitoring and observability for cement plant operations.
Enterprise-grade monitoring with custom metrics, alerting, and logging.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Production Google Cloud imports
try:
    from google.cloud import monitoring_v3
    from google.cloud import logging as cloud_logging
    from google.cloud.monitoring_v3 import query
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Cloud Monitoring not available. Using mock implementation.")

class ProductionMonitoring:
    """
    Enterprise monitoring for cement plant operations.
    Provides custom metrics, alerting, and comprehensive logging.
    """
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'cement-ai-opt-38517')
        
        if MONITORING_AVAILABLE:
            self._initialize_monitoring()
        else:
            self._initialize_mock()
    
    def _initialize_monitoring(self):
        """Initialize Cloud Monitoring with enterprise configuration"""
        try:
            # Initialize monitoring client
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            self.alert_client = monitoring_v3.AlertPolicyServiceClient()
            
            # Initialize logging client
            self.logging_client = cloud_logging.Client(project=self.project_id)
            self.logger = self.logging_client.logger("cement-plant-ai")
            
            # Project name for API calls
            self.project_name = f"projects/{self.project_id}"
            
            print(f"✅ Production monitoring initialized for project: {self.project_id}")
            
        except Exception as e:
            print(f"⚠️ Monitoring initialization failed: {e}")
            self._initialize_mock()
    
    def _initialize_mock(self):
        """Mock implementation for development/testing"""
        self.monitoring_client = None
        self.alert_client = None
        self.logging_client = None
        self.logger = None
        self.project_name = f"projects/{self.project_id}"
        print("🔄 Using mock monitoring implementation")
    
    def create_custom_metrics(self):
        """Create custom metrics for cement plant KPIs"""
        
        if not self.monitoring_client:
            return self._mock_metric_creation()
        
        try:
            metrics = [
                {
                    "type": "custom.googleapis.com/cement_plant/free_lime_deviation",
                    "labels": [
                        {"key": "plant_id", "value_type": "STRING"},
                        {"key": "severity", "value_type": "STRING"},
                        {"key": "shift", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "Free lime deviation from target specification"
                },
                {
                    "type": "custom.googleapis.com/cement_plant/energy_efficiency",
                    "labels": [
                        {"key": "plant_id", "value_type": "STRING"},
                        {"key": "energy_type", "value_type": "STRING"},
                        {"key": "unit", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "Energy efficiency percentage (thermal/electrical)"
                },
                {
                    "type": "custom.googleapis.com/cement_plant/equipment_health_score",
                    "labels": [
                        {"key": "equipment_name", "value_type": "STRING"},
                        {"key": "plant_id", "value_type": "STRING"},
                        {"key": "equipment_type", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "Equipment health score (0-1 scale)"
                },
                {
                    "type": "custom.googleapis.com/cement_plant/production_rate",
                    "labels": [
                        {"key": "plant_id", "value_type": "STRING"},
                        {"key": "product_type", "value_type": "STRING"},
                        {"key": "shift", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "Production rate in tons per hour"
                },
                {
                    "type": "custom.googleapis.com/cement_plant/ai_prediction_accuracy",
                    "labels": [
                        {"key": "model_name", "value_type": "STRING"},
                        {"key": "prediction_type", "value_type": "STRING"},
                        {"key": "plant_id", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "AI model prediction accuracy percentage"
                },
                {
                    "type": "custom.googleapis.com/cement_plant/tsr_achievement",
                    "labels": [
                        {"key": "plant_id", "value_type": "STRING"},
                        {"key": "fuel_type", "value_type": "STRING"},
                        {"key": "target_percent", "value_type": "STRING"}
                    ],
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "description": "Thermal Substitution Rate achievement percentage"
                }
            ]
            
            for metric_config in metrics:
                descriptor = monitoring_v3.MetricDescriptor()
                descriptor.type = metric_config["type"]
                descriptor.metric_kind = getattr(monitoring_v3.MetricDescriptor.MetricKind, metric_config["metric_kind"])
                descriptor.value_type = getattr(monitoring_v3.MetricDescriptor.ValueType, metric_config["value_type"])
                descriptor.description = metric_config["description"]
                
                # Add labels
                for label in metric_config["labels"]:
                    label_descriptor = descriptor.labels.add()
                    label_descriptor.key = label["key"]
                    label_descriptor.value_type = getattr(monitoring_v3.LabelDescriptor.ValueType, label["value_type"])
                
                try:
                    self.monitoring_client.create_metric_descriptor(
                        name=self.project_name,
                        metric_descriptor=descriptor
                    )
                    print(f"✅ Created metric: {metric_config['type']}")
                except Exception as e:
                    print(f"⚠️ Metric already exists or error: {e}")
            
            print("✅ Custom metrics creation completed")
            
        except Exception as e:
            print(f"❌ Custom metrics creation failed: {e}")
            return self._mock_metric_creation()
    
    def send_metric(self, metric_type: str, value: float, labels: Dict[str, str],
                   timestamp: Optional[datetime] = None):
        """
        Send custom metric to Cloud Monitoring.
        
        Args:
            metric_type: Type of metric to send
            value: Metric value
            labels: Metric labels
            timestamp: Optional timestamp (defaults to now)
        """
        
        if not self.monitoring_client:
            return self._mock_send_metric(metric_type, value, labels)
        
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.resource.type = "generic_task"
            series.resource.labels["project_id"] = self.project_id
            series.resource.labels["location"] = "global"
            series.resource.labels["namespace"] = "cement_plant"
            series.resource.labels["task_id"] = "digital_twin"
            
            # Add metric labels
            for key, val in labels.items():
                series.metric.labels[key] = val
            
            # Set timestamp
            if timestamp is None:
                timestamp = datetime.now()
            
            now = timestamp.timestamp()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            
            point = monitoring_v3.Point(
                {"interval": interval, "value": {"double_value": value}}
            )
            
            series.points = [point]
            
            # Send metric
            self.monitoring_client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
            
            print(f"✅ Sent metric: {metric_type} = {value}")
            
        except Exception as e:
            print(f"❌ Failed to send metric: {e}")
            self._mock_send_metric(metric_type, value, labels)
    
    def log_plant_event(self, event_type: str, data: Dict[str, Any], 
                       severity: str = "INFO", plant_id: str = "jk_cement_main"):
        """
        Log structured plant events for compliance and monitoring.
        
        Args:
            event_type: Type of event (e.g., 'quality_deviation', 'equipment_failure')
            data: Event data
            severity: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            plant_id: Plant identifier
        """
        
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "plant_id": plant_id,
            "data": data,
            "service": "cement-plant-ai",
            "environment": "production",
            "version": "1.0"
        }
        
        if self.logger:
            try:
                self.logger.log_struct(log_entry, severity=severity)
                print(f"✅ Logged event: {event_type} ({severity})")
            except Exception as e:
                print(f"❌ Failed to log event: {e}")
        else:
            print(f"🔄 Mock logging: {event_type} - {data}")
    
    def create_production_alerts(self):
        """Create critical production alert policies"""
        
        if not self.alert_client:
            return self._mock_alert_creation()
        
        try:
            # High free lime alert
            free_lime_alert = monitoring_v3.AlertPolicy(
                display_name="High Free Lime Alert - JK Cement",
                documentation=monitoring_v3.AlertPolicy.Documentation(
                    content="Alert when free lime exceeds 2.0% for more than 5 minutes",
                    mime_type="text/markdown"
                ),
                conditions=[
                    monitoring_v3.AlertPolicy.Condition(
                        display_name="Free lime > 2.0%",
                        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                            filter='metric.type="custom.googleapis.com/cement_plant/free_lime_deviation"',
                            comparison=monitoring_v3.ComparisonType.COMPARISON_GREATER_THAN,
                            threshold_value=2.0,
                            duration={"seconds": 300},  # 5 minutes
                            aggregations=[
                                monitoring_v3.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                                    cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_MEAN
                                )
                            ]
                        )
                    )
                ],
                notification_channels=self._get_notification_channels(),
                alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
                    auto_close={"seconds": 1800}  # Auto-close after 30 minutes
                ),
                labels={
                    "industry": "cement",
                    "severity": "high",
                    "plant": "jk_cement"
                }
            )
            
            # Equipment failure risk alert
            equipment_alert = monitoring_v3.AlertPolicy(
                display_name="Equipment Failure Risk - JK Cement",
                documentation=monitoring_v3.AlertPolicy.Documentation(
                    content="Alert when equipment health score drops below 0.4",
                    mime_type="text/markdown"
                ),
                conditions=[
                    monitoring_v3.AlertPolicy.Condition(
                        display_name="Equipment health < 0.4",
                        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                            filter='metric.type="custom.googleapis.com/cement_plant/equipment_health_score"',
                            comparison=monitoring_v3.ComparisonType.COMPARISON_LESS_THAN,
                            threshold_value=0.4,
                            duration={"seconds": 600},  # 10 minutes
                            aggregations=[
                                monitoring_v3.Aggregation(
                                    alignment_period={"seconds": 60},
                                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                                )
                            ]
                        )
                    )
                ],
                notification_channels=self._get_notification_channels(),
                labels={
                    "industry": "cement",
                    "severity": "critical",
                    "plant": "jk_cement"
                }
            )
            
            # Energy efficiency alert
            energy_alert = monitoring_v3.AlertPolicy(
                display_name="Energy Efficiency Degradation - JK Cement",
                documentation=monitoring_v3.AlertPolicy.Documentation(
                    content="Alert when energy efficiency drops below 85%",
                    mime_type="text/markdown"
                ),
                conditions=[
                    monitoring_v3.AlertPolicy.Condition(
                        display_name="Energy efficiency < 85%",
                        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                            filter='metric.type="custom.googleapis.com/cement_plant/energy_efficiency"',
                            comparison=monitoring_v3.ComparisonType.COMPARISON_LESS_THAN,
                            threshold_value=85.0,
                            duration={"seconds": 900},  # 15 minutes
                            aggregations=[
                                monitoring_v3.Aggregation(
                                    alignment_period={"seconds": 300},
                                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                                )
                            ]
                        )
                    )
                ],
                notification_channels=self._get_notification_channels(),
                labels={
                    "industry": "cement",
                    "severity": "medium",
                    "plant": "jk_cement"
                }
            )
            
            # Create alert policies
            policies = [free_lime_alert, equipment_alert, energy_alert]
            
            for policy in policies:
                try:
                    created_policy = self.alert_client.create_alert_policy(
                        name=self.project_name,
                        alert_policy=policy
                    )
                    print(f"✅ Created alert policy: {created_policy.display_name}")
                except Exception as e:
                    print(f"⚠️ Alert policy creation error: {e}")
            
            print("✅ Production alerts creation completed")
            
        except Exception as e:
            print(f"❌ Production alerts creation failed: {e}")
            return self._mock_alert_creation()
    
    def get_metric_data(self, metric_type: str, start_time: datetime, 
                       end_time: datetime, labels: Dict[str, str] = None) -> List[Dict]:
        """
        Retrieve metric data for analysis.
        
        Args:
            metric_type: Type of metric to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            labels: Optional label filters
            
        Returns:
            List of metric data points
        """
        
        if not self.monitoring_client:
            return self._mock_get_metric_data(metric_type)
        
        try:
            # Build filter
            filter_str = f'metric.type="{metric_type}"'
            if labels:
                for key, value in labels.items():
                    filter_str += f' AND metric.labels.{key}="{value}"'
            
            # Create time interval
            start_seconds = int(start_time.timestamp())
            end_seconds = int(end_time.timestamp())
            
            interval = monitoring_v3.TimeInterval({
                "start_time": {"seconds": start_seconds},
                "end_time": {"seconds": end_seconds}
            })
            
            # Query metric data
            results = self.monitoring_client.list_time_series(
                name=self.project_name,
                filter=filter_str,
                interval=interval,
                view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
            )
            
            data_points = []
            for result in results:
                for point in result.points:
                    data_points.append({
                        "timestamp": datetime.fromtimestamp(point.interval.end_time.seconds),
                        "value": point.value.double_value,
                        "labels": dict(result.metric.labels)
                    })
            
            return data_points
            
        except Exception as e:
            print(f"❌ Failed to retrieve metric data: {e}")
            return self._mock_get_metric_data(metric_type)
    
    def _get_notification_channels(self) -> List[str]:
        """Get notification channels for alerts"""
        # This would be configured based on JK Cement's notification preferences
        # For now, return empty list - would include email, SMS, Slack channels
        return []
    
    def _mock_metric_creation(self):
        """Mock metric creation for development"""
        print("🔄 Mock: Custom metrics created")
    
    def _mock_send_metric(self, metric_type: str, value: float, labels: Dict[str, str]):
        """Mock metric sending for development"""
        print(f"🔄 Mock: Sent metric {metric_type} = {value} with labels {labels}")
    
    def _mock_alert_creation(self):
        """Mock alert creation for development"""
        print("🔄 Mock: Production alerts created")
    
    def _mock_get_metric_data(self, metric_type: str) -> List[Dict]:
        """Mock metric data retrieval for development"""
        return [
            {
                "timestamp": datetime.now(),
                "value": 1.2,
                "labels": {"plant_id": "jk_cement_main"}
            }
        ]
