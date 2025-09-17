#!/usr/bin/env python3
"""
Production Monitoring and Alerting Setup for Cement Plant Digital Twin
Sets up comprehensive monitoring, custom metrics, and alert policies.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cement_ai_platform.gcp.production_services import get_production_services

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMonitoringSetup:
    """Sets up production monitoring and alerting for cement plant digital twin"""
    
    def __init__(self):
        self.gcp_services = get_production_services()
        self.project_id = "cement-ai-opt-38517"
        
    def setup_custom_metrics(self):
        """Create custom metric descriptors for cement plant KPIs"""
        logger.info("📊 Setting up custom metrics for cement plant monitoring...")
        
        if not self.gcp_services or not self.gcp_services.monitoring_client:
            logger.warning("⚠️ Cloud Monitoring client not available. Creating fallback metrics.")
            return self._create_fallback_metrics()
        
        try:
            from google.cloud import monitoring_v3
            
            project_name = f"projects/{self.project_id}"
            monitoring_client = self.gcp_services.monitoring_client
            
            # Define custom metrics
            custom_metrics = [
                {
                    "type": "custom.googleapis.com/cement_plant/free_lime_deviation",
                    "display_name": "Free Lime Deviation",
                    "description": "Deviation of free lime from target percentage",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "labels": [
                        {"key": "plant_id", "description": "Plant identifier"},
                        {"key": "kiln_id", "description": "Kiln identifier"},
                        {"key": "shift", "description": "Production shift"}
                    ]
                },
                {
                    "type": "custom.googleapis.com/cement_plant/energy_efficiency",
                    "display_name": "Energy Efficiency",
                    "description": "Thermal energy efficiency percentage",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "labels": [
                        {"key": "plant_id", "description": "Plant identifier"},
                        {"key": "process_unit", "description": "Process unit (kiln, mill, etc.)"},
                        {"key": "fuel_type", "description": "Type of fuel used"}
                    ]
                },
                {
                    "type": "custom.googleapis.com/cement_plant/equipment_health",
                    "display_name": "Equipment Health Score",
                    "description": "Overall equipment health score (0-1)",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "labels": [
                        {"key": "equipment_id", "description": "Equipment identifier"},
                        {"key": "equipment_type", "description": "Type of equipment"},
                        {"key": "plant_id", "description": "Plant identifier"}
                    ]
                },
                {
                    "type": "custom.googleapis.com/cement_plant/quality_compliance",
                    "display_name": "Quality Compliance Rate",
                    "description": "Percentage of products meeting quality specifications",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "labels": [
                        {"key": "plant_id", "description": "Plant identifier"},
                        {"key": "product_type", "description": "Type of cement product"},
                        {"key": "quality_parameter", "description": "Quality parameter (strength, fineness, etc.)"}
                    ]
                },
                {
                    "type": "custom.googleapis.com/cement_plant/emissions_compliance",
                    "display_name": "Emissions Compliance",
                    "description": "Emissions compliance status (1=compliant, 0=non-compliant)",
                    "metric_kind": "GAUGE",
                    "value_type": "DOUBLE",
                    "labels": [
                        {"key": "plant_id", "description": "Plant identifier"},
                        {"key": "emission_type", "description": "Type of emission (NOx, SO2, Dust, CO2)"},
                        {"key": "regulatory_limit", "description": "Regulatory limit category"}
                    ]
                },
                {
                    "type": "custom.googleapis.com/cement_plant/ai_query_tokens",
                    "display_name": "AI Query Token Usage",
                    "description": "Number of tokens used in AI queries",
                    "metric_kind": "GAUGE",
                    "value_type": "INT64",
                    "labels": [
                        {"key": "query_type", "description": "Type of AI query"},
                        {"key": "model_version", "description": "AI model version"},
                        {"key": "plant_id", "description": "Plant identifier"}
                    ]
                }
            ]
            
            # Create metric descriptors
            for metric_config in custom_metrics:
                try:
                    descriptor = monitoring_v3.MetricDescriptor()
                    descriptor.type = metric_config["type"]
                    descriptor.display_name = metric_config["display_name"]
                    descriptor.description = metric_config["description"]
                    descriptor.metric_kind = getattr(monitoring_v3.MetricDescriptor.MetricKind, metric_config["metric_kind"])
                    descriptor.value_type = getattr(monitoring_v3.MetricDescriptor.ValueType, metric_config["value_type"])
                    
                    # Add labels
                    for label_config in metric_config.get("labels", []):
                        label = monitoring_v3.LabelDescriptor()
                        label.key = label_config["key"]
                        label.description = label_config["description"]
                        descriptor.labels.append(label)
                    
                    monitoring_client.create_metric_descriptor(
                        name=project_name,
                        metric_descriptor=descriptor
                    )
                    logger.info(f"✅ Created metric: {metric_config['type']}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Metric creation warning for {metric_config['type']}: {e}")
            
            logger.info("✅ Custom metrics setup completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup custom metrics: {e}")
    
    def _create_fallback_metrics(self):
        """Create fallback metrics when Cloud Monitoring is not available"""
        logger.info("🔄 Creating fallback metrics configuration...")
        
        fallback_metrics = {
            "custom.googleapis.com/cement_plant/free_lime_deviation": {
                "display_name": "Free Lime Deviation",
                "description": "Deviation of free lime from target percentage",
                "type": "GAUGE",
                "labels": ["plant_id", "kiln_id", "shift"]
            },
            "custom.googleapis.com/cement_plant/energy_efficiency": {
                "display_name": "Energy Efficiency", 
                "description": "Thermal energy efficiency percentage",
                "type": "GAUGE",
                "labels": ["plant_id", "process_unit", "fuel_type"]
            },
            "custom.googleapis.com/cement_plant/equipment_health": {
                "display_name": "Equipment Health Score",
                "description": "Overall equipment health score (0-1)",
                "type": "GAUGE",
                "labels": ["equipment_id", "equipment_type", "plant_id"]
            }
        }
        
        import json
        os.makedirs("demo/monitoring", exist_ok=True)
        with open("demo/monitoring/fallback_metrics.json", 'w') as f:
            json.dump(fallback_metrics, f, indent=2)
        
        logger.info("✅ Fallback metrics configuration created")
    
    def create_alert_policies(self):
        """Create production alert policies for critical cement plant conditions"""
        logger.info("🚨 Creating alert policies for cement plant monitoring...")
        
        if not self.gcp_services or not self.gcp_services.monitoring_client:
            logger.warning("⚠️ Cloud Monitoring client not available. Creating fallback alerts.")
            return self._create_fallback_alerts()
        
        try:
            from google.cloud import monitoring_v3
            
            project_name = f"projects/{self.project_id}"
            alert_client = monitoring_v3.AlertPolicyServiceClient()
            
            # Define alert policies
            alert_policies = [
                {
                    "display_name": "Critical: High Free Lime Alert",
                    "documentation": {
                        "content": """
# High Free Lime Alert

**Severity**: CRITICAL
**Impact**: Product quality degradation, potential customer complaints

## Description
Free lime has exceeded 2.0% for more than 5 minutes, indicating incomplete burning in the kiln.

## Immediate Actions Required
1. **Increase kiln temperature** by 20-30°C
2. **Reduce feed rate** by 5-10%
3. **Check fuel quality** and calorific value
4. **Verify kiln speed** optimization
5. **Monitor every 15 minutes** until free lime drops below 1.5%

## Escalation
- If not resolved in 30 minutes: Notify Plant Manager
- If not resolved in 60 minutes: Notify Production Director
- If not resolved in 120 minutes: Consider production shutdown

## Prevention
- Regular kiln maintenance
- Fuel quality monitoring
- Process parameter optimization
                        """,
                        "mime_type": "text/markdown"
                    },
                    "conditions": [
                        {
                            "display_name": "Free lime > 2.0%",
                            "condition_threshold": {
                                "filter": 'metric.type="custom.googleapis.com/cement_plant/free_lime_deviation"',
                                "comparison": "COMPARISON_GREATER_THAN",
                                "threshold_value": 2.0,
                                "duration": {"seconds": 300}  # 5 minutes
                            }
                        }
                    ],
                    "combiner": "AND",
                    "enabled": True
                },
                {
                    "display_name": "Equipment Failure Risk",
                    "documentation": {
                        "content": """
# Equipment Failure Risk Alert

**Severity**: HIGH
**Impact**: Unplanned downtime, production loss, maintenance costs

## Description
Equipment health score has dropped below 0.4, indicating high risk of failure.

## Immediate Actions Required
1. **Schedule vibration analysis** for rotating equipment
2. **Check bearing temperatures** and lubrication
3. **Review motor current patterns** for anomalies
4. **Plan maintenance window** within 48-72 hours
5. **Increase monitoring frequency** to every 15 minutes

## Escalation
- If health score drops below 0.2: Immediate maintenance required
- If health score drops below 0.1: Consider emergency shutdown

## Prevention
- Predictive maintenance scheduling
- Regular equipment inspections
- Lubrication management
- Temperature monitoring
                        """,
                        "mime_type": "text/markdown"
                    },
                    "conditions": [
                        {
                            "display_name": "Equipment health < 0.4",
                            "condition_threshold": {
                                "filter": 'metric.type="custom.googleapis.com/cement_plant/equipment_health"',
                                "comparison": "COMPARISON_LESS_THAN",
                                "threshold_value": 0.4,
                                "duration": {"seconds": 600}  # 10 minutes
                            }
                        }
                    ],
                    "combiner": "AND",
                    "enabled": True
                },
                {
                    "display_name": "Energy Efficiency Degradation",
                    "documentation": {
                        "content": """
# Energy Efficiency Degradation Alert

**Severity**: MEDIUM
**Impact**: Increased operational costs, environmental impact

## Description
Energy efficiency has dropped below 85% for more than 15 minutes.

## Immediate Actions Required
1. **Check combustion efficiency** and air-fuel ratio
2. **Review preheater performance** and stage temperatures
3. **Inspect kiln insulation** for heat loss
4. **Optimize fuel-air mixing** and distribution
5. **Monitor O2 levels** (target: 2-4%)

## Escalation
- If efficiency drops below 80%: Notify Energy Manager
- If efficiency drops below 75%: Consider kiln shutdown for inspection

## Prevention
- Regular combustion tuning
- Preheater maintenance
- Insulation inspection
- Fuel quality monitoring
                        """,
                        "mime_type": "text/markdown"
                    },
                    "conditions": [
                        {
                            "display_name": "Energy efficiency < 85%",
                            "condition_threshold": {
                                "filter": 'metric.type="custom.googleapis.com/cement_plant/energy_efficiency"',
                                "comparison": "COMPARISON_LESS_THAN",
                                "threshold_value": 0.85,
                                "duration": {"seconds": 900}  # 15 minutes
                            }
                        }
                    ],
                    "combiner": "AND",
                    "enabled": True
                }
            ]
            
            # Create alert policies
            for policy_config in alert_policies:
                try:
                    policy = monitoring_v3.AlertPolicy()
                    policy.display_name = policy_config["display_name"]
                    
                    # Add documentation
                    policy.documentation = monitoring_v3.AlertPolicy.Documentation(
                        content=policy_config["documentation"]["content"],
                        mime_type=policy_config["documentation"]["mime_type"]
                    )
                    
                    # Add conditions
                    for condition_config in policy_config["conditions"]:
                        condition = monitoring_v3.AlertPolicy.Condition()
                        condition.display_name = condition_config["display_name"]
                        
                        threshold = monitoring_v3.AlertPolicy.Condition.MetricThreshold()
                        threshold.filter = condition_config["condition_threshold"]["filter"]
                        threshold.comparison = getattr(monitoring_v3.ComparisonType, condition_config["condition_threshold"]["comparison"])
                        threshold.threshold_value = condition_config["condition_threshold"]["threshold_value"]
                        threshold.duration = condition_config["condition_threshold"]["duration"]
                        
                        condition.condition_threshold = threshold
                        policy.conditions.append(condition)
                    
                    policy.combiner = getattr(monitoring_v3.AlertPolicy.ConditionCombinerType, policy_config["combiner"])
                    policy.enabled = policy_config["enabled"]
                    
                    created_policy = alert_client.create_alert_policy(
                        name=project_name,
                        alert_policy=policy
                    )
                    logger.info(f"✅ Created alert policy: {created_policy.display_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Alert policy creation failed for {policy_config['display_name']}: {e}")
            
            logger.info("✅ Alert policies setup completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup alert policies: {e}")
    
    def _create_fallback_alerts(self):
        """Create fallback alert configuration when Cloud Monitoring is not available"""
        logger.info("🔄 Creating fallback alert configuration...")
        
        fallback_alerts = {
            "high_free_lime_alert": {
                "display_name": "Critical: High Free Lime Alert",
                "condition": "free_lime_deviation > 2.0% for 5 minutes",
                "severity": "CRITICAL",
                "actions": [
                    "Increase kiln temperature by 20-30°C",
                    "Reduce feed rate by 5-10%",
                    "Check fuel quality and calorific value",
                    "Verify kiln speed optimization"
                ],
                "escalation": {
                    "30_minutes": "Notify Plant Manager",
                    "60_minutes": "Notify Production Director",
                    "120_minutes": "Consider production shutdown"
                }
            },
            "equipment_failure_risk": {
                "display_name": "Equipment Failure Risk",
                "condition": "equipment_health < 0.4 for 10 minutes",
                "severity": "HIGH",
                "actions": [
                    "Schedule vibration analysis",
                    "Check bearing temperatures and lubrication",
                    "Review motor current patterns",
                    "Plan maintenance window within 48-72 hours"
                ],
                "escalation": {
                    "health_score_0.2": "Immediate maintenance required",
                    "health_score_0.1": "Consider emergency shutdown"
                }
            },
            "energy_efficiency_degradation": {
                "display_name": "Energy Efficiency Degradation",
                "condition": "energy_efficiency < 85% for 15 minutes",
                "severity": "MEDIUM",
                "actions": [
                    "Check combustion efficiency and air-fuel ratio",
                    "Review preheater performance",
                    "Inspect kiln insulation for heat loss",
                    "Optimize fuel-air mixing"
                ],
                "escalation": {
                    "efficiency_80%": "Notify Energy Manager",
                    "efficiency_75%": "Consider kiln shutdown for inspection"
                }
            }
        }
        
        import json
        os.makedirs("demo/monitoring", exist_ok=True)
        with open("demo/monitoring/fallback_alerts.json", 'w') as f:
            json.dump(fallback_alerts, f, indent=2)
        
        logger.info("✅ Fallback alert configuration created")
    
    def test_monitoring_system(self):
        """Test the monitoring system by sending sample metrics"""
        logger.info("🧪 Testing monitoring system with sample metrics...")
        
        if not self.gcp_services:
            logger.warning("⚠️ GCP services not available. Skipping monitoring test.")
            return
        
        # Test metrics
        test_metrics = [
            {
                "metric_name": "free_lime_deviation",
                "value": 1.2,
                "labels": {"plant_id": "jk_cement_main", "kiln_id": "kiln_01", "shift": "day"}
            },
            {
                "metric_name": "energy_efficiency",
                "value": 0.88,
                "labels": {"plant_id": "jk_cement_main", "process_unit": "kiln", "fuel_type": "coal"}
            },
            {
                "metric_name": "equipment_health",
                "value": 0.92,
                "labels": {"equipment_id": "raw_mill_01", "equipment_type": "mill", "plant_id": "jk_cement_main"}
            },
            {
                "metric_name": "ai_query_tokens",
                "value": 150,
                "labels": {"query_type": "quality_analysis", "model_version": "gemini-1.5-pro", "plant_id": "jk_cement_main"}
            }
        ]
        
        for metric in test_metrics:
            success = self.gcp_services.send_custom_metric(
                metric["metric_name"],
                metric["value"],
                metric["labels"]
            )
            
            if success:
                logger.info(f"✅ Sent test metric: {metric['metric_name']} = {metric['value']}")
            else:
                logger.warning(f"⚠️ Failed to send metric: {metric['metric_name']}")
    
    def create_monitoring_dashboard_config(self):
        """Create monitoring dashboard configuration"""
        logger.info("📊 Creating monitoring dashboard configuration...")
        
        dashboard_config = {
            "dashboard_title": "Cement Plant Digital Twin - Production Monitoring",
            "dashboard_description": "Real-time monitoring dashboard for cement plant operations",
            "widgets": [
                {
                    "title": "Free Lime Deviation",
                    "type": "scorecard",
                    "metric": "custom.googleapis.com/cement_plant/free_lime_deviation",
                    "thresholds": [
                        {"value": 2.0, "color": "RED", "label": "Critical"},
                        {"value": 1.5, "color": "YELLOW", "label": "Warning"},
                        {"value": 1.0, "color": "GREEN", "label": "Normal"}
                    ]
                },
                {
                    "title": "Energy Efficiency",
                    "type": "scorecard", 
                    "metric": "custom.googleapis.com/cement_plant/energy_efficiency",
                    "thresholds": [
                        {"value": 0.85, "color": "GREEN", "label": "Good"},
                        {"value": 0.80, "color": "YELLOW", "label": "Fair"},
                        {"value": 0.75, "color": "RED", "label": "Poor"}
                    ]
                },
                {
                    "title": "Equipment Health Score",
                    "type": "scorecard",
                    "metric": "custom.googleapis.com/cement_plant/equipment_health",
                    "thresholds": [
                        {"value": 0.8, "color": "GREEN", "label": "Healthy"},
                        {"value": 0.4, "color": "YELLOW", "label": "At Risk"},
                        {"value": 0.2, "color": "RED", "label": "Critical"}
                    ]
                },
                {
                    "title": "AI Query Token Usage",
                    "type": "line_chart",
                    "metric": "custom.googleapis.com/cement_plant/ai_query_tokens",
                    "aggregation": "SUM",
                    "time_range": "1h"
                }
            ],
            "refresh_interval": "30s",
            "created_at": datetime.now().isoformat()
        }
        
        import json
        os.makedirs("demo/monitoring", exist_ok=True)
        with open("demo/monitoring/dashboard_config.json", 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        logger.info("✅ Monitoring dashboard configuration created")

def main():
    """Main function to setup production monitoring and alerting"""
    logger.info("🚀 Starting Production Monitoring and Alerting Setup")
    
    try:
        setup = ProductionMonitoringSetup()
        
        # Setup custom metrics
        setup.setup_custom_metrics()
        
        # Create alert policies
        setup.create_alert_policies()
        
        # Test monitoring system
        setup.test_monitoring_system()
        
        # Create dashboard configuration
        setup.create_monitoring_dashboard_config()
        
        logger.info("✅ Production Monitoring and Alerting Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Production Monitoring and Alerting Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
