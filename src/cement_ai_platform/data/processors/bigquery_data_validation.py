"""
Automated Data Validation Framework for BigQuery Cement Plant Data

This block implements comprehensive data validation rules and automated
quality checks for the cement plant sensor data warehouse.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("Setting up Automated Data Validation Framework")
print("=" * 55)

# Define comprehensive validation rules
validation_framework = {
    "data_quality_rules": {
        "completeness": {
            "required_fields": {
                "operational_parameters": ["timestamp", "plant_id", "unit_id", "parameter_type", "parameter_name", "value"],
                "energy_consumption": ["timestamp", "plant_id", "unit_id", "energy_type", "consumption_kwh"],
                "quality_metrics": ["timestamp", "plant_id", "batch_id", "product_type", "test_type", "measured_value"],
                "production_summary": ["production_date", "plant_id", "product_type", "total_production_tons"]
            },
            "null_tolerance_percent": 1.0  # Maximum 1% null values allowed
        },
        "accuracy": {
            "value_ranges": {
                "kiln_temperature": {"min": 800, "max": 1600, "unit": "°C"},
                "kiln_pressure": {"min": 0, "max": 30, "unit": "bar"}, 
                "raw_material_flow": {"min": 0, "max": 5000, "unit": "t/h"},
                "fuel_flow": {"min": 0, "max": 1000, "unit": "m3/h"},
                "fan_speed": {"min": 0, "max": 3000, "unit": "rpm"},
                "electrical_power": {"min": 0, "max": 100000, "unit": "kW"},
                "compressive_strength": {"min": 20, "max": 80, "unit": "MPa"},
                "fineness": {"min": 250, "max": 450, "unit": "m2/kg"}
            },
            "outlier_detection": {
                "method": "z_score",
                "threshold": 3.0,
                "window_hours": 24
            }
        },
        "consistency": {
            "timestamp_validation": {
                "future_threshold_minutes": 5,  # Allow 5min future timestamps
                "past_threshold_days": 90,      # Data older than 90 days flagged
                "timezone": "UTC"
            },
            "duplicate_detection": {
                "composite_keys": {
                    "operational_parameters": ["timestamp", "plant_id", "unit_id", "parameter_name"],
                    "energy_consumption": ["timestamp", "plant_id", "unit_id", "energy_type"],
                    "quality_metrics": ["timestamp", "plant_id", "batch_id", "test_type"],
                    "production_summary": ["production_date", "plant_id", "product_type"]
                },
                "tolerance_seconds": 60  # Allow duplicates within 60 seconds
            }
        },
        "timeliness": {
            "ingestion_lag_thresholds": {
                "operational_parameters": {"warning": 300, "critical": 600},  # seconds
                "energy_consumption": {"warning": 900, "critical": 1800},     # seconds
                "quality_metrics": {"warning": 3600, "critical": 7200},      # seconds
                "production_summary": {"warning": 86400, "critical": 172800} # seconds
            }
        }
    }
}

# Create validation SQL queries
def create_validation_sql_queries():
    """Generate comprehensive validation SQL queries"""
    
    queries = {}
    
    # 1. Data Completeness Check
    queries["completeness_check"] = """
    -- Data Completeness Validation
    WITH table_completeness AS (
      SELECT 
        'operational_parameters' as table_name,
        COUNT(*) as total_records,
        COUNTIF(timestamp IS NULL) as null_timestamps,
        COUNTIF(plant_id IS NULL) as null_plant_ids,
        COUNTIF(value IS NULL) as null_values,
        DATE(CURRENT_TIMESTAMP()) as check_date
      FROM `{project_id}.{dataset_name}.operational_parameters`
      WHERE DATE(timestamp) = CURRENT_DATE()
      
      UNION ALL
      
      SELECT 
        'energy_consumption' as table_name,
        COUNT(*) as total_records,
        COUNTIF(timestamp IS NULL) as null_timestamps,
        COUNTIF(plant_id IS NULL) as null_plant_ids,
        COUNTIF(consumption_kwh IS NULL) as null_values,
        DATE(CURRENT_TIMESTAMP()) as check_date
      FROM `{project_id}.{dataset_name}.energy_consumption`
      WHERE DATE(timestamp) = CURRENT_DATE()
      
      UNION ALL
      
      SELECT 
        'quality_metrics' as table_name,
        COUNT(*) as total_records,
        COUNTIF(timestamp IS NULL) as null_timestamps,
        COUNTIF(plant_id IS NULL) as null_plant_ids,
        COUNTIF(measured_value IS NULL) as null_values,
        DATE(CURRENT_TIMESTAMP()) as check_date
      FROM `{project_id}.{dataset_name}.quality_metrics`
      WHERE DATE(timestamp) = CURRENT_DATE()
    )
    SELECT 
      table_name,
      total_records,
      null_timestamps,
      null_plant_ids,
      null_values,
      ROUND(100.0 * (total_records - null_timestamps - null_plant_ids - null_values) / total_records, 2) as completeness_score,
      CASE 
        WHEN ROUND(100.0 * (total_records - null_timestamps - null_plant_ids - null_values) / total_records, 2) >= 99 THEN 'EXCELLENT'
        WHEN ROUND(100.0 * (total_records - null_timestamps - null_plant_ids - null_values) / total_records, 2) >= 95 THEN 'GOOD'
        WHEN ROUND(100.0 * (total_records - null_timestamps - null_plant_ids - null_values) / total_records, 2) >= 90 THEN 'FAIR'
        ELSE 'POOR'
      END as quality_grade
    FROM table_completeness
    ORDER BY completeness_score DESC;
    """
    
    # 2. Value Range Validation
    queries["value_range_check"] = """
    -- Value Range Validation for Operational Parameters
    WITH range_violations AS (
      SELECT 
        plant_id,
        parameter_name,
        COUNT(*) as total_readings,
        COUNTIF(
          (parameter_name = 'kiln_temperature' AND (value < 800 OR value > 1600)) OR
          (parameter_name = 'kiln_pressure' AND (value < 0 OR value > 30)) OR
          (parameter_name = 'raw_material_flow' AND (value < 0 OR value > 5000)) OR
          (parameter_name = 'fuel_flow' AND (value < 0 OR value > 1000)) OR
          (parameter_name = 'fan_speed' AND (value < 0 OR value > 3000))
        ) as violations,
        MIN(value) as min_value,
        MAX(value) as max_value,
        AVG(value) as avg_value,
        STDDEV(value) as std_value
      FROM `{project_id}.{dataset_name}.operational_parameters`
      WHERE DATE(timestamp) = CURRENT_DATE()
      GROUP BY plant_id, parameter_name
    )
    SELECT 
      plant_id,
      parameter_name,
      total_readings,
      violations,
      ROUND(100.0 * violations / total_readings, 2) as violation_rate,
      ROUND(min_value, 2) as min_value,
      ROUND(max_value, 2) as max_value,
      ROUND(avg_value, 2) as avg_value,
      ROUND(std_value, 2) as std_value,
      CASE 
        WHEN violations = 0 THEN 'PASS'
        WHEN ROUND(100.0 * violations / total_readings, 2) < 1 THEN 'WARNING'
        ELSE 'FAIL'
      END as validation_status
    FROM range_violations
    ORDER BY violation_rate DESC, plant_id, parameter_name;
    """
    
    # 3. Duplicate Detection
    queries["duplicate_check"] = """
    -- Duplicate Record Detection
    WITH duplicates AS (
      SELECT 
        'operational_parameters' as table_name,
        timestamp,
        plant_id,
        unit_id,
        parameter_name,
        COUNT(*) as duplicate_count
      FROM `{project_id}.{dataset_name}.operational_parameters`
      WHERE DATE(timestamp) = CURRENT_DATE()
      GROUP BY timestamp, plant_id, unit_id, parameter_name
      HAVING COUNT(*) > 1
      
      UNION ALL
      
      SELECT 
        'energy_consumption' as table_name,
        timestamp,
        plant_id,
        unit_id,
        energy_type as parameter_name,
        COUNT(*) as duplicate_count
      FROM `{project_id}.{dataset_name}.energy_consumption`
      WHERE DATE(timestamp) = CURRENT_DATE()
      GROUP BY timestamp, plant_id, unit_id, energy_type
      HAVING COUNT(*) > 1
    )
    SELECT 
      table_name,
      COUNT(*) as total_duplicate_groups,
      SUM(duplicate_count) as total_duplicate_records,
      AVG(duplicate_count) as avg_duplicates_per_group
    FROM duplicates
    GROUP BY table_name
    ORDER BY total_duplicate_records DESC;
    """
    
    # 4. Data Freshness Check
    queries["freshness_check"] = """
    -- Data Freshness and Timeliness Check
    WITH freshness_metrics AS (
      SELECT 
        'operational_parameters' as table_name,
        MAX(timestamp) as latest_timestamp,
        MAX(ingestion_timestamp) as latest_ingestion,
        COUNT(*) as records_today,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(timestamp), SECOND) as data_lag_seconds,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ingestion_timestamp), SECOND) as ingestion_lag_seconds
      FROM `{project_id}.{dataset_name}.operational_parameters`
      WHERE DATE(timestamp) = CURRENT_DATE()
      
      UNION ALL
      
      SELECT 
        'energy_consumption' as table_name,
        MAX(timestamp) as latest_timestamp,
        MAX(ingestion_timestamp) as latest_ingestion,
        COUNT(*) as records_today,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(timestamp), SECOND) as data_lag_seconds,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ingestion_timestamp), SECOND) as ingestion_lag_seconds
      FROM `{project_id}.{dataset_name}.energy_consumption`
      WHERE DATE(timestamp) = CURRENT_DATE()
      
      UNION ALL
      
      SELECT 
        'quality_metrics' as table_name,
        MAX(timestamp) as latest_timestamp,
        MAX(ingestion_timestamp) as latest_ingestion,
        COUNT(*) as records_today,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(timestamp), SECOND) as data_lag_seconds,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ingestion_timestamp), SECOND) as ingestion_lag_seconds
      FROM `{project_id}.{dataset_name}.quality_metrics`
      WHERE DATE(timestamp) = CURRENT_DATE()
    )
    SELECT 
      table_name,
      latest_timestamp,
      latest_ingestion,
      records_today,
      data_lag_seconds,
      ingestion_lag_seconds,
      CASE 
        WHEN data_lag_seconds <= 300 THEN 'EXCELLENT'
        WHEN data_lag_seconds <= 600 THEN 'GOOD'
        WHEN data_lag_seconds <= 1800 THEN 'FAIR'
        ELSE 'POOR'
      END as freshness_grade
    FROM freshness_metrics
    ORDER BY data_lag_seconds ASC;
    """
    
    return queries

# Generate validation queries
validation_queries_sql = create_validation_sql_queries()

# Create automated validation report template
def create_validation_report_template():
    """Create template for automated validation reports"""
    
    template = """
    -- Automated Data Quality Report for {date}
    -- Generated at {timestamp}
    
    /* EXECUTIVE SUMMARY */
    SELECT 
      '{date}' as report_date,
      'DATA_QUALITY_SUMMARY' as report_type,
      COUNT(DISTINCT table_name) as tables_checked,
      AVG(completeness_score) as avg_completeness,
      MIN(completeness_score) as min_completeness,
      COUNTIF(quality_grade IN ('EXCELLENT', 'GOOD')) as passing_tables,
      COUNT(*) as total_tables
    FROM (
      {completeness_check_query}
    );
    
    /* DETAILED FINDINGS */
    -- 1. Completeness Results
    {completeness_check_query}
    
    -- 2. Value Range Violations  
    {value_range_check_query}
    
    -- 3. Duplicate Records
    {duplicate_check_query}
    
    -- 4. Data Freshness
    {freshness_check_query}
    
    /* RECOMMENDATIONS */
    SELECT 
      'RECOMMENDATIONS' as section,
      CASE 
        WHEN avg_completeness >= 99 THEN 'Data quality is excellent. Continue monitoring.'
        WHEN avg_completeness >= 95 THEN 'Good data quality. Monitor for consistency.'
        WHEN avg_completeness >= 90 THEN 'Acceptable quality. Review validation rules.'
        ELSE 'Poor data quality. Immediate investigation required.'
      END as recommendation
    FROM (
      SELECT AVG(completeness_score) as avg_completeness
      FROM ({completeness_check_query})
    );
    """
    
    return template

validation_report_template = create_validation_report_template()

# Create sample validation test data
def run_sample_validation():
    """Run validation on sample data"""
    
    # Create sample data for validation
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(42)
    n_records = 100
    
    _sample_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_records)],
        'plant_id': np.random.choice(['PLANT_001', 'PLANT_002'], n_records),
        'unit_id': np.random.choice(['KILN_01', 'MILL_01'], n_records),
        'parameter_type': np.random.choice(['TEMPERATURE', 'PRESSURE', 'FLOW'], n_records),
        'parameter_name': [f'param_{i}' for i in range(n_records)],
        'value': np.random.uniform(0, 1000, n_records),
        'unit': np.random.choice(['C', 'bar', 'm3/h'], n_records),
        'quality_flag': np.random.choice(['GOOD', 'WARNING'], n_records),
        'sensor_id': [f'SENSOR_{i:03d}' for i in range(n_records)],
        'ingestion_timestamp': [datetime.now() for _ in range(n_records)]
    })
    
    validation_results = {
        "completeness": {
            "total_records": len(_sample_data),
            "null_timestamps": _sample_data['timestamp'].isnull().sum(),
            "null_plant_ids": _sample_data['plant_id'].isnull().sum(),
            "null_values": _sample_data['value'].isnull().sum(),
            "completeness_score": round(100.0 * (len(_sample_data) - _sample_data.isnull().sum().sum()) / (len(_sample_data) * len(_sample_data.columns)), 2)
        },
        "value_ranges": {
            "temperature_violations": len(_sample_data[
                (_sample_data['parameter_type'] == 'TEMPERATURE') & 
                ((_sample_data['value'] < 800) | (_sample_data['value'] > 1600))
            ]),
            "pressure_violations": len(_sample_data[
                (_sample_data['parameter_type'] == 'PRESSURE') & 
                ((_sample_data['value'] < 0) | (_sample_data['value'] > 30))
            ]),
            "flow_violations": len(_sample_data[
                (_sample_data['parameter_type'] == 'FLOW') & 
                ((_sample_data['value'] < 0) | (_sample_data['value'] > 5000))
            ])
        },
        "duplicates": {
            "duplicate_count": len(_sample_data) - len(_sample_data.drop_duplicates(['timestamp', 'plant_id', 'unit_id', 'parameter_name']))
        },
        "freshness": {
            "latest_timestamp": _sample_data['timestamp'].max(),
            "data_age_hours": 24  # Sample data is 24 hours old
        }
    }
    
    return validation_results

# Run sample validation
sample_validation_results = run_sample_validation()

print("Sample Validation Results:")
print("-" * 30)
print(f"Completeness Score: {sample_validation_results['completeness']['completeness_score']}%")
print(f"Temperature Violations: {sample_validation_results['value_ranges']['temperature_violations']}")
print(f"Pressure Violations: {sample_validation_results['value_ranges']['pressure_violations']}")
print(f"Flow Violations: {sample_validation_results['value_ranges']['flow_violations']}")
print(f"Duplicate Records: {sample_validation_results['duplicates']['duplicate_count']}")

# Create monitoring thresholds and alerts
monitoring_config = {
    "alert_thresholds": {
        "completeness_score": {"warning": 95, "critical": 90},
        "value_violations_percent": {"warning": 1, "critical": 5},
        "duplicate_percent": {"warning": 0.1, "critical": 1.0},
        "data_lag_minutes": {"warning": 15, "critical": 30}
    },
    "notification_channels": {
        "email": ["data-team@company.com", "operations@company.com"],
        "slack": "#data-alerts",
        "dashboard": "data-quality-dashboard"
    },
    "reporting_schedule": {
        "real_time_monitoring": "every_5_minutes",
        "daily_report": "06:00_UTC",
        "weekly_summary": "Monday_08:00_UTC",
        "monthly_analysis": "1st_09:00_UTC"
    }
}

# Export validation framework
data_validation = {
    "validation_framework": validation_framework,
    "validation_queries": validation_queries_sql,
    "validation_report_template": validation_report_template,
    "sample_validation_results": sample_validation_results,
    "monitoring_config": monitoring_config,
    "validation_summary": {
        "total_validation_rules": len(validation_framework["data_quality_rules"]),
        "sql_queries_count": len(validation_queries_sql),
        "sample_completeness_score": sample_validation_results['completeness']['completeness_score'],
        "monitoring_thresholds": len(monitoring_config["alert_thresholds"])
    }
}

print(f"\n✅ Data validation framework configured:")
print(f"  - {data_validation['validation_summary']['total_validation_rules']} validation rule categories")
print(f"  - {data_validation['validation_summary']['sql_queries_count']} automated validation queries")
print(f"  - {data_validation['validation_summary']['monitoring_thresholds']} monitoring thresholds")
print(f"  - Sample data completeness: {data_validation['validation_summary']['sample_completeness_score']}%")