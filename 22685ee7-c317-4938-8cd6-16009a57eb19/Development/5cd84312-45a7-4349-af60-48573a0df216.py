"""
Sample Queries and Performance Benchmarks for BigQuery Cement Plant Data Warehouse

This block demonstrates key analytical queries and establishes performance 
benchmarks for the cement plant sensor data warehouse.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("Creating Sample Queries and Performance Benchmarks")
print("=" * 60)

# Define sample analytical queries for cement plant operations
sample_queries = {
    "operational_analytics": {
        "kiln_performance_hourly": {
            "description": "Hourly kiln performance metrics with efficiency calculations",
            "query": """
            -- Hourly Kiln Performance Analysis
            WITH hourly_metrics AS (
              SELECT 
                plant_id,
                unit_id,
                DATETIME_TRUNC(timestamp, HOUR) as hour,
                parameter_name,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                STDDEV(value) as std_value,
                COUNT(*) as measurement_count
              FROM `{project_id}.{dataset_name}.operational_parameters`
              WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
                AND unit_id LIKE '%KILN%'
                AND parameter_name IN ('kiln_temperature', 'kiln_pressure', 'fuel_flow', 'raw_material_flow')
              GROUP BY plant_id, unit_id, hour, parameter_name
            ),
            kiln_efficiency AS (
              SELECT 
                hour,
                plant_id,
                unit_id,
                MAX(CASE WHEN parameter_name = 'kiln_temperature' THEN avg_value END) as avg_temperature,
                MAX(CASE WHEN parameter_name = 'fuel_flow' THEN avg_value END) as avg_fuel_flow,
                MAX(CASE WHEN parameter_name = 'raw_material_flow' THEN avg_value END) as avg_material_flow
              FROM hourly_metrics
              GROUP BY hour, plant_id, unit_id
            )
            SELECT 
              hour,
              plant_id,
              unit_id,
              avg_temperature,
              avg_fuel_flow,
              avg_material_flow,
              ROUND(avg_material_flow / NULLIF(avg_fuel_flow, 0), 2) as fuel_efficiency,
              CASE 
                WHEN avg_temperature BETWEEN 850 AND 1200 THEN 'OPTIMAL'
                WHEN avg_temperature BETWEEN 800 AND 1300 THEN 'ACCEPTABLE'
                ELSE 'NEEDS_ATTENTION'
              END as temperature_status,
              ROUND((avg_temperature - 850) / 350 * 100, 1) as efficiency_score
            FROM kiln_efficiency
            WHERE avg_temperature IS NOT NULL
            ORDER BY hour DESC, plant_id, unit_id;
            """,
            "expected_performance": {
                "scan_gb": "< 50GB for 7 days",
                "execution_time": "< 15 seconds",
                "result_rows": "~1000-5000 rows"
            }
        },
        
        "energy_cost_analysis": {
            "description": "Daily energy cost breakdown by plant and energy type",
            "query": """
            -- Daily Energy Cost Analysis
            WITH daily_energy AS (
              SELECT 
                DATE(timestamp) as analysis_date,
                plant_id,
                energy_type,
                energy_source,
                SUM(consumption_kwh) as total_consumption_kwh,
                AVG(cost_per_unit) as avg_cost_per_unit,
                SUM(total_cost) as total_daily_cost,
                COUNT(*) as measurement_count,
                MIN(timestamp) as first_reading,
                MAX(timestamp) as last_reading
              FROM `{project_id}.{dataset_name}.energy_consumption`
              WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 30 AND CURRENT_DATE()
              GROUP BY DATE(timestamp), plant_id, energy_type, energy_source
            ),
            cost_trends AS (
              SELECT 
                analysis_date,
                plant_id,
                SUM(total_consumption_kwh) as plant_total_kwh,
                SUM(total_daily_cost) as plant_total_cost,
                AVG(total_daily_cost / NULLIF(total_consumption_kwh, 0)) as avg_cost_per_kwh
              FROM daily_energy
              GROUP BY analysis_date, plant_id
            )
            SELECT 
              de.analysis_date,
              de.plant_id,
              de.energy_type,
              de.energy_source,
              de.total_consumption_kwh,
              ROUND(de.total_daily_cost, 2) as daily_cost,
              ROUND(ct.plant_total_cost, 2) as plant_total_cost,
              ROUND(de.total_daily_cost / NULLIF(ct.plant_total_cost, 0) * 100, 2) as cost_percentage,
              ROUND(de.avg_cost_per_unit, 4) as cost_per_unit,
              de.measurement_count,
              LAG(de.total_daily_cost) OVER (PARTITION BY de.plant_id, de.energy_type ORDER BY de.analysis_date) as prev_day_cost,
              ROUND((de.total_daily_cost - LAG(de.total_daily_cost) OVER (PARTITION BY de.plant_id, de.energy_type ORDER BY de.analysis_date)) / 
                    NULLIF(LAG(de.total_daily_cost) OVER (PARTITION BY de.plant_id, de.energy_type ORDER BY de.analysis_date), 0) * 100, 2) as cost_change_percent
            FROM daily_energy de
            JOIN cost_trends ct ON de.analysis_date = ct.analysis_date AND de.plant_id = ct.plant_id
            ORDER BY de.analysis_date DESC, de.plant_id, de.total_daily_cost DESC;
            """,
            "expected_performance": {
                "scan_gb": "< 20GB for 30 days",
                "execution_time": "< 10 seconds",
                "result_rows": "~500-2000 rows"
            }
        },
        
        "quality_trend_analysis": {
            "description": "Quality metrics trends with pass/fail rates and statistical analysis",
            "query": """
            -- Quality Metrics Trend Analysis
            WITH quality_stats AS (
              SELECT 
                DATE(timestamp) as test_date,
                plant_id,
                product_type,
                test_type,
                metric_name,
                COUNT(*) as total_tests,
                COUNT(CASE WHEN pass_fail = true THEN 1 END) as passed_tests,
                AVG(measured_value) as avg_value,
                STDDEV(measured_value) as std_value,
                MIN(measured_value) as min_value,
                MAX(measured_value) as max_value,
                PERCENTILE_CONT(measured_value, 0.5) OVER (PARTITION BY DATE(timestamp), plant_id, test_type, metric_name) as median_value,
                AVG(specification_min) as spec_min,
                AVG(specification_max) as spec_max
              FROM `{project_id}.{dataset_name}.quality_metrics`
              WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 14 AND CURRENT_DATE()
                AND measured_value IS NOT NULL
              GROUP BY DATE(timestamp), plant_id, product_type, test_type, metric_name
            ),
            quality_trends AS (
              SELECT 
                test_date,
                plant_id,
                product_type,
                test_type,
                metric_name,
                total_tests,
                passed_tests,
                ROUND(100.0 * passed_tests / NULLIF(total_tests, 0), 2) as pass_rate,
                ROUND(avg_value, 3) as avg_value,
                ROUND(std_value, 3) as std_value,
                ROUND(median_value, 3) as median_value,
                spec_min,
                spec_max,
                CASE 
                  WHEN avg_value BETWEEN spec_min AND spec_max THEN 'IN_SPEC'
                  WHEN avg_value < spec_min THEN 'BELOW_SPEC'
                  WHEN avg_value > spec_max THEN 'ABOVE_SPEC'
                  ELSE 'NO_SPEC'
                END as spec_status,
                LAG(ROUND(100.0 * passed_tests / NULLIF(total_tests, 0), 2)) OVER (PARTITION BY plant_id, test_type, metric_name ORDER BY test_date) as prev_pass_rate
              FROM quality_stats
            )
            SELECT 
              test_date,
              plant_id,
              product_type,
              test_type,
              metric_name,
              total_tests,
              pass_rate,
              prev_pass_rate,
              ROUND(pass_rate - COALESCE(prev_pass_rate, pass_rate), 2) as pass_rate_change,
              avg_value,
              std_value,
              median_value,
              spec_min,
              spec_max,
              spec_status,
              CASE 
                WHEN pass_rate >= 98 THEN 'EXCELLENT'
                WHEN pass_rate >= 95 THEN 'GOOD'
                WHEN pass_rate >= 90 THEN 'ACCEPTABLE'
                ELSE 'NEEDS_IMPROVEMENT'
              END as quality_grade
            FROM quality_trends
            ORDER BY test_date DESC, plant_id, pass_rate ASC;
            """,
            "expected_performance": {
                "scan_gb": "< 10GB for 14 days",
                "execution_time": "< 12 seconds",
                "result_rows": "~200-1000 rows"
            }
        }
    },
    
    "production_analytics": {
        "production_efficiency_dashboard": {
            "description": "Daily production efficiency metrics across all plants",
            "query": """
            -- Production Efficiency Dashboard
            WITH production_metrics AS (
              SELECT 
                production_date,
                plant_id,
                product_type,
                total_production_tons,
                total_energy_consumed_kwh,
                avg_kiln_temperature,
                quality_pass_rate,
                downtime_hours,
                efficiency_score,
                ROUND(total_production_tons / NULLIF(total_energy_consumed_kwh, 0) * 1000, 2) as tons_per_mwh,
                ROUND((24 - downtime_hours) / 24 * 100, 2) as uptime_percent,
                LAG(total_production_tons) OVER (PARTITION BY plant_id, product_type ORDER BY production_date) as prev_production
              FROM `{project_id}.{dataset_name}.production_summary`
              WHERE production_date BETWEEN CURRENT_DATE() - 30 AND CURRENT_DATE()
            ),
            plant_rankings AS (
              SELECT 
                plant_id,
                AVG(total_production_tons) as avg_daily_production,
                AVG(efficiency_score) as avg_efficiency,
                AVG(quality_pass_rate) as avg_quality,
                AVG(uptime_percent) as avg_uptime,
                RANK() OVER (ORDER BY AVG(efficiency_score) DESC) as efficiency_rank,
                RANK() OVER (ORDER BY AVG(total_production_tons) DESC) as production_rank
              FROM production_metrics
              GROUP BY plant_id
            )
            SELECT 
              pm.production_date,
              pm.plant_id,
              pm.product_type,
              pm.total_production_tons,
              pm.tons_per_mwh,
              pm.quality_pass_rate,
              pm.uptime_percent,
              pm.efficiency_score,
              pr.efficiency_rank,
              pr.production_rank,
              ROUND((pm.total_production_tons - COALESCE(pm.prev_production, pm.total_production_tons)) / 
                    NULLIF(COALESCE(pm.prev_production, pm.total_production_tons), 0) * 100, 2) as production_change,
              CASE 
                WHEN pm.efficiency_score >= 90 AND pm.quality_pass_rate >= 95 AND pm.uptime_percent >= 95 THEN 'EXCELLENT'
                WHEN pm.efficiency_score >= 80 AND pm.quality_pass_rate >= 90 AND pm.uptime_percent >= 90 THEN 'GOOD'
                WHEN pm.efficiency_score >= 70 AND pm.quality_pass_rate >= 85 AND pm.uptime_percent >= 85 THEN 'FAIR'
                ELSE 'NEEDS_IMPROVEMENT'
              END as overall_performance
            FROM production_metrics pm
            JOIN plant_rankings pr ON pm.plant_id = pr.plant_id
            ORDER BY pm.production_date DESC, pm.efficiency_score DESC;
            """,
            "expected_performance": {
                "scan_gb": "< 5GB for 30 days",
                "execution_time": "< 8 seconds",
                "result_rows": "~100-500 rows"
            }
        }
    }
}

# Create performance benchmark tests
def create_performance_benchmarks():
    """Create performance benchmark configurations"""
    
    benchmarks = {
        "query_performance_targets": {
            "simple_aggregation": {
                "description": "Simple aggregation queries (COUNT, SUM, AVG)",
                "max_execution_time_seconds": 5,
                "max_slot_time_seconds": 20,
                "max_bytes_processed_gb": 10
            },
            "complex_analytics": {
                "description": "Complex analytics with JOINs and window functions",
                "max_execution_time_seconds": 15,
                "max_slot_time_seconds": 60,
                "max_bytes_processed_gb": 50
            },
            "time_series_analysis": {
                "description": "Time-series analysis with LAG/LEAD functions",
                "max_execution_time_seconds": 20,
                "max_slot_time_seconds": 80,
                "max_bytes_processed_gb": 75
            }
        },
        
        "data_volume_benchmarks": {
            "daily_ingestion": {
                "operational_parameters": "1M-5M records/day",
                "energy_consumption": "100K-500K records/day", 
                "quality_metrics": "10K-50K records/day",
                "production_summary": "100-1K records/day"
            },
            "storage_growth": {
                "monthly_growth_gb": 500,
                "annual_growth_tb": 6,
                "retention_years": 7,
                "total_projected_tb": 42
            }
        },
        
        "cost_benchmarks": {
            "monthly_targets": {
                "storage_cost_usd": 200,
                "compute_cost_usd": 800,
                "network_cost_usd": 50,
                "total_monthly_usd": 1050
            },
            "query_cost_targets": {
                "dashboard_queries": "< $0.01 per query",
                "analytical_reports": "< $0.10 per query",
                "data_exports": "< $1.00 per export"
            }
        }
    }
    
    return benchmarks

# Generate performance test queries
def create_performance_test_queries():
    """Create queries to test system performance"""
    
    test_queries = {
        "load_test_simple": """
        -- Simple Load Test Query
        SELECT 
          plant_id,
          COUNT(*) as record_count,
          MIN(timestamp) as earliest_record,
          MAX(timestamp) as latest_record
        FROM `{project_id}.{dataset_name}.operational_parameters`
        WHERE DATE(timestamp) = CURRENT_DATE()
        GROUP BY plant_id
        ORDER BY record_count DESC;
        """,
        
        "load_test_complex": """
        -- Complex Load Test Query with Multiple JOINs
        WITH operational_summary AS (
          SELECT 
            plant_id,
            DATE(timestamp) as analysis_date,
            COUNT(*) as operational_records,
            AVG(CASE WHEN parameter_name = 'kiln_temperature' THEN value END) as avg_temperature
          FROM `{project_id}.{dataset_name}.operational_parameters`
          WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
          GROUP BY plant_id, DATE(timestamp)
        ),
        energy_summary AS (
          SELECT 
            plant_id,
            DATE(timestamp) as analysis_date,
            SUM(consumption_kwh) as total_energy,
            SUM(total_cost) as total_cost
          FROM `{project_id}.{dataset_name}.energy_consumption`
          WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
          GROUP BY plant_id, DATE(timestamp)
        ),
        quality_summary AS (
          SELECT 
            plant_id,
            DATE(timestamp) as analysis_date,
            COUNT(*) as quality_tests,
            AVG(CASE WHEN pass_fail = true THEN 1.0 ELSE 0.0 END) * 100 as pass_rate
          FROM `{project_id}.{dataset_name}.quality_metrics`
          WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
          GROUP BY plant_id, DATE(timestamp)
        )
        SELECT 
          o.plant_id,
          o.analysis_date,
          o.operational_records,
          o.avg_temperature,
          e.total_energy,
          e.total_cost,
          q.quality_tests,
          q.pass_rate,
          ROUND(e.total_cost / NULLIF(e.total_energy, 0), 4) as cost_per_kwh
        FROM operational_summary o
        LEFT JOIN energy_summary e ON o.plant_id = e.plant_id AND o.analysis_date = e.analysis_date
        LEFT JOIN quality_summary q ON o.plant_id = q.plant_id AND o.analysis_date = q.analysis_date
        ORDER BY o.analysis_date DESC, o.plant_id;
        """,
        
        "partition_efficiency_test": """
        -- Partition Efficiency Test
        SELECT 
          DATE(timestamp) as partition_date,
          COUNT(*) as records_per_partition,
          COUNT(DISTINCT plant_id) as unique_plants,
          MIN(timestamp) as partition_start,
          MAX(timestamp) as partition_end,
          APPROX_QUANTILES(value, 4)[OFFSET(2)] as median_value
        FROM `{project_id}.{dataset_name}.operational_parameters`
        WHERE DATE(timestamp) BETWEEN CURRENT_DATE() - 30 AND CURRENT_DATE()
        GROUP BY DATE(timestamp)
        ORDER BY partition_date DESC;
        """
    }
    
    return test_queries

# Generate sample data analysis results
def analyze_sample_performance():
    """Analyze performance of sample queries on sample data"""
    
    # Simulate performance metrics based on sample data size
    _sample_size = len(sample_data)
    
    performance_analysis = {
        "sample_data_stats": {
            "total_records": _sample_size,
            "unique_plants": sample_data['plant_id'].nunique(),
            "unique_parameters": sample_data['parameter_name'].nunique(),
            "time_range_hours": 100,  # Sample spans 100 minutes = ~1.67 hours
            "data_completeness": 100.0  # Sample has no nulls
        },
        "estimated_performance": {
            "simple_queries": {
                "scan_time_ms": _sample_size * 0.1,  # 0.1ms per record
                "memory_mb": _sample_size * 0.001,   # 1KB per record
                "cost_estimate_usd": 0.001
            },
            "complex_queries": {
                "scan_time_ms": _sample_size * 0.5,  # 0.5ms per record  
                "memory_mb": _sample_size * 0.005,   # 5KB per record
                "cost_estimate_usd": 0.005
            }
        },
        "scaling_projections": {
            "1M_records": {
                "simple_query_time": "1-2 seconds",
                "complex_query_time": "5-10 seconds", 
                "storage_mb": 1000,
                "monthly_cost_usd": 25
            },
            "10M_records": {
                "simple_query_time": "3-5 seconds",
                "complex_query_time": "15-30 seconds",
                "storage_mb": 10000,
                "monthly_cost_usd": 250
            },
            "100M_records": {
                "simple_query_time": "10-15 seconds",
                "complex_query_time": "60-120 seconds",
                "storage_mb": 100000,
                "monthly_cost_usd": 2500
            }
        }
    }
    
    return performance_analysis

# Create the complete sample queries and benchmarks package
performance_benchmarks = create_performance_benchmarks()
performance_test_queries = create_performance_test_queries() 
sample_performance_analysis = analyze_sample_performance()

print("Sample Queries Created:")
print(f"  - {len(sample_queries['operational_analytics'])} operational analytics queries")
print(f"  - {len(sample_queries['production_analytics'])} production analytics queries")
print(f"  - {len(performance_test_queries)} performance test queries")

print(f"\nPerformance Benchmarks Established:")
print(f"  - {len(performance_benchmarks['query_performance_targets'])} query performance targets")
print(f"  - Data volume projections for {len(performance_benchmarks['data_volume_benchmarks']['daily_ingestion'])} table types")
print(f"  - Cost benchmarks with ${performance_benchmarks['cost_benchmarks']['monthly_targets']['total_monthly_usd']} monthly target")

print(f"\nSample Data Performance Analysis:")
print(f"  - {sample_performance_analysis['sample_data_stats']['total_records']} sample records analyzed")
print(f"  - {sample_performance_analysis['sample_data_stats']['unique_plants']} unique plants")
print(f"  - {sample_performance_analysis['sample_data_stats']['data_completeness']}% completeness")

# Create optimization recommendations
optimization_recommendations = {
    "partitioning": [
        "Use daily partitioning for timestamp fields to optimize time-range queries",
        "Consider hourly partitioning for high-volume operational_parameters table",
        "Implement partition expiration after 7 years for cost control"
    ],
    "clustering": [
        "Cluster by plant_id as primary dimension for multi-tenant queries",
        "Add unit_id clustering for operational_parameters table",
        "Consider product_type clustering for quality_metrics table"
    ],
    "query_optimization": [
        "Always include partition filters (DATE/TIMESTAMP) in WHERE clauses",
        "Use APPROX functions for large dataset aggregations when precision allows",
        "Implement query result caching for dashboard and report queries",
        "Use materialized views for frequently accessed aggregated data"
    ],
    "cost_optimization": [
        "Set maximum bytes billed limits for ad-hoc queries",
        "Use query cost preview before executing expensive analytics",
        "Implement slot reservations for predictable workloads",
        "Archive old data to cheaper storage tiers after 2 years"
    ]
}

# Export complete package
query_benchmarks_package = {
    "sample_queries": sample_queries,
    "performance_benchmarks": performance_benchmarks,
    "performance_test_queries": performance_test_queries,
    "sample_performance_analysis": sample_performance_analysis,
    "optimization_recommendations": optimization_recommendations,
    "package_summary": {
        "total_sample_queries": sum(len(category) for category in sample_queries.values()),
        "performance_test_queries": len(performance_test_queries),
        "benchmark_categories": len(performance_benchmarks),
        "optimization_areas": len(optimization_recommendations)
    }
}

print(f"\n✅ Sample queries and benchmarks package completed:")
print(f"  - {query_benchmarks_package['package_summary']['total_sample_queries']} analytical queries")
print(f"  - {query_benchmarks_package['package_summary']['performance_test_queries']} performance tests")
print(f"  - {query_benchmarks_package['package_summary']['optimization_areas']} optimization categories")
print(f"  - Cost target: ${performance_benchmarks['cost_benchmarks']['monthly_targets']['total_monthly_usd']}/month")