import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

def generate_comprehensive_validation_report(
    completeness_results: Dict,
    mass_balance_results: Dict,
    temperature_results: Dict,
    energy_results: Dict,
    temporal_results: Dict
) -> Dict[str, Any]:
    """
    Generate comprehensive data validation report combining all validation components
    
    Returns:
        Dictionary with comprehensive validation results
    """
    
    # Initialize comprehensive report structure
    comprehensive_report = {
        'validation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_validation_passed': True,
        'overall_quality_score': 0.0,
        'validation_summary': {},
        'detailed_results': {},
        'critical_issues': [],
        'warnings': [],
        'recommendations': [],
        'data_quality_metrics': {},
        'anomaly_detection_summary': {},
        'completeness_percentage': 100.0
    }
    
    # Extract validation results from each component
    validation_components = {
        'data_completeness': completeness_results,
        'mass_balance_constraints': mass_balance_results, 
        'temperature_profile_consistency': temperature_results,
        'energy_consumption_alignment': energy_results,
        'temporal_data_integrity': temporal_results
    }
    
    # Component weights for overall score calculation
    component_weights = {
        'data_completeness': 0.25,
        'mass_balance_constraints': 0.25,
        'temperature_profile_consistency': 0.20,
        'energy_consumption_alignment': 0.15,
        'temporal_data_integrity': 0.15
    }
    
    # Aggregate results from each component
    component_scores = {}
    component_status = {}
    all_issues = []
    all_warnings = []
    all_recommendations = []
    
    for component_name, results in validation_components.items():
        if results is None or not isinstance(results, dict):
            component_scores[component_name] = 0.0
            component_status[component_name] = False
            all_issues.append(f"No results available for {component_name}")
            continue
            
        # Extract key metrics from each component
        if component_name == 'data_completeness':
            score = results.get('data_quality_score', 0.0)
            passed = results.get('validation_passed', False)
            component_status[component_name] = passed
            component_scores[component_name] = score
            
            # Extract issues
            issues = results.get('issues_found', [])
            all_issues.extend([f"Data Completeness: {issue}" for issue in issues])
            
            # Extract recommendations
            recs = results.get('recommendations', [])
            all_recommendations.extend([f"Data Completeness: {rec}" for rec in recs])
            
        elif component_name == 'mass_balance_constraints':
            score = results.get('overall_balance_score', 0.0)
            passed = results.get('validation_passed', False)
            component_status[component_name] = passed
            component_scores[component_name] = score
            
            # Extract issues
            issues = results.get('constraint_violations', [])
            all_issues.extend([f"Mass Balance: {issue}" for issue in issues[:3]])
            
            # Extract recommendations
            recs = results.get('recommendations', [])
            all_recommendations.extend([f"Mass Balance: {rec}" for rec in recs])
            
        elif component_name == 'temperature_profile_consistency':
            score = results.get('consistency_score', 0.0)
            passed = results.get('validation_passed', False)
            component_status[component_name] = passed
            component_scores[component_name] = score
            
            # Extract issues
            issues = results.get('issues_found', [])
            all_issues.extend([f"Temperature Profile: {issue}" for issue in issues])
            
            # Extract recommendations  
            recs = results.get('recommendations', [])
            all_recommendations.extend([f"Temperature Profile: {rec}" for rec in recs])
            
        elif component_name == 'energy_consumption_alignment':
            score = results.get('alignment_score', 0.0) 
            passed = results.get('validation_passed', False)
            component_status[component_name] = passed
            component_scores[component_name] = score
            
            # Extract issues
            issues = results.get('issues_found', [])
            all_issues.extend([f"Energy Consumption: {issue}" for issue in issues])
            
            # Extract recommendations
            recs = results.get('recommendations', [])
            all_recommendations.extend([f"Energy Consumption: {rec}" for rec in recs])
            
        elif component_name == 'temporal_data_integrity':
            score = results.get('data_continuity_score', 0.0)
            passed = results.get('validation_passed', False)
            component_status[component_name] = passed
            component_scores[component_name] = score
            
            # Extract issues
            issues = results.get('issues_found', [])
            all_issues.extend([f"Temporal Integrity: {issue}" for issue in issues])
            
            # Extract recommendations
            recs = results.get('recommendations', [])
            all_recommendations.extend([f"Temporal Integrity: {rec}" for rec in recs])
    
    # Calculate overall quality score
    weighted_score = 0.0
    total_weight = 0.0
    
    for component, score in component_scores.items():
        weight = component_weights.get(component, 0.2)
        weighted_score += score * weight
        total_weight += weight
    
    if total_weight > 0:
        comprehensive_report['overall_quality_score'] = weighted_score / total_weight
    
    # Determine overall validation status
    comprehensive_report['overall_validation_passed'] = all(component_status.values())
    
    # Calculate completeness percentage
    passed_components = sum(1 for passed in component_status.values() if passed)
    comprehensive_report['completeness_percentage'] = (passed_components / len(component_status)) * 100
    
    # Populate detailed results
    comprehensive_report['detailed_results'] = {
        'component_scores': component_scores,
        'component_status': component_status,
        'component_weights': component_weights,
        'weighted_contributions': {
            comp: score * component_weights.get(comp, 0.2) 
            for comp, score in component_scores.items()
        }
    }
    
    # Categorize issues by severity
    critical_issues = [issue for issue in all_issues if any(
        keyword in issue.lower() for keyword in ['critical', 'failed', 'violation', 'error']
    )]
    
    warning_issues = [issue for issue in all_issues if issue not in critical_issues]
    
    comprehensive_report['critical_issues'] = critical_issues[:10]  # Top 10 critical
    comprehensive_report['warnings'] = warning_issues[:15]  # Top 15 warnings
    comprehensive_report['recommendations'] = all_recommendations[:20]  # Top 20 recommendations
    
    # Validation summary
    comprehensive_report['validation_summary'] = {
        'total_components_validated': len(validation_components),
        'components_passed': sum(component_status.values()),
        'components_failed': len(component_status) - sum(component_status.values()),
        'overall_score': comprehensive_report['overall_quality_score'],
        'critical_issues_count': len(critical_issues),
        'warnings_count': len(warning_issues),
        'recommendations_count': len(all_recommendations)
    }
    
    # Data quality metrics aggregation
    comprehensive_report['data_quality_metrics'] = {
        'data_completeness_score': component_scores.get('data_completeness', 0.0),
        'physical_constraints_score': component_scores.get('mass_balance_constraints', 0.0),
        'temporal_consistency_score': component_scores.get('temporal_data_integrity', 0.0),
        'process_validation_score': (
            component_scores.get('temperature_profile_consistency', 0.0) + 
            component_scores.get('energy_consumption_alignment', 0.0)
        ) / 2.0,
        'anomaly_detection_score': min(component_scores.values()) if component_scores else 0.0
    }
    
    # Anomaly detection summary
    anomaly_counts = {
        'mass_balance_violations': len(mass_balance_results.get('constraint_violations', [])),
        'temperature_outliers': temperature_results.get('outlier_count', 0),
        'energy_anomalies': energy_results.get('outlier_count', 0), 
        'temporal_gaps': temporal_results.get('temporal_gaps', 0),
        'temporal_duplicates': temporal_results.get('duplicate_timestamps', 0)
    }
    
    comprehensive_report['anomaly_detection_summary'] = {
        'total_anomalies_detected': sum(anomaly_counts.values()),
        'anomaly_breakdown': anomaly_counts,
        'anomaly_rate': sum(anomaly_counts.values()) / max(
            temporal_results.get('total_samples', 1000), 1
        ) * 100
    }
    
    return comprehensive_report

# Generate comprehensive validation report
validation_report = generate_comprehensive_validation_report(
    completeness_results=completeness_validation,
    mass_balance_results=mass_balance_results,
    temperature_results=temperature_profile_results,
    energy_results=energy_consumption_results,
    temporal_results=temporal_integrity_results
)

print("📊 COMPREHENSIVE DATA VALIDATION REPORT")
print("=" * 60)
print(f"Validation Timestamp: {validation_report['validation_timestamp']}")
print(f"Overall Validation: {'✅ PASSED' if validation_report['overall_validation_passed'] else '❌ FAILED'}")
print(f"Overall Quality Score: {validation_report['overall_quality_score']:.3f}")
print(f"Data Completeness: {validation_report['completeness_percentage']:.1f}%")

print(f"\n🎯 VALIDATION SUMMARY")
summary = validation_report['validation_summary']
print(f"Components Validated: {summary['total_components_validated']}")
print(f"Components Passed: {summary['components_passed']}")
print(f"Components Failed: {summary['components_failed']}")
print(f"Critical Issues: {summary['critical_issues_count']}")
print(f"Warnings: {summary['warnings_count']}")

print(f"\n📈 COMPONENT SCORES")
for component, score in validation_report['detailed_results']['component_scores'].items():
    status = "✅" if validation_report['detailed_results']['component_status'][component] else "❌"
    component_name = component.replace('_', ' ').title()
    print(f"  {status} {component_name}: {score:.3f}")

print(f"\n📊 DATA QUALITY METRICS")
metrics = validation_report['data_quality_metrics']
for metric, value in metrics.items():
    metric_name = metric.replace('_', ' ').title()
    print(f"  {metric_name}: {value:.3f}")

print(f"\n🔍 ANOMALY DETECTION SUMMARY")
anomaly_summary = validation_report['anomaly_detection_summary']
print(f"Total Anomalies: {anomaly_summary['total_anomalies_detected']}")
print(f"Anomaly Rate: {anomaly_summary['anomaly_rate']:.2f}%")

if validation_report['critical_issues']:
    print(f"\n⚠️  CRITICAL ISSUES ({len(validation_report['critical_issues'])}):")
    for issue in validation_report['critical_issues'][:5]:
        print(f"  • {issue}")

if validation_report['warnings']:
    print(f"\n⚡ WARNINGS ({len(validation_report['warnings'])}):")
    for warning in validation_report['warnings'][:3]:
        print(f"  • {warning}")

print(f"\n💡 KEY RECOMMENDATIONS:")
for rec in validation_report['recommendations'][:5]:
    print(f"  • {rec}")

# Create visual summary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Component scores radar chart simulation with bar chart
component_names = list(validation_report['detailed_results']['component_scores'].keys())
component_scores = list(validation_report['detailed_results']['component_scores'].values())
component_names_clean = [name.replace('_', ' ').title() for name in component_names]

bars = ax1.barh(component_names_clean, component_scores, 
               color=['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' 
                     for score in component_scores])
ax1.set_xlabel('Validation Score')
ax1.set_title('Component Validation Scores')
ax1.set_xlim(0, 1)

# Add score labels on bars
for bar, score in zip(bars, component_scores):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{score:.3f}', va='center', fontsize=9)

# Quality metrics pie chart
metrics_data = validation_report['data_quality_metrics']
metrics_labels = [label.replace('_score', '').replace('_', ' ').title() 
                 for label in metrics_data.keys()]
metrics_values = list(metrics_data.values())

ax2.pie(metrics_values, labels=metrics_labels, autopct='%1.1f%%', startangle=90)
ax2.set_title('Data Quality Metrics Distribution')

# Anomaly detection breakdown
anomaly_data = validation_report['anomaly_detection_summary']['anomaly_breakdown']
anomaly_labels = [label.replace('_', ' ').title() for label in anomaly_data.keys()]
anomaly_counts = list(anomaly_data.values())

bars = ax3.bar(range(len(anomaly_labels)), anomaly_counts, 
               color=['red' if count > 10 else 'orange' if count > 5 else 'green' 
                     for count in anomaly_counts])
ax3.set_xticks(range(len(anomaly_labels)))
ax3.set_xticklabels(anomaly_labels, rotation=45, ha='right')
ax3.set_ylabel('Count')
ax3.set_title('Anomaly Detection Breakdown')

# Add count labels on bars
for bar, count in zip(bars, anomaly_counts):
    if count > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=9)

# Overall validation status summary
status_data = {
    'Passed': validation_report['validation_summary']['components_passed'],
    'Failed': validation_report['validation_summary']['components_failed']
}

colors = ['green', 'red']
ax4.pie(status_data.values(), labels=status_data.keys(), autopct='%1.0f', 
        colors=colors, startangle=90)
ax4.set_title(f"Overall Validation Status\n({validation_report['completeness_percentage']:.1f}% Complete)")

plt.tight_layout()
plt.show()

print(f"\n🏁 VALIDATION PIPELINE COMPLETED")
print(f"Data Quality Framework Status: {'✅ OPERATIONAL' if validation_report['overall_quality_score'] > 0.7 else '⚠️  NEEDS ATTENTION'}")
print(f"100% Data Completeness Verification: {'✅ ACHIEVED' if validation_report['completeness_percentage'] == 100 else '❌ PARTIAL'}")

comprehensive_validation_report = validation_report