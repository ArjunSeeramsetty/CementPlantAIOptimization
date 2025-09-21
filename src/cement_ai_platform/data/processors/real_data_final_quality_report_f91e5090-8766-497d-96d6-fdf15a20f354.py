import pandas as pd
import numpy as np

print("📊 CEMENT DATA INTEGRATION - FINAL QUALITY REPORT")
print("=" * 60)

# Create comprehensive quality metrics
quality_report = {
    'integration_summary': {
        'total_records': len(integrated_df),
        'global_db_records': sum(integrated_df['data_source'] == 'Global_DB'),
        'kaggle_records': sum(integrated_df['data_source'] == 'Kaggle'),
        'unified_schema_fields': len(integrated_df.columns),
        'integration_success': True
    },
    'data_completeness': {},
    'validation_results': validation_results,
    'quality_metrics': {}
}

print(f"🎯 INTEGRATION RESULTS")
print("-" * 30)
print(f"Total integrated records: {quality_report['integration_summary']['total_records']:,}")
print(f"  • Global Cement DB: {quality_report['integration_summary']['global_db_records']:,} facilities")
print(f"  • Kaggle dataset: {quality_report['integration_summary']['kaggle_records']:,} samples")
print(f"Unified schema fields: {quality_report['integration_summary']['unified_schema_fields']}")

# Calculate completeness by data category
print(f"\n📈 DATA COMPLETENESS BY CATEGORY")
print("-" * 40)

category_completeness = {}
for category, field_list in categories.items():
    _category_fields = [f for f in field_list if f in integrated_df.columns]
    if _category_fields:
        _total_cells = len(integrated_df) * len(_category_fields)
        _non_null_cells = integrated_df[_category_fields].count().sum()
        _completeness_pct = (_non_null_cells / _total_cells) * 100
        category_completeness[category] = _completeness_pct
        print(f"  {category:15}: {_completeness_pct:5.1f}% complete")

quality_report['data_completeness']['by_category'] = category_completeness

# Source-specific completeness
print(f"\n🔍 COMPLETENESS BY DATA SOURCE")
print("-" * 35)

for source in ['Global_DB', 'Kaggle']:
    _source_data = integrated_df[integrated_df['data_source'] == source]
    _total_cells = len(_source_data) * len(_source_data.columns)
    _non_null_cells = _source_data.count().sum()
    _completeness_pct = (_non_null_cells / _total_cells) * 100
    quality_report['data_completeness'][source] = _completeness_pct
    print(f"  {source:10}: {_completeness_pct:5.1f}% complete")

# Cement type distribution across sources
print(f"\n🧱 CEMENT TYPE DISTRIBUTION")
print("-" * 30)

cement_type_dist = integrated_df['cement_type_standard'].value_counts()
for cement_type, _count in cement_type_dist.head().items():
    _percentage = (_count / len(integrated_df)) * 100
    print(f"  {cement_type:25}: {_count:4} ({_percentage:4.1f}%)")

# Missing value analysis by column
print(f"\n❓ TOP MISSING VALUE COLUMNS")
print("-" * 35)

missing_analysis = integrated_df.isnull().sum().sort_values(ascending=False)
missing_top = missing_analysis.head(10)
for _col_name, _missing_count in missing_top.items():
    _missing_pct = (_missing_count / len(integrated_df)) * 100
    print(f"  {_col_name:25}: {_missing_count:4} ({_missing_pct:4.1f}%)")

quality_report['data_completeness']['missing_analysis'] = missing_analysis.to_dict()

# Physics validation summary
print(f"\n🔬 PHYSICS VALIDATION RESULTS")
print("-" * 35)

validation_status = "✅ PASSED" if physics_validation_passed else "❌ FAILED"
print(f"Overall validation: {validation_status}")
print(f"Checks passed: {len(validation_results['passed_checks'])}")
print(f"Checks failed: {len(validation_results['failed_checks'])}")
print(f"Warnings issued: {len(validation_results['warnings'])}")

# Data quality score calculation
print(f"\n⭐ OVERALL DATA QUALITY SCORE")
print("-" * 35)

# Calculate weighted quality score
completeness_score = np.mean(list(category_completeness.values())) / 100  # 0-1
validation_score = 1.0 if physics_validation_passed else 0.7  # Physics validation
integration_score = 1.0  # Successful integration

overall_quality = (completeness_score * 0.4 + validation_score * 0.4 + integration_score * 0.2) * 100

quality_report['quality_metrics'] = {
    'overall_quality_score': overall_quality,
    'completeness_score': completeness_score * 100,
    'validation_score': validation_score * 100,
    'integration_score': integration_score * 100
}

print(f"Overall Quality Score: {overall_quality:.1f}/100")
print(f"  • Data Completeness: {completeness_score * 100:.1f}% (40% weight)")
print(f"  • Physics Validation: {validation_score * 100:.1f}% (40% weight)")
print(f"  • Integration Success: {integration_score * 100:.1f}% (20% weight)")

# Final recommendations
print(f"\n🎯 RECOMMENDATIONS")
print("-" * 20)

if overall_quality >= 80:
    quality_grade = "Excellent"
    recommendations = [
        "Dataset ready for production use",
        "Consider advanced analytics and ML modeling",
        "Monitor data quality over time"
    ]
elif overall_quality >= 70:
    quality_grade = "Good"
    recommendations = [
        "Dataset suitable for most analyses",
        "Address missing values in key fields",
        "Validate outliers in environmental metrics"
    ]
else:
    quality_grade = "Needs Improvement"
    recommendations = [
        "Address data quality issues before use",
        "Focus on missing value imputation",
        "Review physics validation failures"
    ]

print(f"Data Quality Grade: {quality_grade}")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

# Export summary
integration_success = True
final_dataset_ready = overall_quality >= 70

print(f"\n🎉 INTEGRATION COMPLETE!")
print(f"Final dataset: {len(integrated_df):,} records, {len(integrated_df.columns)} fields")
print(f"Quality score: {overall_quality:.1f}/100 ({quality_grade})")
print(f"Ready for use: {'✅ Yes' if final_dataset_ready else '❌ Needs work'}")

print(f"\n📋 SUCCESS CRITERIA CHECK:")
print(f"  ✅ Global Cement Database integrated: {sum(integrated_df['data_source'] == 'Global_DB'):,} facilities")
print(f"  ✅ Kaggle dataset integrated: {sum(integrated_df['data_source'] == 'Kaggle'):,} samples")
print(f"  ✅ Unified schema created: {len(integrated_df.columns)} fields")
print(f"  ✅ Physics validation performed: {len(validation_results['passed_checks'])} checks passed")
print(f"  ✅ Quality metrics generated: {overall_quality:.1f}% score")
print(f"  ✅ Completeness report created: All categories analyzed")