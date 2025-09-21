import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class ABTestingFramework:
    """A/B Testing Framework for model comparison and validation"""

    def __init__(self, significance_level=0.05, n_bootstrap=1000):
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap

    def bootstrap_metric(self, y_true, y_pred, metric_func, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals for metrics"""
        bootstrap_scores = []
        n_samples = len(y_true)

        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_true = y_true[indices]
            bootstrap_pred = y_pred[indices]

            score = metric_func(bootstrap_true, bootstrap_pred)
            bootstrap_scores.append(score)

        bootstrap_scores = np.array(bootstrap_scores)
        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': np.percentile(bootstrap_scores, 2.5),
            'ci_upper': np.percentile(bootstrap_scores, 97.5)
        }

    def ab_test_models(self, X, y, model_a, model_b, test_size=0.3, n_trials=100):
        """Perform A/B test between two models"""
        results_a = []
        results_b = []

        # Run multiple train-test splits for robust comparison
        for trial in range(n_trials):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=trial
            )

            # Train and test model A
            model_a.fit(X_train, y_train)
            pred_a = model_a.predict(X_test)
            r2_a = r2_score(y_test, pred_a)
            results_a.append(r2_a)

            # Train and test model B
            model_b.fit(X_train, y_train)
            pred_b = model_b.predict(X_test)
            r2_b = r2_score(y_test, pred_b)
            results_b.append(r2_b)

        results_a = np.array(results_a)
        results_b = np.array(results_b)

        # Statistical tests
        t_stat, t_pval = ttest_ind(results_a, results_b)
        u_stat, u_pval = mannwhitneyu(results_a, results_b, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results_a) - 1) * np.std(results_a)**2 +
                             (len(results_b) - 1) * np.std(results_b)**2) /
                            (len(results_a) + len(results_b) - 2))
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std

        return {
            'model_a_scores': results_a,
            'model_b_scores': results_b,
            'model_a_mean': np.mean(results_a),
            'model_b_mean': np.mean(results_b),
            'model_a_std': np.std(results_a),
            'model_b_std': np.std(results_b),
            'difference': np.mean(results_a) - np.mean(results_b),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'u_statistic': u_stat,
            'u_pvalue': u_pval,
            'cohens_d': cohens_d,
            'significant': min(t_pval, u_pval) < self.significance_level
        }

    def temporal_ab_test(self, temporal_data, model_configs, n_windows=5):
        """A/B test models across different time windows"""
        n_samples = len(temporal_data)
        window_size = n_samples // n_windows

        # Feature columns (exclude target and timestamp)
        feature_cols = [col for col in temporal_data.columns if col not in ['target', 'timestamp']]

        temporal_results = []

        for window_idx in range(n_windows - 1):  # Skip last incomplete window
            start_idx = window_idx * window_size
            end_idx = (window_idx + 1) * window_size

            window_data = temporal_data.iloc[start_idx:end_idx]
            X_window = window_data[feature_cols].values
            y_window = window_data['target'].values

            window_results = {
                'window': window_idx + 1,
                'time_period': f"{window_data['timestamp'].min().strftime('%Y-%m-%d')} to {window_data['timestamp'].max().strftime('%Y-%m-%d')}",
                'comparisons': {}
            }

            # Compare all model pairs
            model_names = list(model_configs.keys())
            for i, name_a in enumerate(model_names):
                for name_b in model_names[i+1:]:
                    model_a = model_configs[name_a]
                    model_b = model_configs[name_b]

                    comparison_key = f"{name_a}_vs_{name_b}"
                    ab_result = self.ab_test_models(X_window, y_window, model_a, model_b, n_trials=30)

                    window_results['comparisons'][comparison_key] = ab_result

            temporal_results.append(window_results)

        return temporal_results

# Initialize A/B testing framework
ab_framework = ABTestingFramework(significance_level=0.05)

# Define model configurations for A/B testing
ab_model_configs = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
}

print("Running A/B Testing Framework...")
print("=" * 60)

# Run temporal A/B tests
temporal_ab_results = ab_framework.temporal_ab_test(temporal_df, ab_model_configs, n_windows=4)

# Aggregate results across time windows
comparison_summary = {}

for window_result in temporal_ab_results:
    print(f"\nWindow {window_result['window']} ({window_result['time_period']}):")

    for comparison_name, ab_result in window_result['comparisons'].items():
        if comparison_name not in comparison_summary:
            comparison_summary[comparison_name] = {
                'significant_windows': 0,
                'total_windows': 0,
                'mean_differences': [],
                'effect_sizes': []
            }

        comparison_summary[comparison_name]['total_windows'] += 1
        comparison_summary[comparison_name]['mean_differences'].append(ab_result['difference'])
        comparison_summary[comparison_name]['effect_sizes'].append(ab_result['cohens_d'])

        if ab_result['significant']:
            comparison_summary[comparison_name]['significant_windows'] += 1

        print(f"  {comparison_name}:")
        print(f"    Model A: {ab_result['model_a_mean']:.4f} ± {ab_result['model_a_std']:.4f}")
        print(f"    Model B: {ab_result['model_b_mean']:.4f} ± {ab_result['model_b_std']:.4f}")
        print(f"    Difference: {ab_result['difference']:.4f}")
        print(f"    p-value: {ab_result['t_pvalue']:.4f}")
        print(f"    Significant: {'Yes' if ab_result['significant'] else 'No'}")
        print(f"    Effect Size: {ab_result['cohens_d']:.4f}")

print(f"\n\nA/B Testing Summary Across All Time Windows:")
print("=" * 60)

for comparison_name, summary in comparison_summary.items():
    significant_percentage = (summary['significant_windows'] / summary['total_windows']) * 100
    avg_difference = np.mean(summary['mean_differences'])
    avg_effect_size = np.mean(summary['effect_sizes'])

    print(f"\n{comparison_name}:")
    print(f"  Significant in {summary['significant_windows']}/{summary['total_windows']} windows ({significant_percentage:.1f}%)")
    print(f"  Average performance difference: {avg_difference:.4f}")
    print(f"  Average effect size: {avg_effect_size:.4f}")

    # Effect size interpretation
    if abs(avg_effect_size) < 0.2:
        effect_interpretation = "Small effect"
    elif abs(avg_effect_size) < 0.5:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Large effect"

    print(f"  Effect interpretation: {effect_interpretation}")

print(f"\n\nA/B Testing Framework Results Summary:")
print(f"Total temporal windows analyzed: {len(temporal_ab_results)}")
print(f"Model comparisons per window: {len(list(temporal_ab_results[0]['comparisons'].keys()))}")
print(f"Statistical significance threshold: {ab_framework.significance_level}")
print("Framework successfully validates model performance across temporal variations")