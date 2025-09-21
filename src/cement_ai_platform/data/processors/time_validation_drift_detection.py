import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """Advanced data drift detection for temporal validation"""

    def __init__(self, reference_window_size=0.3, significance_level=0.05):
        self.reference_window_size = reference_window_size
        self.significance_level = significance_level
        self.drift_threshold = 0.05  # JS divergence threshold

    def detect_numerical_drift(self, reference_data, test_data, feature_name):
        """Detect drift in numerical features using KS test"""
        # KS test for distribution shift
        statistic, p_value = ks_2samp(reference_data, test_data)

        # Jensen-Shannon divergence
        ref_hist, ref_bins = np.histogram(reference_data, bins=20, density=True)
        test_hist, _ = np.histogram(test_data, bins=ref_bins, density=True)

        # Normalize to ensure they sum to 1
        ref_hist = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else ref_hist
        test_hist = test_hist / np.sum(test_hist) if np.sum(test_hist) > 0 else test_hist

        # Handle zero-sum cases for JS divergence
        if np.sum(ref_hist) == 0 or np.sum(test_hist) == 0:
            js_divergence = 1.0  # Maximum divergence
        else:
            js_divergence = jensenshannon(ref_hist, test_hist, base=2)

        drift_detected = (p_value < self.significance_level) or (js_divergence > self.drift_threshold)

        return {
            'feature': feature_name,
            'drift_detected': drift_detected,
            'ks_statistic': statistic,
            'ks_p_value': p_value,
            'js_divergence': js_divergence,
            'method': 'numerical'
        }

    def detect_categorical_drift(self, reference_data, test_data, feature_name):
        """Detect drift in categorical features using Chi-square test"""
        ref_counts = pd.Series(reference_data).value_counts()
        test_counts = pd.Series(test_data).value_counts()

        # Align categories
        all_categories = set(ref_counts.index) | set(test_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        test_aligned = [test_counts.get(cat, 0) for cat in all_categories]

        # Chi-square test
        try:
            if len(all_categories) > 1 and sum(ref_aligned) > 0 and sum(test_aligned) > 0:
                chi2_stat, p_value, dof, expected = chi2_contingency([ref_aligned, test_aligned])
                drift_detected = p_value < self.significance_level
            else:
                chi2_stat, p_value = 0, 1
                drift_detected = False
        except:
            # Fallback if chi-square fails
            chi2_stat, p_value = 0, 1
            drift_detected = False

        return {
            'feature': feature_name,
            'drift_detected': drift_detected,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': p_value,
            'method': 'categorical'
        }

    def detect_temporal_drift(self, temporal_data, feature_cols, time_windows=5):
        """Detect drift across multiple time windows"""
        n_samples = len(temporal_data)
        window_size = n_samples // time_windows

        drift_results = []

        # Use first window as reference
        reference_window = temporal_data.iloc[:window_size]

        for window_idx in range(1, time_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, n_samples)
            test_window = temporal_data.iloc[start_idx:end_idx]

            window_results = {
                'window': window_idx + 1,
                'time_period': f"{test_window['timestamp'].min().strftime('%Y-%m-%d')} to {test_window['timestamp'].max().strftime('%Y-%m-%d')}",
                'drift_features': []
            }

            for feature in feature_cols:
                if feature in ['timestamp', 'target']:
                    continue

                ref_values = reference_window[feature].values
                test_values = test_window[feature].values

                # Check if numerical or categorical
                if temporal_data[feature].dtype in ['int64', 'float64']:
                    if feature.endswith('_encoded'):
                        # Treat encoded categoricals as categorical
                        drift_result = self.detect_categorical_drift(ref_values, test_values, feature)
                    else:
                        drift_result = self.detect_numerical_drift(ref_values, test_values, feature)
                else:
                    drift_result = self.detect_categorical_drift(ref_values, test_values, feature)

                if drift_result['drift_detected']:
                    window_results['drift_features'].append(drift_result)

            drift_results.append(window_results)

        return drift_results

# Initialize drift detector
drift_detector = DriftDetector(significance_level=0.05)

# Feature columns for drift detection
drift_feature_cols = [col for col in temporal_df.columns if col not in ['timestamp', 'target']]

# Detect temporal drift
temporal_drift_results = drift_detector.detect_temporal_drift(temporal_df, drift_feature_cols, time_windows=5)

print("Temporal Drift Detection Results:")
print("=" * 60)

drift_summary = {
    'total_windows_analyzed': len(temporal_drift_results),
    'windows_with_drift': 0,
    'features_with_drift': set(),
    'most_affected_features': {}
}

for window_result in temporal_drift_results:
    print(f"\nWindow {window_result['window']} ({window_result['time_period']}):")

    if window_result['drift_features']:
        drift_summary['windows_with_drift'] += 1
        print(f"  Drift detected in {len(window_result['drift_features'])} features:")

        for drift_info in window_result['drift_features']:
            feature_name = drift_info['feature']
            drift_summary['features_with_drift'].add(feature_name)

            if feature_name not in drift_summary['most_affected_features']:
                drift_summary['most_affected_features'][feature_name] = 0
            drift_summary['most_affected_features'][feature_name] += 1

            print(f"    - {feature_name}: ", end="")
            if drift_info['method'] == 'numerical':
                print(f"JS divergence = {drift_info['js_divergence']:.4f}, KS p-value = {drift_info['ks_p_value']:.4f}")
            else:
                print(f"Chi2 p-value = {drift_info['chi2_p_value']:.4f}")
    else:
        print("  No drift detected")

print(f"\n\nDrift Detection Summary:")
print(f"Windows with drift: {drift_summary['windows_with_drift']}/{drift_summary['total_windows_analyzed']}")
print(f"Features affected by drift: {len(drift_summary['features_with_drift'])}")

if drift_summary['most_affected_features']:
    print("Most frequently drifting features:")
    for feature, count in sorted(drift_summary['most_affected_features'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {count} windows")