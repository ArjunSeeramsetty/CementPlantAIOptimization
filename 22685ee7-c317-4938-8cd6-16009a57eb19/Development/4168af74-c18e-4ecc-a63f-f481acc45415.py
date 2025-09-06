import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Anomaly Tracking Dashboard
def create_anomaly_dashboard(df, mass_energy_results=None, temporal_results=None):
    """Create comprehensive anomaly tracking dashboard"""
    
    dashboard_data = {
        'anomaly_summary': {},
        'statistical_anomalies': [],
        'isolation_forest_anomalies': [],
        'multivariate_anomalies': [],
        'dashboard_metrics': {},
        'visualization_data': {}
    }
    
    print("=== Building Anomaly Tracking Dashboard ===")
    
    # Get numerical columns for analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        # Statistical anomaly detection (Z-score method)
        z_scores = np.abs(stats.zscore(df[numerical_cols]))
        statistical_anomalies = np.where(z_scores > 3)  # 3-sigma rule
        
        anomaly_indices = list(set(statistical_anomalies[0]))
        dashboard_data['statistical_anomalies'] = [
            {
                'index': int(idx),
                'column': numerical_cols[statistical_anomalies[1][i]],
                'z_score': float(z_scores[statistical_anomalies[0][i], statistical_anomalies[1][i]]),
                'value': float(df.iloc[statistical_anomalies[0][i], statistical_anomalies[1][i]])
            } for i in range(len(statistical_anomalies[0]))
        ]
        
        print(f"Statistical anomalies detected: {len(anomaly_indices)}")
        
        # Isolation Forest anomaly detection
        if len(numerical_cols) >= 2:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(df[numerical_cols])
            
            isolation_anomalies = np.where(anomaly_scores == -1)[0]
            dashboard_data['isolation_forest_anomalies'] = [
                {
                    'index': int(idx),
                    'anomaly_score': float(iso_forest.decision_function(df[numerical_cols].iloc[[idx]])[0])
                } for idx in isolation_anomalies
            ]
            
            print(f"Isolation Forest anomalies: {len(isolation_anomalies)}")
        
        # Multivariate anomaly detection using Mahalanobis distance
        if len(numerical_cols) >= 3:
            data_matrix = df[numerical_cols].values
            mean_vec = np.mean(data_matrix, axis=0)
            cov_matrix = np.cov(data_matrix.T)
            
            try:
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                mahalanobis_dist = []
                
                for i in range(len(data_matrix)):
                    diff = data_matrix[i] - mean_vec
                    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahalanobis_dist.append(mahal_dist)
                
                mahalanobis_dist = np.array(mahalanobis_dist)
                threshold = np.percentile(mahalanobis_dist, 95)  # Top 5% as anomalies
                
                multivariate_anomalies = np.where(mahalanobis_dist > threshold)[0]
                dashboard_data['multivariate_anomalies'] = [
                    {
                        'index': int(idx),
                        'mahalanobis_distance': float(mahalanobis_dist[idx]),
                        'threshold': float(threshold)
                    } for idx in multivariate_anomalies
                ]
                
                print(f"Multivariate anomalies: {len(multivariate_anomalies)}")
                
            except np.linalg.LinAlgError:
                print("Covariance matrix is singular, skipping Mahalanobis distance calculation")
        
        # Combine all anomaly sources
        all_anomaly_indices = set()
        all_anomaly_indices.update(anomaly_indices)
        if 'isolation_forest_anomalies' in dashboard_data:
            all_anomaly_indices.update([a['index'] for a in dashboard_data['isolation_forest_anomalies']])
        if 'multivariate_anomalies' in dashboard_data:
            all_anomaly_indices.update([a['index'] for a in dashboard_data['multivariate_anomalies']])
        
        # Add physics-based and temporal anomalies if available
        physics_anomalies = set()
        temporal_anomalies = set()
        
        if mass_energy_results and 'mass_balance_violations' in mass_energy_results:
            physics_anomalies.update(mass_energy_results['mass_balance_violations'])
        if mass_energy_results and 'energy_balance_violations' in mass_energy_results:
            physics_anomalies.update(mass_energy_results['energy_balance_violations'])
            
        if temporal_results and 'rate_of_change_violations' in temporal_results:
            temporal_anomalies.update([v['index'] for v in temporal_results['rate_of_change_violations']])
        if temporal_results and 'trend_violations' in temporal_results:
            temporal_anomalies.update([v['index'] for v in temporal_results['trend_violations']])
        
        all_anomaly_indices.update(physics_anomalies)
        all_anomaly_indices.update(temporal_anomalies)
        
        # Dashboard metrics
        dashboard_data['anomaly_summary'] = {
            'total_records': len(df),
            'total_anomalies': len(all_anomaly_indices),
            'anomaly_percentage': float(len(all_anomaly_indices) / len(df) * 100),
            'statistical_anomalies': len(anomaly_indices),
            'isolation_forest_anomalies': len(dashboard_data.get('isolation_forest_anomalies', [])),
            'multivariate_anomalies': len(dashboard_data.get('multivariate_anomalies', [])),
            'physics_based_anomalies': len(physics_anomalies),
            'temporal_anomalies': len(temporal_anomalies)
        }
        
        # Calculate data quality score
        base_quality = 1.0 - (len(all_anomaly_indices) / len(df))
        consistency_penalty = 0.1 * (len(physics_anomalies) / max(1, len(df)))
        temporal_penalty = 0.05 * (len(temporal_anomalies) / max(1, len(df)))
        
        quality_score = max(0.0, base_quality - consistency_penalty - temporal_penalty)
        
        dashboard_data['dashboard_metrics'] = {
            'overall_quality_score': float(quality_score),
            'data_consistency_score': float(1.0 - (len(physics_anomalies) / max(1, len(df)))),
            'temporal_consistency_score': float(1.0 - (len(temporal_anomalies) / max(1, len(df)))),
            'statistical_consistency_score': float(1.0 - (len(anomaly_indices) / max(1, len(df)))),
            'improvement_target': float(quality_score * 1.15)  # 15% improvement target
        }
        
        # Create visualization data for plotting
        if len(numerical_cols) >= 2:
            # PCA for dimensionality reduction and visualization
            pca = PCA(n_components=min(2, len(numerical_cols)))
            pca_data = pca.fit_transform(StandardScaler().fit_transform(df[numerical_cols]))
            
            # Mark anomalies in PCA space
            anomaly_mask = [i in all_anomaly_indices for i in range(len(df))]
            
            dashboard_data['visualization_data'] = {
                'pca_coordinates': pca_data.tolist(),
                'anomaly_mask': anomaly_mask,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'feature_contribution': {
                    f'PC1': pca.components_[0].tolist(),
                    f'PC2': pca.components_[1].tolist() if pca.n_components_ > 1 else []
                }
            }
    
    return dashboard_data

# Apply to available data
print("Creating anomaly tracking dashboard...")

dashboard_results = None
if 'X_final' in globals():
    # Get the physics and temporal results from previous blocks
    physics_data = mass_energy_results if 'mass_energy_results' in globals() else None
    temporal_data = temporal_results if 'temporal_results' in globals() else None
    
    dashboard_results = create_anomaly_dashboard(X_final, physics_data, temporal_data)
    
    print(f"\n=== Dashboard Summary ===")
    print(f"Total anomalies: {dashboard_results['anomaly_summary']['total_anomalies']}")
    print(f"Anomaly percentage: {dashboard_results['anomaly_summary']['anomaly_percentage']:.2f}%")
    print(f"Overall quality score: {dashboard_results['dashboard_metrics']['overall_quality_score']:.4f}")
    print(f"Improvement target: {dashboard_results['dashboard_metrics']['improvement_target']:.4f}")
    
else:
    print("No X_final data available for dashboard creation")
    dashboard_results = {'status': 'no_data_available'}