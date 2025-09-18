import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from cement_ai_platform.validation.drift_detection import DataDriftDetector

def launch_data_validation_demo():
    """Launch data validation and drift detection demo"""
    
    st.set_page_config(
        page_title="🧪 Data Validation & Drift Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Data Validation & Drift Detection")
    st.markdown("**Automated Data Quality Monitoring & Model Retraining**")
    
    # Initialize drift detector
    if 'drift_detector' not in st.session_state:
        st.session_state.drift_detector = DataDriftDetector()
    
    drift_detector = st.session_state.drift_detector
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Validation Settings")
        
        validation_frequency = st.selectbox("Validation Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
        drift_sensitivity = st.slider("Drift Sensitivity", 0.01, 0.10, 0.05)
        retraining_threshold = st.selectbox("Auto-Retraining Threshold", ["High", "Critical"])
        
        st.header("📊 Data Sources")
        
        st.checkbox("Process Variables", value=True)
        st.checkbox("Quality Parameters", value=True)
        st.checkbox("Energy Consumption", value=True)
        st.checkbox("Emissions Data", value=True)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Drift Analysis", "🔍 Data Quality", "🔄 Retraining Pipeline", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("📊 Data Drift Analysis")
        
        # Generate synthetic current and reference data for demo
        if st.button("🔄 Run Drift Analysis"):
            
            with st.spinner("Analyzing data drift..."):
                
                # Generate synthetic data
                np.random.seed(42)
                
                # Reference data (baseline)
                ref_data = pd.DataFrame({
                    'free_lime_percent': np.random.normal(1.2, 0.3, 1000),
                    'thermal_energy_kcal_kg': np.random.normal(690, 25, 1000),
                    'feed_rate_tph': np.random.normal(167, 15, 1000),
                    'burning_zone_temp_c': np.random.normal(1450, 20, 1000),
                    'nox_mg_nm3': np.random.normal(500, 50, 1000)
                })
                
                # Create reference snapshot
                drift_detector.create_reference_snapshot(ref_data, "baseline")
                
                # Current data with simulated drift
                current_data = pd.DataFrame({
                    'free_lime_percent': np.random.normal(1.5, 0.4, 500),  # Mean shift + variance increase
                    'thermal_energy_kcal_kg': np.random.normal(710, 30, 500),  # Energy increase
                    'feed_rate_tph': np.random.normal(165, 18, 500),  # Slight decrease
                    'burning_zone_temp_c': np.random.normal(1445, 25, 500),  # Temperature decrease
                    'nox_mg_nm3': np.random.normal(520, 60, 500)  # Emissions increase
                })
                
                # Run drift detection
                drift_results = drift_detector.detect_data_drift(current_data, "baseline")
                
                # Store results in session state
                st.session_state.drift_results = drift_results
        
        # Display drift results
        if 'drift_results' in st.session_state:
            results = st.session_state.drift_results
            
            # Overall status
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                drift_status = "🚨 DRIFT DETECTED" if results['drift_detected'] else "✅ NO DRIFT"
                st.metric("Drift Status", drift_status)
            
            with col2:
                summary = results['drift_summary']
                st.metric("Variables with Drift", f"{summary['variables_with_drift']}/{summary['total_variables_analyzed']}")
            
            with col3:
                st.metric("Max Severity", summary['max_severity'])
            
            with col4:
                st.metric("Categories Affected", summary['categories_with_drift'])
            
            # Detailed drift analysis
            st.subheader("🔍 Detailed Drift Analysis")
            
            for category, category_results in results['drift_results'].items():
                if category_results['variables']:
                    
                    with st.expander(f"📋 {category.title()} Variables", 
                                   expanded=category_results['category_drift_detected']):
                        
                        for var, var_results in category_results['variables'].items():
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Variable drift metrics
                                drift_detected = "🚨 YES" if var_results['drift_detected'] else "✅ NO"
                                
                                st.markdown(f"""
                                **{var.replace('_', ' ').title()}**
                                - **Drift Detected:** {drift_detected}
                                - **Severity:** {var_results['severity']}
                                - **Mean Shift:** {var_results['mean_shift']:.3f}
                                - **KS Test p-value:** {var_results['ks_pvalue']:.4f}
                                - **Outlier Rate:** {var_results['outlier_rate']:.2%}
                                """)
                            
                            with col2:
                                # Drift visualization
                                if var_results['drift_detected']:
                                    
                                    # Create simple drift indicator chart
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=['Current', 'Reference'],
                                        y=[var_results['current_mean'], var_results['reference_mean']],
                                        name='Mean Value',
                                        marker_color=['red', 'blue']
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{var} Mean Comparison",
                                        height=250
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if results['drift_detected']:
                st.subheader("💡 Recommendations")
                
                for recommendation in summary['recommendations']:
                    st.markdown(f"• {recommendation}")
                
                # Retraining decision
                if summary['max_severity'] in ["High", "Critical"]:
                    st.warning("⚠️ **Model retraining recommended due to significant data drift**")
                    
                    if st.button("🔄 Trigger Model Retraining"):
                        retraining_result = drift_detector.trigger_model_retraining(summary)
                        
                        if retraining_result['retraining_triggered']:
                            st.success(f"✅ Model retraining pipeline triggered: {retraining_result['retraining_config']['pipeline_id']}")
                            st.info(f"Estimated completion: {retraining_result['estimated_completion'].strftime('%Y-%m-%d %H:%M')}")
                        else:
                            st.info(f"ℹ️ {retraining_result['reason']}")
    
    with tab2:
        st.subheader("🔍 Data Quality Assessment")
        
        # Data quality metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Completeness", "98.5%", delta="+0.3%")
        with col2:
            st.metric("Data Accuracy", "96.2%", delta="-0.8%")
        with col3:
            st.metric("Data Consistency", "99.1%", delta="+0.1%")
        
        # Quality issues breakdown
        st.subheader("📋 Quality Issues")
        
        quality_issues = [
            {"Issue": "Missing Values", "Count": 12, "Severity": "Low", "Affected Variables": "temperature_sensors"},
            {"Issue": "Outliers Detected", "Count": 8, "Severity": "Medium", "Affected Variables": "free_lime_percent"},
            {"Issue": "Range Violations", "Count": 3, "Severity": "High", "Affected Variables": "fuel_rate_tph"},
            {"Issue": "Duplicate Records", "Count": 1, "Severity": "Low", "Affected Variables": "timestamp"}
        ]
        
        quality_df = pd.DataFrame(quality_issues)
        st.dataframe(quality_df, use_container_width=True)
        
        # Quality trend chart
        st.subheader("📈 Data Quality Trends")
        
        # Generate quality trend data
        dates = pd.date_range('2025-09-01', '2025-09-18', freq='D')
        quality_scores = np.random.uniform(95, 99, len(dates))
        
        fig = px.line(x=dates, y=quality_scores, title="Data Quality Score Over Time")
        fig.update_layout(xaxis_title="Date", yaxis_title="Quality Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔄 Model Retraining Pipeline")
        
        # Pipeline status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pipeline Status", "🟢 Active")
        with col2:
            st.metric("Last Retraining", "2025-09-15")
        with col3:
            st.metric("Next Scheduled", "2025-09-22")
        
        # Retraining history
        st.subheader("📋 Retraining History")
        
        retraining_history = [
            {"Date": "2025-09-15", "Trigger": "Scheduled", "Duration": "3.2 hours", "Status": "✅ Success"},
            {"Date": "2025-09-10", "Trigger": "Data Drift", "Duration": "2.8 hours", "Status": "✅ Success"},
            {"Date": "2025-09-05", "Trigger": "Performance Drop", "Duration": "4.1 hours", "Status": "✅ Success"},
            {"Date": "2025-08-30", "Trigger": "Scheduled", "Duration": "3.5 hours", "Status": "✅ Success"}
        ]
        
        history_df = pd.DataFrame(retraining_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Model performance comparison
        st.subheader("📊 Model Performance Comparison")
        
        models = ['Current Model', 'Previous Model', 'Baseline Model']
        accuracy_scores = [94.2, 92.8, 89.5]
        
        fig = px.bar(x=models, y=accuracy_scores, title="Model Accuracy Comparison")
        fig.update_layout(yaxis_title="Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚙️ Drift Detection Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Statistical Thresholds**")
            
            ks_threshold = st.slider("KS Test p-value threshold", 0.01, 0.10, 0.05)
            mean_shift_threshold = st.slider("Mean shift threshold", 0.10, 0.30, 0.15)
            std_shift_threshold = st.slider("Std deviation shift threshold", 0.15, 0.35, 0.20)
            outlier_threshold = st.slider("Outlier rate threshold", 0.05, 0.20, 0.10)
            
            st.markdown("**🔄 Retraining Settings**")
            
            auto_retrain = st.checkbox("Enable automatic retraining", value=True)
            retrain_threshold = st.selectbox("Retraining severity threshold", ["High", "Critical"])
            max_retrain_freq = st.selectbox("Max retraining frequency", ["Daily", "Weekly", "Monthly"])
        
        with col2:
            st.markdown("**📧 Alert Configuration**")
            
            email_alerts = st.checkbox("Email alerts", value=True)
            slack_alerts = st.checkbox("Slack notifications", value=False)
            dashboard_alerts = st.checkbox("Dashboard alerts", value=True)
            
            st.text_input("Email recipients", "ops-team@jkcement.com")
            st.text_input("Slack channel", "#data-quality")
            
            st.markdown("**📈 Monitoring Integration**")
            
            cloud_monitoring = st.checkbox("Google Cloud Monitoring", value=True)
            custom_metrics = st.checkbox("Custom metrics dashboard", value=True)
            prometheus_export = st.checkbox("Prometheus metrics export", value=False)
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully!")

if __name__ == "__main__":
    launch_data_validation_demo()
