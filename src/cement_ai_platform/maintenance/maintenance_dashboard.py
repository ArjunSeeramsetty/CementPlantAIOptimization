import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from cement_ai_platform.maintenance.predictive_maintenance import PredictiveMaintenanceEngine

def launch_predictive_maintenance_demo():
    """Launch predictive maintenance demo interface"""
    
    st.set_page_config(
        page_title="🔧 Predictive Maintenance System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Predictive Maintenance System")
    st.markdown("**AI-Powered Equipment Health Monitoring & Failure Prediction**")
    
    # Initialize system
    if 'pm_engine' not in st.session_state:
        st.session_state.pm_engine = PredictiveMaintenanceEngine()
    
    pm_engine = st.session_state.pm_engine
    
    # Sidebar
    with st.sidebar:
        st.header("🏭 Plant Selection")
        plant_id = st.selectbox("Select Plant", ["JK_Rajasthan_1", "JK_MP_1", "UltraTech_Gujarat_1"])
        
        st.header("⚙️ Analysis Settings")
        prediction_horizon = st.slider("Prediction Horizon (days)", 7, 90, 30)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "📋 Maintenance Report", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("📊 Maintenance Overview")
        
        # Generate maintenance report
        report = pm_engine.generate_maintenance_report(plant_id, prediction_horizon)
        summary = report['summary']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Equipment", "5", delta="Active")
        with col2:
            st.metric("Pending Actions", summary['total_recommendations'])
        with col3:
            st.metric("Critical Alerts", summary['critical_count'])
        with col4:
            st.metric("Estimated Cost", f"${summary['total_estimated_cost']:,.0f}")
        
        # Priority breakdown chart
        st.subheader("🚨 Priority Breakdown")
        
        priority_data = {
            'Priority': ['Critical', 'High', 'Medium', 'Low'],
            'Count': [summary['critical_count'], summary['high_count'], 
                     summary['medium_count'], summary['low_count']]
        }
        
        priority_df = pd.DataFrame(priority_data)
        
        fig = px.bar(priority_df, x='Priority', y='Count', 
                    title="Maintenance Actions by Priority",
                    color='Priority',
                    color_discrete_map={
                        'Critical': '#F44336',
                        'High': '#FF9800', 
                        'Medium': '#2196F3',
                        'Low': '#4CAF50'
                    })
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Equipment Failure Predictions")
        
        # Get recommendations from report
        recommendations = report['recommendations']
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                
                # Priority color
                priority_colors = {
                    'Critical': '#F44336',
                    'High': '#FF9800',
                    'Medium': '#2196F3', 
                    'Low': '#4CAF50'
                }
                
                color = priority_colors.get(rec.priority, '#666')
                
                with st.expander(f"{rec.equipment_name} - {rec.priority} Priority", 
                               expanded=rec.priority in ['Critical', 'High']):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Equipment:** {rec.equipment_name}  
                        **Failure Probability:** {rec.failure_probability:.1%}  
                        **Time to Failure:** {rec.time_to_failure_hours:.0f} hours ({rec.time_to_failure_hours/24:.1f} days)  
                        **Predicted Failure Date:** {rec.predicted_failure_date.strftime('%Y-%m-%d %H:%M')}  
                        **Confidence:** {rec.confidence:.1%}  
                        **Estimated Cost:** ${rec.estimated_cost:,.0f}  
                        
                        **Impact:** {rec.impact_description}
                        """)
                        
                        st.markdown("**🔧 Recommended Actions:**")
                        for action in rec.recommended_actions:
                            st.markdown(f"• {action}")
                    
                    with col2:
                        # Failure probability gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = rec.failure_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Failure Risk %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgray"},
                                    {'range': [25, 50], 'color': "gray"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Action buttons
                        if st.button(f"📋 Create Work Order", key=f"wo_{i}"):
                            st.success(f"✅ Work Order Created: WO-{rec.equipment_id}-{datetime.now().strftime('%Y%m%d')}")
                        
                        if st.button(f"📅 Schedule Maintenance", key=f"schedule_{i}"):
                            st.info("🗓️ Maintenance scheduled successfully")
        else:
            st.info("✅ No maintenance actions required in the selected time period")
    
    with tab3:
        st.subheader("📋 Comprehensive Maintenance Report")
        
        # Report generation controls
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox("Report Type", [
                "Equipment Health Summary",
                "Failure Risk Assessment", 
                "Maintenance Schedule",
                "Cost Analysis"
            ])
        
        with col2:
            export_format = st.selectbox("Export Format", ["PDF", "Excel", "CSV"])
        
        if st.button("📊 Generate Report"):
            
            with st.spinner("Generating comprehensive maintenance report..."):
                import time
                time.sleep(2)
                
                # Display report summary
                st.success("✅ Report generated successfully!")
                
                # Report content based on type
                if report_type == "Equipment Health Summary":
                    st.markdown(f"""
                    ## 🏥 Equipment Health Summary - {plant_id}
                    
                    **Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    **Plant:** {plant_id}
                    **Period:** Next {prediction_horizon} days
                    
                    ### 📊 Overall Health Status
                    
                    | Equipment | Health Score | Status | Next Maintenance |
                    |-----------|--------------|--------|------------------|
                    | Kiln #1 | 87% | Good | 45 days |
                    | Raw Mill #1 | 72% | Fair | 15 days |
                    | Cement Mill #1 | 91% | Excellent | 60 days |
                    | ID Fan #1 | 68% | Fair | 10 days |
                    | Cooler #1 | 89% | Good | 30 days |
                    
                    ### 🚨 Critical Findings
                    
                    1. **ID Fan #1**: Bearing vibration trending upward
                    2. **Raw Mill #1**: Oil contamination above normal levels
                    3. **All Equipment**: Within acceptable operational parameters
                    
                    ### 💰 Financial Impact
                    
                    - **Preventive Maintenance Cost**: ${summary['total_estimated_cost']:,.0f}
                    - **Potential Failure Cost Avoidance**: ${summary['total_estimated_cost'] * 3:,.0f}
                    - **ROI**: 300%
                    """)
                
                elif report_type == "Failure Risk Assessment":
                    st.markdown("""
                    ## ⚠️ Failure Risk Assessment
                    
                    ### Risk Matrix
                    
                    **High Risk (>60% probability):**
                    - ID Fan #1: Bearing failure risk (68%)
                    
                    **Medium Risk (30-60% probability):** 
                    - Raw Mill #1: Liner wear (45%)
                    
                    **Low Risk (<30% probability):**
                    - Kiln #1, Cement Mill #1, Cooler #1
                    
                    ### Recommended Actions
                    
                    1. **Immediate (0-7 days):** ID Fan bearing inspection
                    2. **Short-term (8-30 days):** Raw mill liner assessment
                    3. **Long-term (30+ days):** Routine maintenance as scheduled
                    """)
                
                # Download button
                st.download_button(
                    label=f"📁 Download {report_type} ({export_format})",
                    data="Report content would be exported here...",
                    file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                    mime="application/octet-stream"
                )
    
    with tab4:
        st.subheader("⚙️ Predictive Maintenance Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎛️ Model Settings**")
            
            model_update_freq = st.selectbox("Model Update Frequency", ["Daily", "Weekly", "Monthly"])
            prediction_accuracy = st.slider("Required Prediction Accuracy", 0.70, 0.95, 0.85)
            alert_threshold = st.slider("Alert Threshold (Failure Probability)", 0.3, 0.9, 0.6)
            
            st.markdown("**📊 Data Sources**")
            
            st.checkbox("Vibration Sensors", value=True)
            st.checkbox("Temperature Sensors", value=True)
            st.checkbox("Oil Analysis", value=True)
            st.checkbox("Power Consumption", value=True)
            st.checkbox("Historical Maintenance", value=True)
        
        with col2:
            st.markdown("**🔔 Alert Configuration**")
            
            st.checkbox("Email Alerts", value=True)
            st.checkbox("SMS Notifications", value=False) 
            st.checkbox("Dashboard Alerts", value=True)
            st.checkbox("CMMS Integration", value=True)
            
            st.markdown("**⏰ Maintenance Windows**")
            
            st.selectbox("Preferred Maintenance Day", ["Monday", "Saturday", "Sunday"])
            st.time_input("Preferred Start Time", datetime.now().time())
            st.number_input("Max Maintenance Duration (hours)", min_value=1, max_value=72, value=8)
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully!")

if __name__ == "__main__":
    launch_predictive_maintenance_demo()
