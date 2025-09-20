import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
import json

class LIMSIntegration:
    """
    Laboratory Information Management System (LIMS) Integration
    with robotic lab automation and real-time quality monitoring
    """
    
    def __init__(self):
        # LIMS configuration
        self.lims_config = {
            'server_endpoint': 'https://lims.jkcement.com/api/v2',
            'authentication': 'oauth2_bearer_token',
            'sample_frequency_minutes': 30,
            'auto_analysis_enabled': True,
            'robotic_lab_connected': True
        }
        
        # Quality parameters and specifications
        self.quality_parameters = {
            'raw_meal': {
                'lsf': {'target': 95.0, 'range': [92, 98], 'unit': '%'},
                'sm': {'target': 2.3, 'range': [2.0, 2.6], 'unit': '-'},
                'am': {'target': 1.6, 'range': [1.2, 2.0], 'unit': '-'},
                'free_lime': {'target': 1.2, 'range': [0.8, 1.8], 'unit': '%'}
            },
            'clinker': {
                'free_lime': {'target': 1.0, 'range': [0.5, 1.5], 'unit': '%'},
                'c3s': {'target': 60, 'range': [55, 65], 'unit': '%'},
                'c2s': {'target': 15, 'range': [12, 18], 'unit': '%'},
                'c3a': {'target': 8, 'range': [6, 10], 'unit': '%'},
                'c4af': {'target': 12, 'range': [10, 14], 'unit': '%'}
            },
            'cement': {
                'blaine': {'target': 3400, 'range': [3200, 3600], 'unit': 'cm²/g'},
                '28d_strength': {'target': 48, 'range': [45, 52], 'unit': 'MPa'},
                'setting_time': {'target': 180, 'range': [120, 240], 'unit': 'minutes'},
                'so3': {'target': 2.8, 'range': [2.5, 3.2], 'unit': '%'}
            }
        }
        
        # Robotic lab configuration
        self.robotic_lab = {
            'x_ray_analyzer': {
                'model': 'Malvern Panalytical Epsilon 4',
                'status': 'Online',
                'sample_time_minutes': 8,
                'accuracy_percentage': 99.2,
                'last_calibration': '2025-09-15'
            },
            'sample_preparation': {
                'model': 'Herzog HRP-63',
                'status': 'Online', 
                'throughput_samples_hour': 12,
                'grinding_time_minutes': 3.5
            },
            'pneumatic_transport': {
                'status': 'Online',
                'transport_time_minutes': 2.5,
                'sample_integrity': 99.8
            },
            'auto_sampling': {
                'points_configured': 24,
                'active_points': 22,
                'sampling_interval_minutes': 30,
                'last_maintenance': '2025-09-10'
            }
        }
        
        # Generate historical lab data
        self._generate_historical_lab_data()
    
    def _generate_historical_lab_data(self):
        """Generate synthetic historical laboratory data"""
        
        # Generate 7 days of lab data (every 30 minutes)
        hours = 7 * 24
        samples = hours * 2  # Every 30 minutes
        base_time = datetime.now() - timedelta(hours=hours)
        
        self.lab_data = {
            'timestamps': [base_time + timedelta(minutes=i*30) for i in range(samples)],
            'sample_ids': [f"LAB{datetime.now().strftime('%Y%m%d')}{i+1000:04d}" for i in range(samples)],
            'material_types': [random.choice(['raw_meal', 'clinker', 'cement']) for _ in range(samples)]
        }
        
        # Generate quality data based on material type
        for param_group, parameters in self.quality_parameters.items():
            self.lab_data[param_group] = {}
            
            for param, spec in parameters.items():
                target = spec['target']
                range_val = spec['range']
                
                # Generate data with realistic variation and occasional out-of-spec
                data_points = []
                for i in range(samples):
                    if self.lab_data['material_types'][i] == param_group:
                        # 95% within spec, 5% potential outliers
                        if random.random() < 0.95:
                            value = np.random.normal(target, (range_val[1] - range_val[0]) * 0.15)
                        else:
                            # Occasional outlier
                            value = target + random.choice([-1, 1]) * (range_val[1] - range_val[0]) * 0.4
                        
                        data_points.append(max(0, value))
                    else:
                        data_points.append(None)
                
                self.lab_data[param_group][param] = data_points
    
    def get_real_time_lab_status(self) -> Dict:
        """Get current real-time laboratory status"""
        
        current_time = datetime.now()
        
        # Simulate current lab status
        status = {
            'timestamp': current_time.isoformat(),
            'overall_status': 'Online',
            'samples_in_queue': random.randint(2, 8),
            'samples_analyzed_today': random.randint(35, 48),
            'average_turnaround_minutes': random.uniform(12, 18),
            'equipment_status': self.robotic_lab,
            'active_alarms': self._generate_current_alarms(),
            'quality_alerts': self._generate_quality_alerts()
        }
        
        return status
    
    def _generate_current_alarms(self) -> List[Dict]:
        """Generate current equipment alarms"""
        
        alarms = []
        
        # Simulate occasional equipment issues
        if random.random() < 0.1:  # 10% chance of alarm
            alarm_types = [
                {'equipment': 'X-ray Analyzer', 'message': 'Tube intensity fluctuation detected', 'severity': 'Medium'},
                {'equipment': 'Sample Preparation', 'message': 'Grinding chamber cleaning required', 'severity': 'Low'},
                {'equipment': 'Pneumatic Transport', 'message': 'Line 3 pressure variation', 'severity': 'Medium'},
                {'equipment': 'Auto Sampling', 'message': 'Point 12 sample line blocked', 'severity': 'High'}
            ]
            
            alarms.append(random.choice(alarm_types))
        
        return alarms
    
    def _generate_quality_alerts(self) -> List[Dict]:
        """Generate current quality alerts"""
        
        alerts = []
        
        # Check recent samples for out-of-spec conditions
        recent_samples = 10
        
        for material_type in ['raw_meal', 'clinker', 'cement']:
            for param, spec in self.quality_parameters[material_type].items():
                # Get recent values
                recent_values = [v for v in self.lab_data[material_type][param][-recent_samples:] if v is not None]
                
                if recent_values:
                    latest_value = recent_values[-1]
                    target_range = spec['range']
                    
                    if latest_value < target_range[0] or latest_value > target_range[1]:
                        alerts.append({
                            'material': material_type,
                            'parameter': param,
                            'current_value': latest_value,
                            'specification': target_range,
                            'deviation': abs(latest_value - spec['target']),
                            'severity': 'High' if abs(latest_value - spec['target']) > (target_range[1] - target_range[0]) * 0.3 else 'Medium'
                        })
        
        return alerts
    
    def predict_quality_trends(self, material_type: str, parameter: str, hours_ahead: int = 4) -> Dict:
        """Predict quality parameter trends using historical data"""
        
        # Input validation
        if not isinstance(hours_ahead, int) or hours_ahead < 1 or hours_ahead > 24:
            return {'error': 'hours_ahead must be an integer between 1 and 24'}
        
        if not isinstance(material_type, str) or not material_type.strip():
            return {'error': 'material_type must be a non-empty string'}
        
        if not isinstance(parameter, str) or not parameter.strip():
            return {'error': 'parameter must be a non-empty string'}
        
        if material_type not in self.quality_parameters or parameter not in self.quality_parameters[material_type]:
            return {'error': 'Invalid material type or parameter'}
        
        # Get historical data for the parameter
        historical_data = [v for v in self.lab_data[material_type][parameter] if v is not None]
        
        if len(historical_data) < 10:
            return {'error': 'Insufficient historical data'}
        
        # Simple trend prediction using moving averages and linear regression
        recent_data = historical_data[-24:]  # Last 12 hours (24 samples)
        
        # Calculate trend
        x = np.arange(len(recent_data))
        coefficients = np.polyfit(x, recent_data, 1)
        trend_slope = coefficients[0]
        
        # Predict future values
        prediction_points = hours_ahead * 2  # Every 30 minutes
        future_x = np.arange(len(recent_data), len(recent_data) + prediction_points)
        predicted_values = np.polyval(coefficients, future_x)
        
        # Add realistic variation
        predicted_values = predicted_values + np.random.normal(0, np.std(recent_data) * 0.3, len(predicted_values))
        
        # Calculate prediction confidence
        recent_variation = np.std(recent_data[-12:])  # Last 6 hours
        confidence = max(0.6, 1.0 - (recent_variation / np.mean(recent_data)))
        
        spec = self.quality_parameters[material_type][parameter]
        
        prediction_result = {
            'current_value': recent_data[-1],
            'trend_direction': 'increasing' if trend_slope > 0.01 else 'decreasing' if trend_slope < -0.01 else 'stable',
            'trend_slope': trend_slope,
            'predicted_values': predicted_values.tolist(),
            'prediction_confidence': confidence,
            'specification_range': spec['range'],
            'target_value': spec['target'],
            'risk_assessment': self._assess_quality_risk(predicted_values, spec),
            'recommended_actions': self._generate_quality_recommendations(material_type, parameter, predicted_values, spec)
        }
        
        return prediction_result
    
    def _assess_quality_risk(self, predicted_values: np.ndarray, spec: Dict) -> Dict:
        """Assess risk of quality deviation"""
        
        spec_range = spec['range']
        target = spec['target']
        
        # Calculate risk metrics
        out_of_spec_count = sum(1 for v in predicted_values if v < spec_range[0] or v > spec_range[1])
        out_of_spec_percentage = (out_of_spec_count / len(predicted_values)) * 100
        
        max_deviation = max(abs(v - target) for v in predicted_values)
        
        # Risk categorization
        if out_of_spec_percentage > 25 or max_deviation > (spec_range[1] - spec_range[0]) * 0.5:
            risk_level = 'High'
        elif out_of_spec_percentage > 10 or max_deviation > (spec_range[1] - spec_range[0]) * 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_level': risk_level,
            'out_of_spec_probability': out_of_spec_percentage,
            'maximum_deviation': max_deviation,
            'stability_index': 1.0 - (np.std(predicted_values) / np.mean(predicted_values))
        }
    
    def _generate_quality_recommendations(self, material_type: str, parameter: str, 
                                        predicted_values: np.ndarray, spec: Dict) -> List[str]:
        """Generate quality improvement recommendations"""
        
        recommendations = []
        avg_predicted = np.mean(predicted_values)
        target = spec['target']
        spec_range = spec['range']
        
        # Parameter-specific recommendations
        if material_type == 'raw_meal':
            if parameter == 'lsf':
                if avg_predicted < spec_range[0]:
                    recommendations.append("Increase limestone proportion in raw meal blend")
                    recommendations.append("Reduce silica-bearing materials (clay, sand)")
                elif avg_predicted > spec_range[1]:
                    recommendations.append("Reduce limestone content")
                    recommendations.append("Increase silica modulus corrective materials")
            
            elif parameter == 'free_lime':
                if avg_predicted > spec_range[1]:
                    recommendations.append("Increase kiln temperature by 5-10°C")
                    recommendations.append("Reduce kiln speed to increase residence time")
                    recommendations.append("Check fuel distribution and combustion efficiency")
        
        elif material_type == 'clinker':
            if parameter == 'free_lime':
                if avg_predicted > spec_range[1]:
                    recommendations.append("Increase burning zone temperature")
                    recommendations.append("Optimize fuel/air ratio for better combustion")
                    recommendations.append("Check raw meal fineness and homogeneity")
            
            elif parameter == 'c3s':
                if avg_predicted < target:
                    recommendations.append("Increase LSF in raw meal composition")
                    recommendations.append("Optimize burning conditions for better formation")
        
        elif material_type == 'cement':
            if parameter == 'blaine':
                if avg_predicted < target:
                    recommendations.append("Increase cement mill grinding time")
                    recommendations.append("Optimize separator efficiency")
                    recommendations.append("Check grinding media charge and gradation")
            
            elif parameter == '28d_strength':
                if avg_predicted < spec_range[0]:
                    recommendations.append("Optimize clinker mineral composition")
                    recommendations.append("Increase cement fineness (Blaine)")
                    recommendations.append("Review gypsum content and SO3 levels")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue current process parameters - quality is stable")
            recommendations.append("Monitor trends closely for any deviations")
        
        return recommendations

def launch_lims_integration_demo():
    """Launch LIMS Integration Dashboard"""
    
    st.set_page_config(
        page_title="🧪 LIMS Integration Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 LIMS Integration Dashboard")
    st.markdown("**Laboratory Information Management System with Robotic Lab Integration**")
    
    # Initialize LIMS
    if 'lims_integration' not in st.session_state:
        st.session_state.lims_integration = LIMSIntegration()
    
    lims = st.session_state.lims_integration
    
    # Sidebar controls
    with st.sidebar:
        st.header("🏭 Lab Configuration")
        
        selected_material = st.selectbox("Material Type", [
            "raw_meal",
            "clinker", 
            "cement"
        ], format_func=lambda x: x.replace('_', ' ').title())
        
        available_params = list(lims.quality_parameters[selected_material].keys())
        selected_parameter = st.selectbox("Quality Parameter", available_params)
        
        st.header("🔬 Analysis Settings")
        
        prediction_hours = st.slider("Prediction Horizon (hours)", 2, 12, 4)
        confidence_threshold = st.slider("Alert Threshold (%)", 70, 95, 85)
        
        st.header("🤖 Robotic Lab Status")
        
        lab_status = lims.get_real_time_lab_status()
        
        st.metric("Samples in Queue", lab_status['samples_in_queue'])
        st.metric("Analyzed Today", lab_status['samples_analyzed_today'])
        st.metric("Avg Turnaround", f"{lab_status['average_turnaround_minutes']:.1f} min")
        
        status_color = "🟢" if lab_status['overall_status'] == 'Online' else "🔴"
        st.markdown(f"**Status:** {status_color} {lab_status['overall_status']}")
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time Monitoring", "🔮 Quality Prediction", "🤖 Robotic Lab Status", "📈 Historical Analysis"])
    
    with tab1:
        st.subheader("📊 Real-time Quality Monitoring")
        
        # Current quality status
        col1, col2, col3 = st.columns(3)
        
        # Get latest values for each material type
        latest_values = {}
        for material in ['raw_meal', 'clinker', 'cement']:
            latest_values[material] = {}
            for param in lims.quality_parameters[material]:
                recent_data = [v for v in lims.lab_data[material][param] if v is not None]
                if recent_data:
                    latest_values[material][param] = recent_data[-1]
        
        with col1:
            st.markdown("**🏭 Raw Meal Quality**")
            for param, value in latest_values.get('raw_meal', {}).items():
                spec = lims.quality_parameters['raw_meal'][param]
                status = "✅" if spec['range'][0] <= value <= spec['range'][1] else "⚠️"
                st.metric(
                    f"{status} {param.upper()}", 
                    f"{value:.2f} {spec['unit']}", 
                    delta=f"Target: {spec['target']}"
                )
        
        with col2:
            st.markdown("**🔥 Clinker Quality**")
            for param, value in latest_values.get('clinker', {}).items():
                spec = lims.quality_parameters['clinker'][param]
                status = "✅" if spec['range'][0] <= value <= spec['range'][1] else "⚠️"
                st.metric(
                    f"{status} {param.upper()}", 
                    f"{value:.1f} {spec['unit']}", 
                    delta=f"Target: {spec['target']}"
                )
        
        with col3:
            st.markdown("**🏗️ Cement Quality**")
            for param, value in latest_values.get('cement', {}).items():
                spec = lims.quality_parameters['cement'][param]
                status = "✅" if spec['range'][0] <= value <= spec['range'][1] else "⚠️"
                st.metric(
                    f"{status} {param}", 
                    f"{value:.0f} {spec['unit']}", 
                    delta=f"Target: {spec['target']}"
                )
        
        # Quality alerts
        quality_alerts = lab_status['quality_alerts']
        
        if quality_alerts:
            st.subheader("🚨 Quality Alerts")
            
            for alert in quality_alerts:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alert['severity']]
                
                st.markdown(f"""
                <div style="border-left: 4px solid {'#F44336' if alert['severity'] == 'High' else '#FF9800' if alert['severity'] == 'Medium' else '#4CAF50'}; 
                            padding: 1rem; margin: 0.5rem 0; background: #f9f9f9; border-radius: 0 8px 8px 0;">
                    <strong>{severity_color} {alert['material'].replace('_', ' ').title()} - {alert['parameter'].upper()}</strong><br>
                    Current: {alert['current_value']:.2f} | Spec: {alert['specification'][0]}-{alert['specification'][1]}<br>
                    Deviation: {alert['deviation']:.2f} ({alert['severity']} Priority)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ All quality parameters within specification limits")
        
        # Real-time trend chart
        st.subheader(f"📈 Real-time Trend - {selected_material.replace('_', ' ').title()} {selected_parameter.upper()}")
        
        # Get data for selected parameter
        timestamps = lims.lab_data['timestamps']
        parameter_data = lims.lab_data[selected_material][selected_parameter]
        
        # Filter non-null values
        filtered_data = [(t, v) for t, v in zip(timestamps, parameter_data) if v is not None]
        
        if filtered_data:
            times, values = zip(*filtered_data)
            
            spec = lims.quality_parameters[selected_material][selected_parameter]
            
            fig = go.Figure()
            
            # Add parameter trend
            fig.add_trace(go.Scatter(
                x=times[-50:],  # Last 50 points
                y=values[-50:],
                mode='lines+markers',
                name=f'{selected_parameter.upper()}',
                line=dict(color='blue', width=2)
            ))
            
            # Add specification limits
            fig.add_hline(
                y=spec['range'][0], 
                line_dash="dash", 
                line_color="red",
                annotation_text="Lower Spec Limit"
            )
            
            fig.add_hline(
                y=spec['range'][1], 
                line_dash="dash", 
                line_color="red",
                annotation_text="Upper Spec Limit"
            )
            
            fig.add_hline(
                y=spec['target'], 
                line_dash="dot", 
                line_color="green",
                annotation_text="Target"
            )
            
            fig.update_layout(
                title=f"{selected_material.replace('_', ' ').title()} {selected_parameter.upper()} Trend",
                xaxis_title="Time",
                yaxis_title=f"{selected_parameter.upper()} ({spec['unit']})",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"🔮 Quality Prediction - {selected_material.replace('_', ' ').title()} {selected_parameter.upper()}")
        
        if st.button("🚀 Generate Prediction"):
            
            with st.spinner("Analyzing historical data and generating predictions..."):
                
                prediction = lims.predict_quality_trends(
                    selected_material, 
                    selected_parameter, 
                    prediction_hours
                )
                
                # Store prediction in session state
                st.session_state.quality_prediction = prediction
            
            st.success(f"✅ Prediction generated for next {prediction_hours} hours")
        
        # Display prediction results
        if 'quality_prediction' in st.session_state:
            
            prediction = st.session_state.quality_prediction
            
            if 'error' not in prediction:
                
                # Prediction summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Value", 
                        f"{prediction['current_value']:.2f}",
                        delta=prediction['trend_direction'].title()
                    )
                
                with col2:
                    st.metric(
                        "Prediction Confidence", 
                        f"{prediction['prediction_confidence']:.1%}"
                    )
                
                with col3:
                    risk = prediction['risk_assessment']
                    risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk['risk_level']]
                    st.metric(
                        "Quality Risk", 
                        f"{risk_color} {risk['risk_level']}",
                        delta=f"{risk['out_of_spec_probability']:.1f}% out-of-spec"
                    )
                
                with col4:
                    st.metric(
                        "Stability Index", 
                        f"{risk['stability_index']:.2f}"
                    )
                
                # Prediction chart
                st.subheader("📊 Quality Prediction Chart")
                
                # Create prediction timeline
                current_time = datetime.now()
                prediction_times = [current_time + timedelta(minutes=i*30) for i in range(len(prediction['predicted_values']))]
                
                spec = lims.quality_parameters[selected_material][selected_parameter]
                
                fig = go.Figure()
                
                # Add predicted values
                fig.add_trace(go.Scatter(
                    x=prediction_times,
                    y=prediction['predicted_values'],
                    mode='lines+markers',
                    name='Predicted Values',
                    line=dict(color='orange', width=3)
                ))
                
                # Add current value
                fig.add_trace(go.Scatter(
                    x=[current_time],
                    y=[prediction['current_value']],
                    mode='markers',
                    name='Current Value',
                    marker=dict(color='blue', size=12)
                ))
                
                # Add specification limits
                fig.add_hline(y=spec['range'][0], line_dash="dash", line_color="red", annotation_text="Lower Spec")
                fig.add_hline(y=spec['range'][1], line_dash="dash", line_color="red", annotation_text="Upper Spec")
                fig.add_hline(y=spec['target'], line_dash="dot", line_color="green", annotation_text="Target")
                
                fig.update_layout(
                    title=f"Quality Prediction - Next {prediction_hours} Hours",
                    xaxis_title="Time",
                    yaxis_title=f"{selected_parameter.upper()} ({spec['unit']})",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 AI Recommendations")
                
                recommendations = prediction['recommended_actions']
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {rec}")
                
                # Risk assessment details
                with st.expander("🔍 Detailed Risk Assessment", expanded=False):
                    
                    risk = prediction['risk_assessment']
                    
                    st.markdown(f"""
                    **Quality Risk Analysis:**
                    
                    • **Risk Level**: {risk['risk_level']}
                    • **Out-of-Spec Probability**: {risk['out_of_spec_probability']:.1f}%
                    • **Maximum Deviation**: {risk['maximum_deviation']:.3f}
                    • **Process Stability**: {risk['stability_index']:.2f}
                    
                    **Interpretation:**
                    - Stability Index > 0.9: Excellent process control
                    - Stability Index 0.7-0.9: Good process control  
                    - Stability Index < 0.7: Process optimization needed
                    """)
            
            else:
                st.error(f"❌ Prediction Error: {prediction['error']}")
    
    with tab3:
        st.subheader("🤖 Robotic Laboratory Status")
        
        lab_status = lims.get_real_time_lab_status()
        
        # Equipment status overview
        st.subheader("⚙️ Equipment Status")
        
        equipment_data = []
        for equipment, details in lab_status['equipment_status'].items():
            
            status_icon = "🟢" if details['status'] == 'Online' else "🔴"
            
            if equipment == 'x_ray_analyzer':
                equipment_data.append({
                    'Equipment': 'X-ray Analyzer',
                    'Status': f"{status_icon} {details['status']}",
                    'Model': details['model'],
                    'Performance': f"{details['accuracy_percentage']:.1f}% accuracy",
                    'Last Service': details['last_calibration']
                })
            elif equipment == 'sample_preparation':
                equipment_data.append({
                    'Equipment': 'Sample Preparation',
                    'Status': f"{status_icon} {details['status']}",
                    'Model': details['model'],
                    'Performance': f"{details['throughput_samples_hour']} samples/hr",
                    'Last Service': 'N/A'
                })
            elif equipment == 'pneumatic_transport':
                equipment_data.append({
                    'Equipment': 'Pneumatic Transport',
                    'Status': f"{status_icon} {details['status']}",
                    'Model': 'Pneumatic System',
                    'Performance': f"{details['sample_integrity']:.1f}% integrity",
                    'Last Service': 'N/A'
                })
            elif equipment == 'auto_sampling':
                equipment_data.append({
                    'Equipment': 'Auto Sampling',
                    'Status': f"{status_icon} {details['status']}",
                    'Model': 'Multi-point Sampling',
                    'Performance': f"{details['active_points']}/{details['points_configured']} points active",
                    'Last Service': details['last_maintenance']
                })
        
        equipment_df = pd.DataFrame(equipment_data)
        st.dataframe(equipment_df, use_container_width=True)
        
        # Active alarms
        if lab_status['active_alarms']:
            st.subheader("🚨 Active Equipment Alarms")
            
            for alarm in lab_status['active_alarms']:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alarm['severity']]
                
                st.markdown(f"""
                <div style="border-left: 4px solid {'#F44336' if alarm['severity'] == 'High' else '#FF9800' if alarm['severity'] == 'Medium' else '#4CAF50'}; 
                            padding: 1rem; margin: 0.5rem 0; background: #fff3f3; border-radius: 0 8px 8px 0;">
                    <strong>{severity_color} {alarm['equipment']}</strong><br>
                    {alarm['message']}<br>
                    <small>Priority: {alarm['severity']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active equipment alarms")
        
        # Lab throughput metrics
        st.subheader("📊 Laboratory Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Daily Performance**")
            st.metric("Samples Analyzed", lab_status['samples_analyzed_today'], delta="+5 vs yesterday")
            st.metric("Average Turnaround", f"{lab_status['average_turnaround_minutes']:.1f} min", delta="-2.1 min")
            st.metric("First Pass Success", "96.8%", delta="+1.2%")
        
        with col2:
            st.markdown("**🎯 Quality Metrics**")
            st.metric("Analysis Accuracy", "99.2%", delta="+0.1%")
            st.metric("Calibration Status", "Current", delta="Valid")
            st.metric("Sample Integrity", "99.8%", delta="+0.2%")
        
        with col3:
            st.markdown("**⚙️ Equipment Efficiency**")
            st.metric("Overall Equipment Effectiveness", "94.5%", delta="+2.3%")
            st.metric("Planned Downtime", "2.1 hrs/week", delta="-0.5 hrs")
            st.metric("Unplanned Downtime", "0.3 hrs/week", delta="-0.8 hrs")
        
        # Sample tracking
        st.subheader("📦 Sample Tracking")
        
        # Simulate sample queue
        sample_queue = []
        for i in range(lab_status['samples_in_queue']):
            sample_queue.append({
                'Sample ID': f"LAB{datetime.now().strftime('%Y%m%d')}{2000+i:04d}",
                'Material': random.choice(['Raw Meal', 'Clinker', 'Cement']),
                'Collection Time': (datetime.now() - timedelta(minutes=random.randint(5, 45))).strftime('%H:%M'),
                'Status': random.choice(['Preparation', 'Analysis', 'Review']),
                'ETA': f"{random.randint(5, 25)} min"
            })
        
        queue_df = pd.DataFrame(sample_queue)
        st.dataframe(queue_df, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Historical Quality Analysis")
        
        # Historical trend analysis
        analysis_period = st.selectbox("Analysis Period", [
            "Last 24 Hours",
            "Last 3 Days", 
            "Last Week",
            "Last Month"
        ])
        
        period_hours = {"Last 24 Hours": 24, "Last 3 Days": 72, "Last Week": 168, "Last Month": 720}
        hours = period_hours[analysis_period]
        
        # Multi-parameter comparison
        st.subheader(f"🔍 Multi-Parameter Analysis - {analysis_period}")
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Raw Meal LSF', 'Clinker Free Lime', 'Cement Blaine', 'Cement 28d Strength'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Raw meal LSF
        times = lims.lab_data['timestamps'][-hours*2:]
        lsf_data = [v for v in lims.lab_data['raw_meal']['lsf'][-hours*2:] if v is not None]
        lsf_times = [t for t, v in zip(times, lims.lab_data['raw_meal']['lsf'][-hours*2:]) if v is not None]
        
        if lsf_data:
            fig.add_trace(
                go.Scatter(x=lsf_times, y=lsf_data, name='LSF (%)', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Clinker free lime
        clinker_lime_data = [v for v in lims.lab_data['clinker']['free_lime'][-hours*2:] if v is not None]
        clinker_times = [t for t, v in zip(times, lims.lab_data['clinker']['free_lime'][-hours*2:]) if v is not None]
        
        if clinker_lime_data:
            fig.add_trace(
                go.Scatter(x=clinker_times, y=clinker_lime_data, name='Free Lime (%)', line=dict(color='red')),
                row=1, col=2
            )
        
        # Cement blaine
        blaine_data = [v for v in lims.lab_data['cement']['blaine'][-hours*2:] if v is not None]
        blaine_times = [t for t, v in zip(times, lims.lab_data['cement']['blaine'][-hours*2:]) if v is not None]
        
        if blaine_data:
            fig.add_trace(
                go.Scatter(x=blaine_times, y=blaine_data, name='Blaine (cm²/g)', line=dict(color='green')),
                row=2, col=1
            )
        
        # Cement 28d strength
        strength_data = [v for v in lims.lab_data['cement']['28d_strength'][-hours*2:] if v is not None]
        strength_times = [t for t, v in zip(times, lims.lab_data['cement']['28d_strength'][-hours*2:]) if v is not None]
        
        if strength_data:
            fig.add_trace(
                go.Scatter(x=strength_times, y=strength_data, name='28d Strength (MPa)', line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Historical Quality Trends - {analysis_period}",
            showlegend=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        # Calculate statistics for all parameters
        stats_data = []
        
        for material_type, parameters in lims.quality_parameters.items():
            for param, spec in parameters.items():
                data = [v for v in lims.lab_data[material_type][param][-hours*2:] if v is not None]
                
                if data:
                    stats_data.append({
                        'Material': material_type.replace('_', ' ').title(),
                        'Parameter': param.upper(),
                        'Count': len(data),
                        'Average': f"{np.mean(data):.2f}",
                        'Std Dev': f"{np.std(data):.2f}",
                        'Min': f"{np.min(data):.2f}",
                        'Max': f"{np.max(data):.2f}",
                        'Within Spec': f"{sum(1 for v in data if spec['range'][0] <= v <= spec['range'][1]) / len(data) * 100:.1f}%"
                    })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    launch_lims_integration_demo()
