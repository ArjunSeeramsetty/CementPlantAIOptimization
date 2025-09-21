# src/cement_ai_platform/dashboard/dynamic_plant_twin.py
"""
Dynamic Plant Digital Twin UI for Live POC Demonstration
Real-time plant status dashboard with AI-powered insights and scenario injection
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import threading
import queue
from typing import Dict, List, Any

class DynamicPlantTwin:
    """
    Real-time dynamic plant digital twin for live POC demonstration
    """
    
    def __init__(self):
        # Initialize plant state with realistic values
        self.plant_state = {
            'kiln_temp_c': 1450,
            'free_lime_pct': 1.2,
            'feed_rate_tph': 167,
            'fuel_rate_tph': 16.5,
            'kiln_speed_rpm': 3.5,
            'o2_percent': 3.2,
            'production_rate_tph': 165,
            'energy_efficiency_pct': 92,
            'nox_mg_nm3': 480,
            'preheater_temp_c': 880,
            'cooler_temp_c': 95,
            'vibration_mm_s': 4.5,
            'power_consumption_mw': 45
        }
        
        # Historical data for trends
        self.history_length = 100
        self.time_history = []
        self.data_history = {key: [] for key in self.plant_state.keys()}
        
        # Plant equipment status
        self.equipment_status = {
            'kiln': {'status': 'Running', 'efficiency': 95, 'maintenance_due': 45},
            'raw_mill': {'status': 'Running', 'efficiency': 88, 'maintenance_due': 12},
            'cement_mill': {'status': 'Running', 'efficiency': 92, 'maintenance_due': 30},
            'id_fan': {'status': 'Running', 'efficiency': 87, 'maintenance_due': 8},
            'cooler': {'status': 'Running', 'efficiency': 91, 'maintenance_due': 22}
        }
        
        # Anomaly flags and AI recommendations
        self.anomalies = []
        self.ai_recommendations = []
        
        # Initialize history
        self._initialize_history()
    
    def _initialize_history(self):
        """Initialize historical data with realistic patterns"""
        base_time = datetime.now() - timedelta(minutes=self.history_length)
        
        for i in range(self.history_length):
            timestamp = base_time + timedelta(minutes=i)
            self.time_history.append(timestamp)
            
            for key, base_value in self.plant_state.items():
                # Add realistic variation with some correlation
                if key == 'kiln_temp_c':
                    variation = np.sin(i * 0.1) * 10 + np.random.normal(0, base_value * 0.02)
                elif key == 'free_lime_pct':
                    # Free lime inversely correlated with temperature
                    temp_variation = np.sin(i * 0.1) * 10
                    variation = -temp_variation * 0.01 + np.random.normal(0, base_value * 0.03)
                else:
                    variation = np.random.normal(0, base_value * 0.02)
                
                self.data_history[key].append(base_value + variation)
    
    def update_plant_state(self):
        """Update plant state with realistic dynamics and correlations"""
        
        current_time = datetime.now()
        
        # Simulate realistic plant dynamics with correlations
        
        # Temperature oscillation with slow trend
        temp_trend = np.sin(time.time() * 0.001) * 5  # Slow temperature oscillation
        temp_noise = np.random.normal(0, 2)
        self.plant_state['kiln_temp_c'] += temp_trend + temp_noise
        self.plant_state['kiln_temp_c'] = np.clip(self.plant_state['kiln_temp_c'], 1430, 1470)
        
        # Free lime inversely correlated with temperature
        temp_deviation = (self.plant_state['kiln_temp_c'] - 1450)
        lime_effect = temp_deviation * -0.01
        lime_noise = np.random.normal(0, 0.05)
        self.plant_state['free_lime_pct'] += lime_effect + lime_noise
        self.plant_state['free_lime_pct'] = np.clip(self.plant_state['free_lime_pct'], 0.8, 2.5)
        
        # Feed rate affects production
        feed_variation = np.random.normal(0, 2)
        self.plant_state['feed_rate_tph'] += feed_variation
        self.plant_state['feed_rate_tph'] = np.clip(self.plant_state['feed_rate_tph'], 150, 180)
        
        # Production correlated with feed rate (98% efficiency)
        self.plant_state['production_rate_tph'] = (
            self.plant_state['feed_rate_tph'] * 0.98 + np.random.normal(0, 2)
        )
        
        # Fuel rate responds to free lime (control logic)
        if self.plant_state['free_lime_pct'] > 1.8:
            self.plant_state['fuel_rate_tph'] += np.random.normal(0.2, 0.1)
        elif self.plant_state['free_lime_pct'] < 1.0:
            self.plant_state['fuel_rate_tph'] += np.random.normal(-0.1, 0.1)
        else:
            self.plant_state['fuel_rate_tph'] += np.random.normal(0, 0.05)
        
        self.plant_state['fuel_rate_tph'] = np.clip(self.plant_state['fuel_rate_tph'], 14, 20)
        
        # Kiln speed with slight variation
        self.plant_state['kiln_speed_rpm'] += np.random.normal(0, 0.05)
        self.plant_state['kiln_speed_rpm'] = np.clip(self.plant_state['kiln_speed_rpm'], 3.0, 4.0)
        
        # Oxygen percentage
        self.plant_state['o2_percent'] += np.random.normal(0, 0.1)
        self.plant_state['o2_percent'] = np.clip(self.plant_state['o2_percent'], 2.5, 4.5)
        
        # Energy efficiency affected by operations
        efficiency_base = 92 - abs(self.plant_state['free_lime_pct'] - 1.2) * 2
        self.plant_state['energy_efficiency_pct'] = efficiency_base + np.random.normal(0, 1)
        self.plant_state['energy_efficiency_pct'] = np.clip(self.plant_state['energy_efficiency_pct'], 80, 98)
        
        # Environmental parameters
        self.plant_state['nox_mg_nm3'] += np.random.normal(0, 15)
        self.plant_state['nox_mg_nm3'] = np.clip(self.plant_state['nox_mg_nm3'], 400, 600)
        
        self.plant_state['preheater_temp_c'] += np.random.normal(0, 5)
        self.plant_state['preheater_temp_c'] = np.clip(self.plant_state['preheater_temp_c'], 850, 920)
        
        self.plant_state['cooler_temp_c'] += np.random.normal(0, 3)
        self.plant_state['cooler_temp_c'] = np.clip(self.plant_state['cooler_temp_c'], 85, 110)
        
        # Vibration with equipment degradation
        self.plant_state['vibration_mm_s'] += np.random.normal(0, 0.2)
        self.plant_state['vibration_mm_s'] = np.clip(self.plant_state['vibration_mm_s'], 3.0, 8.0)
        
        # Power consumption correlated with production
        self.plant_state['power_consumption_mw'] = (
            self.plant_state['production_rate_tph'] * 0.27 + np.random.normal(0, 1)
        )
        
        # Update history
        self.time_history.append(current_time)
        for key, value in self.plant_state.items():
            self.data_history[key].append(value)
        
        # Maintain history length
        if len(self.time_history) > self.history_length:
            self.time_history = self.time_history[-self.history_length:]
            for key in self.data_history:
                self.data_history[key] = self.data_history[key][-self.history_length:]
        
        # Check for anomalies and generate recommendations
        self._check_anomalies()
        self._generate_ai_recommendations()
        self._update_equipment_status()
    
    def _check_anomalies(self):
        """Check for plant anomalies and create alerts"""
        current_anomalies = []
        
        # Quality anomalies
        if self.plant_state['free_lime_pct'] > 2.0:
            current_anomalies.append({
                'type': 'Quality',
                'severity': 'High',
                'message': f"Free lime elevated at {self.plant_state['free_lime_pct']:.2f}% (Normal: <1.5%)",
                'timestamp': datetime.now()
            })
        
        # Process anomalies
        if self.plant_state['kiln_temp_c'] > 1465:
            current_anomalies.append({
                'type': 'Process',
                'severity': 'Medium',
                'message': f"Kiln temperature high at {self.plant_state['kiln_temp_c']:.0f}°C (Target: 1450°C)",
                'timestamp': datetime.now()
            })
        
        # Equipment anomalies
        if self.plant_state['vibration_mm_s'] > 7.0:
            current_anomalies.append({
                'type': 'Equipment',
                'severity': 'High',
                'message': f"High vibration detected: {self.plant_state['vibration_mm_s']:.1f} mm/s (Normal: <6.0)",
                'timestamp': datetime.now()
            })
        
        # Energy efficiency anomalies
        if self.plant_state['energy_efficiency_pct'] < 88:
            current_anomalies.append({
                'type': 'Energy',
                'severity': 'Medium',
                'message': f"Energy efficiency low at {self.plant_state['energy_efficiency_pct']:.1f}% (Target: >90%)",
                'timestamp': datetime.now()
            })
        
        # Environmental anomalies
        if self.plant_state['nox_mg_nm3'] > 550:
            current_anomalies.append({
                'type': 'Environmental',
                'severity': 'Medium',
                'message': f"NOx emissions high at {self.plant_state['nox_mg_nm3']:.0f} mg/Nm³ (Limit: 500)",
                'timestamp': datetime.now()
            })
        
        # Update anomalies list (keep last 10)
        self.anomalies.extend(current_anomalies)
        self.anomalies = self.anomalies[-10:]
    
    def _generate_ai_recommendations(self):
        """Generate AI recommendations based on current plant state"""
        recommendations = []
        
        # Quality recommendations
        if self.plant_state['free_lime_pct'] > 1.8:
            new_fuel_rate = min(20, self.plant_state['fuel_rate_tph'] + 0.5)
            recommendations.append({
                'type': 'Control Action',
                'priority': 'High',
                'action': f"Increase fuel rate to {new_fuel_rate:.1f} t/h",
                'expected_benefit': "Reduce free lime by 0.3-0.5% within 30 minutes",
                'confidence': 0.87,
                'timestamp': datetime.now()
            })
        
        # Energy optimization
        if self.plant_state['energy_efficiency_pct'] < 90:
            new_speed = max(3.0, self.plant_state['kiln_speed_rpm'] - 0.1)
            recommendations.append({
                'type': 'Optimization',
                'priority': 'Medium',
                'action': f"Adjust kiln speed to {new_speed:.2f} rpm",
                'expected_benefit': "Improve energy efficiency by 1-2%",
                'confidence': 0.73,
                'timestamp': datetime.now()
            })
        
        # Maintenance recommendations
        if self.plant_state['vibration_mm_s'] > 6.5:
            recommendations.append({
                'type': 'Maintenance',
                'priority': 'High',
                'action': "Schedule bearing inspection for kiln drive",
                'expected_benefit': "Prevent potential equipment failure within 48 hours",
                'confidence': 0.91,
                'timestamp': datetime.now()
            })
        
        # Production optimization
        if self.plant_state['production_rate_tph'] < 160:
            new_feed_rate = min(180, self.plant_state['feed_rate_tph'] + 3)
            recommendations.append({
                'type': 'Production',
                'priority': 'Medium',
                'action': f"Increase feed rate to {new_feed_rate:.1f} t/h",
                'expected_benefit': "Increase production by 2-3 t/h",
                'confidence': 0.82,
                'timestamp': datetime.now()
            })
        
        # Environmental recommendations
        if self.plant_state['nox_mg_nm3'] > 520:
            recommendations.append({
                'type': 'Environmental',
                'priority': 'Medium',
                'action': "Optimize oxygen levels and fuel distribution",
                'expected_benefit': "Reduce NOx emissions by 10-15%",
                'confidence': 0.76,
                'timestamp': datetime.now()
            })
        
        # Update recommendations (keep last 5)
        self.ai_recommendations.extend(recommendations)
        self.ai_recommendations = self.ai_recommendations[-5:]
    
    def _update_equipment_status(self):
        """Update equipment status based on plant conditions"""
        
        # Raw mill affected by vibration
        if self.plant_state['vibration_mm_s'] > 6.0:
            self.equipment_status['raw_mill']['efficiency'] = max(75, 
                self.equipment_status['raw_mill']['efficiency'] - 1)
        else:
            self.equipment_status['raw_mill']['efficiency'] = min(95,
                self.equipment_status['raw_mill']['efficiency'] + 0.5)
        
        # Kiln efficiency affected by temperature control
        temp_deviation = abs(self.plant_state['kiln_temp_c'] - 1450)
        if temp_deviation > 15:
            self.equipment_status['kiln']['efficiency'] = max(85,
                self.equipment_status['kiln']['efficiency'] - 0.5)
        else:
            self.equipment_status['kiln']['efficiency'] = min(98,
                self.equipment_status['kiln']['efficiency'] + 0.2)
        
        # ID fan efficiency based on oxygen levels
        o2_deviation = abs(self.plant_state['o2_percent'] - 3.2)
        if o2_deviation > 0.5:
            self.equipment_status['id_fan']['efficiency'] = max(80,
                self.equipment_status['id_fan']['efficiency'] - 0.3)
        else:
            self.equipment_status['id_fan']['efficiency'] = min(95,
                self.equipment_status['id_fan']['efficiency'] + 0.1)
        
        # Cooler efficiency based on temperature
        if self.plant_state['cooler_temp_c'] > 105:
            self.equipment_status['cooler']['efficiency'] = max(80,
                self.equipment_status['cooler']['efficiency'] - 0.4)
        else:
            self.equipment_status['cooler']['efficiency'] = min(96,
                self.equipment_status['cooler']['efficiency'] + 0.1)
        
        # Decrease maintenance days
        for equipment in self.equipment_status:
            self.equipment_status[equipment]['maintenance_due'] -= 0.01
            if self.equipment_status[equipment]['maintenance_due'] < 0:
                self.equipment_status[equipment]['maintenance_due'] = random.randint(30, 60)

def launch_dynamic_plant_twin():
    """Launch dynamic plant twin dashboard"""
    
    # Note: st.set_page_config() is called in the main unified dashboard
    
    # CSS for dynamic styling
    st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-running { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FF9800; font-weight: bold; }
    .status-critical { color: #F44336; font-weight: bold; }
    
    .anomaly-alert {
        border-left: 4px solid #ff4444;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fff3f3;
        border-radius: 0 8px 8px 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .recommendation-card {
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f0f8ff;
        border-radius: 0 8px 8px 0;
    }
    
    .equipment-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize plant twin in session state
    if 'plant_twin' not in st.session_state:
        st.session_state.plant_twin = DynamicPlantTwin()
    
    plant_twin = st.session_state.plant_twin
    
    # Title and header
    st.title("🏭 Dynamic Plant Digital Twin - Live POC Demo")
    st.markdown("**Real-time cement plant simulation with AI-powered insights and predictive analytics**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        auto_update = st.checkbox("🔄 Auto Update", value=True, help="Automatically update plant data")
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 3, help="How often to refresh data")
        
        st.header("🎯 Scenario Injection")
        st.markdown("*Inject realistic plant scenarios for demonstration*")
        
        if st.button("🔥 Inject High Temperature", help="Simulate kiln overheating"):
            plant_twin.plant_state['kiln_temp_c'] = 1475
            st.success("High temperature scenario injected!")
        
        if st.button("⚠️ Inject Quality Issue", help="Simulate free lime elevation"):
            plant_twin.plant_state['free_lime_pct'] = 2.3
            st.warning("Quality issue scenario injected!")
        
        if st.button("📳 Inject Vibration Alert", help="Simulate equipment vibration"):
            plant_twin.plant_state['vibration_mm_s'] = 7.5
            st.error("Vibration alert scenario injected!")
        
        if st.button("🌪️ Inject Environmental Issue", help="Simulate high NOx emissions"):
            plant_twin.plant_state['nox_mg_nm3'] = 580
            st.error("Environmental issue scenario injected!")
        
        if st.button("🔄 Reset to Normal", help="Reset all parameters to normal"):
            plant_twin.plant_state['kiln_temp_c'] = 1450
            plant_twin.plant_state['free_lime_pct'] = 1.2
            plant_twin.plant_state['vibration_mm_s'] = 4.5
            plant_twin.plant_state['nox_mg_nm3'] = 480
            st.info("Plant state reset to normal!")
        
        st.header("📊 Display Options")
        show_history = st.checkbox("Show Historical Trends", value=True)
        show_equipment = st.checkbox("Show Equipment Status", value=True)
        show_anomalies = st.checkbox("Show Anomaly Alerts", value=True)
        show_recommendations = st.checkbox("Show AI Recommendations", value=True)
    
    # Main dashboard layout
    
    # Top-level KPIs
    st.subheader("📈 Live Plant KPIs")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        temp_status = "🔥" if plant_twin.plant_state['kiln_temp_c'] > 1465 else "🌡️"
        temp_delta = np.random.uniform(-2, 2)
        st.metric(
            f"{temp_status} Kiln Temperature", 
            f"{plant_twin.plant_state['kiln_temp_c']:.1f}°C",
            delta=f"{temp_delta:+.1f}°C",
            help="Target: 1450°C"
        )
    
    with col2:
        lime_status = "⚠️" if plant_twin.plant_state['free_lime_pct'] > 1.8 else "✅"
        lime_delta = np.random.uniform(-0.1, 0.1)
        st.metric(
            f"{lime_status} Free Lime", 
            f"{plant_twin.plant_state['free_lime_pct']:.2f}%",
            delta=f"{lime_delta:+.2f}%",
            help="Target: <1.5%"
        )
    
    with col3:
        prod_delta = np.random.uniform(-2, 3)
        st.metric(
            "📦 Production Rate", 
            f"{plant_twin.plant_state['production_rate_tph']:.1f} t/h",
            delta=f"{prod_delta:+.1f} t/h",
            help="Current production rate"
        )
    
    with col4:
        efficiency_status = "⚡" if plant_twin.plant_state['energy_efficiency_pct'] > 90 else "🔋"
        eff_delta = np.random.uniform(-0.5, 1.0)
        st.metric(
            f"{efficiency_status} Energy Efficiency", 
            f"{plant_twin.plant_state['energy_efficiency_pct']:.1f}%",
            delta=f"{eff_delta:+.1f}%",
            help="Target: >90%"
        )
    
    with col5:
        vibration_status = "🚨" if plant_twin.plant_state['vibration_mm_s'] > 6.5 else "📳"
        vib_delta = np.random.uniform(-0.2, 0.2)
        st.metric(
            f"{vibration_status} Vibration", 
            f"{plant_twin.plant_state['vibration_mm_s']:.1f} mm/s",
            delta=f"{vib_delta:+.1f} mm/s",
            help="Normal: <6.0 mm/s"
        )
    
    # Real-time charts
    if show_history:
        st.subheader("📊 Real-Time Process Trends")
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Process Variables', 'Quality & Production', 'Energy & Environment', 'Equipment Health'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Process variables
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=plant_twin.data_history['kiln_temp_c'],
                name='Kiln Temp (°C)',
                line=dict(color='red', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=[temp/10 for temp in plant_twin.data_history['preheater_temp_c']],
                name='Preheater Temp (°C/10)',
                line=dict(color='orange', width=2),
                mode='lines'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Quality & Production
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=plant_twin.data_history['free_lime_pct'],
                name='Free Lime (%)',
                line=dict(color='green', width=2),
                mode='lines'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=[rate/100 for rate in plant_twin.data_history['production_rate_tph']],
                name='Production (t/h/100)',
                line=dict(color='blue', width=2),
                mode='lines'
            ),
            row=1, col=2, secondary_y=True
        )
        
        # Energy & Environment
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=plant_twin.data_history['energy_efficiency_pct'],
                name='Energy Efficiency (%)',
                line=dict(color='purple', width=2),
                mode='lines'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=[nox/10 for nox in plant_twin.data_history['nox_mg_nm3']],
                name='NOx (mg/Nm³/10)',
                line=dict(color='brown', width=2),
                mode='lines'
            ),
            row=2, col=1, secondary_y=True
        )
        
        # Equipment health
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=plant_twin.data_history['vibration_mm_s'],
                name='Vibration (mm/s)',
                line=dict(color='red', width=2),
                mode='lines'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=plant_twin.time_history,
                y=[power*2 for power in plant_twin.data_history['power_consumption_mw']],
                name='Power (MW*2)',
                line=dict(color='navy', width=2),
                mode='lines'
            ),
            row=2, col=2, secondary_y=True
        )
        
        fig.update_layout(
            title="Real-Time Plant Process Trends (Last 100 Minutes)",
            showlegend=True,
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Equipment status
    if show_equipment:
        st.subheader("⚙️ Equipment Status Dashboard")
        
        eq_cols = st.columns(len(plant_twin.equipment_status))
        
        for i, (equipment, status) in enumerate(plant_twin.equipment_status.items()):
            with eq_cols[i]:
                # Determine status color
                if status['efficiency'] > 90:
                    status_color = "#4CAF50"
                elif status['efficiency'] > 80:
                    status_color = "#FF9800"
                else:
                    status_color = "#F44336"
                
                # Equipment status card
                st.markdown(f"""
                <div class="equipment-card">
                    <h4>{equipment.replace('_', ' ').title()}</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {status_color};">
                        {status['efficiency']:.1f}%
                    </div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                        Status: <span style="color: {status_color};">{status['status']}</span><br>
                        Maintenance: {status['maintenance_due']:.0f} days
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Two-column layout for alerts and recommendations
    col_left, col_right = st.columns(2)
    
    with col_left:
        if show_anomalies and plant_twin.anomalies:
            st.subheader("🚨 Live Anomaly Alerts")
            
            for anomaly in reversed(plant_twin.anomalies[-5:]):  # Show last 5
                severity_color = {"High": "#F44336", "Medium": "#FF9800", "Low": "#4CAF50"}[anomaly['severity']]
                
                st.markdown(f"""
                <div class="anomaly-alert" style="border-left-color: {severity_color};">
                    <strong>{anomaly['type']} - {anomaly['severity']}</strong><br>
                    {anomaly['message']}<br>
                    <small style="color: #666;">{anomaly['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        elif show_anomalies:
            st.subheader("✅ No Active Anomalies")
            st.success("All systems operating within normal parameters")
    
    with col_right:
        if show_recommendations and plant_twin.ai_recommendations:
            st.subheader("🤖 AI Recommendations")
            
            for rec in reversed(plant_twin.ai_recommendations[-3:]):  # Show last 3
                priority_color = {"High": "#F44336", "Medium": "#FF9800", "Low": "#4CAF50"}[rec['priority']]
                
                st.markdown(f"""
                <div class="recommendation-card" style="border-left-color: {priority_color};">
                    <strong>{rec['type']} - {rec['priority']} Priority</strong><br>
                    <strong>Action:</strong> {rec['action']}<br>
                    <strong>Benefit:</strong> {rec['expected_benefit']}<br>
                    <strong>Confidence:</strong> {rec['confidence']:.0%}<br>
                    <small style="color: #666;">{rec['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
        elif show_recommendations:
            st.subheader("🤖 AI Monitoring Active")
            st.info("AI system continuously monitoring plant performance...")
    
    # Plant schematic visualization
    st.subheader("🏭 Live Plant Schematic")
    
    # Create plant schematic with real-time values
    fig_schematic = go.Figure()
    
    # Kiln (rectangle) - color based on temperature
    kiln_color = "red" if plant_twin.plant_state['kiln_temp_c'] > 1465 else "orange"
    fig_schematic.add_shape(
        type="rect",
        x0=2, y0=2, x1=8, y1=4,
        fillcolor=kiln_color,
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # Raw Mill (circle) - color based on vibration
    raw_mill_color = "red" if plant_twin.plant_state['vibration_mm_s'] > 6.5 else "lightgreen"
    fig_schematic.add_shape(
        type="circle",
        x0=0.5, y0=0.5, x1=1.5, y1=1.5,
        fillcolor=raw_mill_color,
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # Cement Mill (circle)
    fig_schematic.add_shape(
        type="circle",
        x0=8.5, y0=0.5, x1=9.5, y1=1.5,
        fillcolor="lightblue",
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # Preheater (rectangle)
    fig_schematic.add_shape(
        type="rect",
        x0=2, y0=4.5, x1=4, y1=6,
        fillcolor="orange",
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # Cooler (rectangle)
    cooler_color = "lightcoral" if plant_twin.plant_state['cooler_temp_c'] > 105 else "lightcyan"
    fig_schematic.add_shape(
        type="rect",
        x0=6, y0=0.5, x1=8, y1=1.5,
        fillcolor=cooler_color,
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # ID Fan (circle)
    fig_schematic.add_shape(
        type="circle",
        x0=4.5, y0=5.5, x1=5.5, y1=6.5,
        fillcolor="lightgray",
        opacity=0.7,
        line=dict(width=2, color="black")
    )
    
    # Add annotations with live values
    annotations = [
        dict(x=5, y=3, text=f"KILN<br>{plant_twin.plant_state['kiln_temp_c']:.0f}°C", 
             showarrow=False, font=dict(size=12, color="white")),
        dict(x=1, y=1, text=f"RAW MILL<br>{plant_twin.plant_state['vibration_mm_s']:.1f} mm/s", 
             showarrow=False, font=dict(size=10)),
        dict(x=9, y=1, text=f"CEMENT<br>MILL", 
             showarrow=False, font=dict(size=10)),
        dict(x=3, y=5.25, text=f"PREHEATER<br>{plant_twin.plant_state['preheater_temp_c']:.0f}°C", 
             showarrow=False, font=dict(size=10)),
        dict(x=7, y=1, text=f"COOLER<br>{plant_twin.plant_state['cooler_temp_c']:.0f}°C", 
             showarrow=False, font=dict(size=10)),
        dict(x=5, y=6, text=f"ID FAN<br>{plant_twin.plant_state['o2_percent']:.1f}% O₂", 
             showarrow=False, font=dict(size=10))
    ]
    
    fig_schematic.update_layout(
        title="Live Plant Schematic with Real-time Values",
        xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, 7], showgrid=False, showticklabels=False),
        annotations=annotations,
        height=400,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig_schematic, use_container_width=True)
    
    # Additional process parameters
    st.subheader("🔧 Detailed Process Parameters")
    
    param_cols = st.columns(4)
    
    with param_cols[0]:
        st.metric("Feed Rate", f"{plant_twin.plant_state['feed_rate_tph']:.1f} t/h")
        st.metric("Fuel Rate", f"{plant_twin.plant_state['fuel_rate_tph']:.1f} t/h")
    
    with param_cols[1]:
        st.metric("Kiln Speed", f"{plant_twin.plant_state['kiln_speed_rpm']:.2f} rpm")
        st.metric("Oxygen", f"{plant_twin.plant_state['o2_percent']:.1f}%")
    
    with param_cols[2]:
        st.metric("Power Consumption", f"{plant_twin.plant_state['power_consumption_mw']:.1f} MW")
        st.metric("NOx Emissions", f"{plant_twin.plant_state['nox_mg_nm3']:.0f} mg/Nm³")
    
    with param_cols[3]:
        st.metric("Preheater Temp", f"{plant_twin.plant_state['preheater_temp_c']:.0f}°C")
        st.metric("Cooler Temp", f"{plant_twin.plant_state['cooler_temp_c']:.0f}°C")
    
    # Auto-update mechanism
    if auto_update:
        # Update plant state
        plant_twin.update_plant_state()
        
        # Auto-refresh the page
        time.sleep(update_interval)
        st.rerun()

if __name__ == "__main__":
    launch_dynamic_plant_twin()
