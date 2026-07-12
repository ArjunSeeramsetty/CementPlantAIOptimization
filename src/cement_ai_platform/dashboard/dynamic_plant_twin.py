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

from cement_ai_platform.simulation.kiln_gym_env import KilnCoolerGymEnv
from cement_ai_platform.models.agents.unified_kiln_cooler_controller import UnifiedKilnCoolerController

class DynamicPlantTwin:
    """
    Real-time dynamic plant digital twin for live POC demonstration
    Uses parallel physics-informed simulator runs to compare heuristic PID vs SOTA RL.
    """
    
    def __init__(self):
        self.controller = UnifiedKilnCoolerController()
        
        # Initialize parallel environments
        self.baseline_env = KilnCoolerGymEnv()
        self.rl_env = KilnCoolerGymEnv()
        
        # Reset environments
        self.baseline_obs = self.baseline_env.reset()
        self.rl_obs = self.rl_env.reset()
        
        # Default plant_state (will be bound dynamically)
        self.plant_state = {
            'kiln_temp_c': 1450.0,
            'free_lime_pct': 1.2,
            'feed_rate_tph': 200.0,
            'fuel_rate_tph': 15.0,
            'kiln_speed_rpm': 3.0,
            'o2_percent': 3.2,
            'production_rate_tph': 196.0,
            'energy_efficiency_pct': 92.0,
            'nox_mg_nm3': 500.0,
            'preheater_temp_c': 550.0,
            'cooler_temp_c': 95.0,
            'vibration_mm_s': 4.5,
            'power_consumption_mw': 54.0
        }
        
        # Historical data for trends
        self.history_length = 100
        self.time_history = []
        
        # Histories for both controllers
        self.baseline_history = {
            'kiln_temp_c': [],
            'free_lime_pct': [],
            'feed_rate_tph': [],
            'fuel_rate_tph': [],
            'kiln_speed_rpm': [],
            'nox_mg_nm3': [],
            'preheater_temp_c': [],
            'production_rate_tph': [],
            'energy_efficiency_pct': [],
            'vibration_mm_s': [],
            'power_consumption_mw': []
        }
        
        self.rl_history = {
            'kiln_temp_c': [],
            'free_lime_pct': [],
            'feed_rate_tph': [],
            'fuel_rate_tph': [],
            'kiln_speed_rpm': [],
            'nox_mg_nm3': [],
            'preheater_temp_c': [],
            'production_rate_tph': [],
            'energy_efficiency_pct': [],
            'vibration_mm_s': [],
            'power_consumption_mw': []
        }
        
        # Cumulative savings tracking
        self.cumulative_savings = {
            'fuel_tons_saved': 0.0,
            'co2_tons_avoided': 0.0,
            'cost_usd_saved': 0.0
        }
        
        # Plant equipment status
        self.equipment_status = {
            'kiln': {'status': 'Running', 'efficiency': 95.0, 'maintenance_due': 45},
            'raw_mill': {'status': 'Running', 'efficiency': 88.0, 'maintenance_due': 12},
            'cement_mill': {'status': 'Running', 'efficiency': 92.0, 'maintenance_due': 30},
            'id_fan': {'status': 'Running', 'efficiency': 87.0, 'maintenance_due': 8},
            'cooler': {'status': 'Running', 'efficiency': 91.0, 'maintenance_due': 22}
        }
        
        # Anomaly flags and AI recommendations
        self.anomalies = []
        self.ai_recommendations = []
        
        # Initialize history
        self._initialize_history()

    def _initialize_history(self):
        """Initialize historical data by pre-running both environments"""
        base_time = datetime.now() - timedelta(minutes=self.history_length)
        
        # Ensure reset
        self.baseline_obs = self.baseline_env.reset()
        self.rl_obs = self.rl_env.reset()
        
        for i in range(self.history_length):
            timestamp = base_time + timedelta(minutes=i)
            self.time_history.append(timestamp)
            
            # Step environments
            self._step_simulation_run(use_rl=False)
            self._step_simulation_run(use_rl=True)
            
            # Save step data to histories
            self._record_history_step(use_rl=False)
            self._record_history_step(use_rl=True)
            
            # Calculate savings history
            baseline_fuel = self.baseline_history['fuel_rate_tph'][-1] / 60.0
            rl_fuel = self.rl_history['fuel_rate_tph'][-1] / 60.0
            fuel_saved = max(0.0, baseline_fuel - rl_fuel)
            
            self.cumulative_savings['fuel_tons_saved'] += fuel_saved
            self.cumulative_savings['cost_usd_saved'] += fuel_saved * 120.0
            self.cumulative_savings['co2_tons_avoided'] += fuel_saved * 2.42

        # Bind initial active state
        self.bind_active_state(use_rl=False)

    def _step_simulation_run(self, use_rl: bool):
        """Execute one step in the simulator using either baseline or RL control setpoints"""
        env = self.rl_env if use_rl else self.baseline_env
        obs = self.rl_obs if use_rl else self.baseline_obs
        
        # Map flat observation array to sensor data dictionary
        sensor_data = {
            'burning_zone_temp_c': float(obs[0]),
            'fuel_rate_tph': float(obs[1]),
            'kiln_speed_rpm': float(obs[2]),
            'feed_rate_tph': float(obs[3]),
            'free_lime_percent': float(obs[4]),
            'nox_mg_nm3': float(obs[5]),
            'preheater_temp_c': float(obs[6]),
            # Additional keys for model compatibility
            'cooler_outlet_temp_c': 95.0,
            'cooler_air_flow_nm3_h': 150000.0,
            'gas_flow_nm3_h': 200000.0
        }
        
        # Get setpoints from controller
        output = self.controller.compute_setpoints(sensor_data, use_rl=use_rl)
        
        # Calculate adjustments (actions) to step the gym environment
        target_speed = output['kiln_setpoints']['kiln_speed_rpm']
        target_fuel = output['kiln_setpoints']['fuel_rate_tph']
        target_feed = output['kiln_setpoints']['feed_rate_tph']
        
        current_speed = float(obs[2])
        current_fuel = float(obs[1])
        current_feed = float(obs[3])
        
        speed_adj = target_speed - current_speed
        fuel_adj = target_fuel - current_fuel
        feed_adj = target_feed - current_feed
        
        action = np.array([speed_adj, fuel_adj, feed_adj], dtype=np.float32)
        
        # Step Gym environment
        next_obs, reward, done, info = env.step(action)
        
        # If the environment trips or is done, reset it
        if done:
            next_obs = env.reset()
            
        if use_rl:
            self.rl_obs = next_obs
        else:
            self.baseline_obs = next_obs

    def _record_history_step(self, use_rl: bool):
        """Log state fields into history dictionary"""
        obs = self.rl_obs if use_rl else self.baseline_obs
        history = self.rl_history if use_rl else self.baseline_history
        
        kiln_temp_c = float(obs[0])
        fuel_rate_tph = float(obs[1])
        kiln_speed_rpm = float(obs[2])
        feed_rate_tph = float(obs[3])
        free_lime_pct = float(obs[4])
        nox_mg_nm3 = float(obs[5])
        preheater_temp_c = float(obs[6])
        
        production_rate_tph = feed_rate_tph * 0.98
        energy_efficiency_pct = 92.0 - abs(free_lime_pct - 1.2) * 2.0
        vibration_mm_s = 4.5 + np.random.normal(0, 0.05)
        power_consumption_mw = production_rate_tph * 0.27
        
        history['kiln_temp_c'].append(kiln_temp_c)
        history['fuel_rate_tph'].append(fuel_rate_tph)
        history['kiln_speed_rpm'].append(kiln_speed_rpm)
        history['feed_rate_tph'].append(feed_rate_tph)
        history['free_lime_pct'].append(free_lime_pct)
        history['nox_mg_nm3'].append(nox_mg_nm3)
        history['preheater_temp_c'].append(preheater_temp_c)
        history['production_rate_tph'].append(production_rate_tph)
        history['energy_efficiency_pct'].append(energy_efficiency_pct)
        history['vibration_mm_s'].append(vibration_mm_s)
        history['power_consumption_mw'].append(power_consumption_mw)
        
        # Maintain history length
        for key in history:
            if len(history[key]) > self.history_length:
                history[key] = history[key][-self.history_length:]

    def bind_active_state(self, use_rl: bool):
        """Sync the primary plant_state dictionary to either RL or Baseline metrics"""
        obs = self.rl_obs if use_rl else self.baseline_obs
        history = self.rl_history if use_rl else self.baseline_history
        
        self.plant_state = {
            'kiln_temp_c': float(obs[0]),
            'fuel_rate_tph': float(obs[1]),
            'kiln_speed_rpm': float(obs[2]),
            'feed_rate_tph': float(obs[3]),
            'free_lime_pct': float(obs[4]),
            'nox_mg_nm3': float(obs[5]),
            'preheater_temp_c': float(obs[6]),
            'production_rate_tph': history['production_rate_tph'][-1],
            'energy_efficiency_pct': history['energy_efficiency_pct'][-1],
            'vibration_mm_s': history['vibration_mm_s'][-1],
            'power_consumption_mw': history['power_consumption_mw'][-1],
            'o2_percent': 3.2,
            'cooler_temp_c': 95.0
        }
        
        # Check for anomalies and generate recommendations based on the active state
        self._check_anomalies()
        self._generate_ai_recommendations()
        self._update_equipment_status()

    def update_plant_state(self):
        """Update plant state with realistic dynamics and correlations"""
        current_time = datetime.now()
        
        # Step both baseline and RL simulations
        self._step_simulation_run(use_rl=False)
        self._step_simulation_run(use_rl=True)
        
        # Record histories
        self._record_history_step(use_rl=False)
        self._record_history_step(use_rl=True)
        
        # Append time
        self.time_history.append(current_time)
        if len(self.time_history) > self.history_length:
            self.time_history = self.time_history[-self.history_length:]
            
        # Calculate cumulative savings:
        # Save dt = 1 minute = 1/60 hours
        baseline_fuel = self.baseline_history['fuel_rate_tph'][-1] / 60.0
        rl_fuel = self.rl_history['fuel_rate_tph'][-1] / 60.0
        fuel_saved = max(0.0, baseline_fuel - rl_fuel)
        
        # Cumulative updates
        self.cumulative_savings['fuel_tons_saved'] += fuel_saved
        self.cumulative_savings['cost_usd_saved'] += fuel_saved * 120.0
        self.cumulative_savings['co2_tons_avoided'] += fuel_saved * 2.42

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
            self.equipment_status['raw_mill']['efficiency'] = max(75.0, 
                self.equipment_status['raw_mill']['efficiency'] - 1.0)
        else:
            self.equipment_status['raw_mill']['efficiency'] = min(95.0,
                self.equipment_status['raw_mill']['efficiency'] + 0.5)
        
        # Kiln efficiency affected by temperature control
        temp_deviation = abs(self.plant_state['kiln_temp_c'] - 1450.0)
        if temp_deviation > 15.0:
            self.equipment_status['kiln']['efficiency'] = max(85.0,
                self.equipment_status['kiln']['efficiency'] - 0.5)
        else:
            self.equipment_status['kiln']['efficiency'] = min(98.0,
                self.equipment_status['kiln']['efficiency'] + 0.2)
        
        # ID fan efficiency based on oxygen levels
        o2_deviation = abs(self.plant_state['o2_percent'] - 3.2)
        if o2_deviation > 0.5:
            self.equipment_status['id_fan']['efficiency'] = max(80.0,
                self.equipment_status['id_fan']['efficiency'] - 0.3)
        else:
            self.equipment_status['id_fan']['efficiency'] = min(95.0,
                self.equipment_status['id_fan']['efficiency'] + 0.1)
        
        # Cooler efficiency based on temperature
        if self.plant_state['cooler_temp_c'] > 105.0:
            self.equipment_status['cooler']['efficiency'] = max(80.0,
                self.equipment_status['cooler']['efficiency'] - 0.4)
        else:
            self.equipment_status['cooler']['efficiency'] = min(96.0,
                self.equipment_status['cooler']['efficiency'] + 0.1)
        
        # Decrease maintenance days
        for equipment in self.equipment_status:
            self.equipment_status[equipment]['maintenance_due'] -= 0.01
            if self.equipment_status[equipment]['maintenance_due'] < 0:
                self.equipment_status[equipment]['maintenance_due'] = random.randint(30, 60)

    def inject_high_temp(self):
        self.baseline_env.burning_zone_temp = 1560.0
        self.rl_env.burning_zone_temp = 1560.0
        self.baseline_obs[0] = 1560.0
        self.rl_obs[0] = 1560.0
        
    def inject_quality_issue(self):
        self.baseline_env.free_lime = 2.5
        self.rl_env.free_lime = 2.5
        self.baseline_obs[4] = 2.5
        self.rl_obs[4] = 2.5
        
    def inject_vibration_alert(self):
        self.baseline_history['vibration_mm_s'][-1] = 7.5
        self.rl_history['vibration_mm_s'][-1] = 7.5
        
    def inject_environmental_issue(self):
        self.baseline_env.nox = 750.0
        self.rl_env.nox = 750.0
        self.baseline_obs[5] = 750.0
        self.rl_obs[5] = 750.0
        
    def reset_normal(self):
        self.baseline_obs = self.baseline_env.reset()
        self.rl_obs = self.rl_env.reset()

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
        color: #1a1a1a;
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
        color: #1a1a1a;
        border-radius: 0 8px 8px 0;
    }
    
    .equipment-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1a1a1a;
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
        
        st.header("🧠 SOTA RL Control Mode")
        use_rl = st.toggle("Enable SOTA RL Control Mode", value=False, help="Enable prescriptive PyTorch RL control of the Sintering Zone.")
        
        st.header("🎯 Scenario Injection")
        st.markdown("*Inject realistic plant scenarios for demonstration*")
        
        if st.button("🔥 Inject High Temperature", help="Simulate kiln overheating"):
            plant_twin.inject_high_temp()
            plant_twin.bind_active_state(use_rl)
            st.success("High temperature scenario injected!")
        
        if st.button("⚠️ Inject Quality Issue", help="Simulate free lime elevation"):
            plant_twin.inject_quality_issue()
            plant_twin.bind_active_state(use_rl)
            st.warning("Quality issue scenario injected!")
        
        if st.button("📳 Inject Vibration Alert", help="Simulate equipment vibration"):
            plant_twin.inject_vibration_alert()
            plant_twin.bind_active_state(use_rl)
            st.error("Vibration alert scenario injected!")
        
        if st.button("🌪️ Inject Environmental Issue", help="Simulate high NOx emissions"):
            plant_twin.inject_environmental_issue()
            plant_twin.bind_active_state(use_rl)
            st.error("Environmental issue scenario injected!")
        
        if st.button("🔄 Reset to Normal", help="Reset all parameters to normal"):
            plant_twin.reset_normal()
            plant_twin.bind_active_state(use_rl)
            st.info("Plant state reset to normal!")
        
        st.header("📊 Display Options")
        show_history = st.checkbox("Show Historical Trends", value=True)
        show_equipment = st.checkbox("Show Equipment Status", value=True)
        show_anomalies = st.checkbox("Show Anomaly Alerts", value=True)
        show_recommendations = st.checkbox("Show AI Recommendations", value=True)

    # Bind metrics to selected controller mode
    plant_twin.bind_active_state(use_rl)
    
    # Main dashboard layout
    
    if use_rl:
        st.subheader("🧠 SOTA RL Prescriptive Control Active")
        
        # Check if RL agent weights were loaded
        if plant_twin.controller.rl_model_loaded:
            st.success("🔌 Custom PyTorch Actor-Critic PPO policy loaded successfully from `models/kiln_rl_actor.pt`")
        else:
            st.warning("⚠️ RL Model weights not found. Falling back to baseline PID heuristic controls.")
            
        # Render Cumulative Savings summary
        st.markdown("#### 💰 Cumulative Sustainability & Savings Benefits")
        save_col1, save_col2, save_col3 = st.columns(3)
        with save_col1:
            st.metric(
                label="🌿 Cumulative CO₂ Avoided",
                value=f"{plant_twin.cumulative_savings['co2_tons_avoided']:.2f} Tons",
                delta="Reduced carbon footprint"
            )
        with save_col2:
            st.metric(
                label="🔥 Fuel Coal Saved",
                value=f"{plant_twin.cumulative_savings['fuel_tons_saved']:.2f} Tons",
                delta="Reduced thermal energy input"
            )
        with save_col3:
            st.metric(
                label="💵 Cumulative Cost Savings",
                value=f"${plant_twin.cumulative_savings['cost_usd_saved']:.2f} USD",
                delta="Saved on fuel cost"
            )
        st.markdown("---")

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
        
        if use_rl:
            # Create subplot with multiple charts comparing Baseline vs RL
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Burning Zone Temperature (°C)', 'Free Lime Quality Index (%)', 'Thermal Fuel Input Rate (t/h)', 'NOx Emissions (mg/Nm³)')
            )
            
            # Burning Zone Temperature comparison
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.baseline_history['kiln_temp_c'],
                    name='Baseline Temp',
                    line=dict(color='rgba(255, 0, 0, 0.4)', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.rl_history['kiln_temp_c'],
                    name='RL Temp',
                    line=dict(color='red', width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Free Lime comparison
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.baseline_history['free_lime_pct'],
                    name='Baseline Free Lime',
                    line=dict(color='rgba(0, 128, 0, 0.4)', width=2, dash='dash'),
                    mode='lines'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.rl_history['free_lime_pct'],
                    name='RL Free Lime',
                    line=dict(color='green', width=2),
                    mode='lines'
                ),
                row=1, col=2
            )
            
            # Fuel Rate comparison
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.baseline_history['fuel_rate_tph'],
                    name='Baseline Fuel',
                    line=dict(color='rgba(255, 165, 0, 0.4)', width=2, dash='dash'),
                    mode='lines'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.rl_history['fuel_rate_tph'],
                    name='RL Fuel',
                    line=dict(color='orange', width=2),
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # NOx comparison
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.baseline_history['nox_mg_nm3'],
                    name='Baseline NOx',
                    line=dict(color='rgba(128, 0, 128, 0.4)', width=2, dash='dash'),
                    mode='lines'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=plant_twin.rl_history['nox_mg_nm3'],
                    name='RL NOx',
                    line=dict(color='purple', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
            
            # Add horizontal target lines
            # Target Temp = 1450 C
            fig.add_shape(type="line", x0=plant_twin.time_history[0], y0=1450, x1=plant_twin.time_history[-1], y1=1450,
                          line=dict(color="black", width=1, dash="dot"), row=1, col=1)
            # Target Free Lime = 1.2%
            fig.add_shape(type="line", x0=plant_twin.time_history[0], y0=1.2, x1=plant_twin.time_history[-1], y1=1.2,
                          line=dict(color="black", width=1, dash="dot"), row=1, col=2)
            # Target NOx = 520
            fig.add_shape(type="line", x0=plant_twin.time_history[0], y0=520, x1=plant_twin.time_history[-1], y1=520,
                          line=dict(color="black", width=1, dash="dot"), row=2, col=2)
            
            fig.update_layout(
                title="Parallel Comparative Run: Baseline PID vs SOTA RL Optimization Agent",
                showlegend=True,
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Original charts (showing only baseline history)
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
                    y=plant_twin.baseline_history['kiln_temp_c'],
                    name='Kiln Temp (°C)',
                    line=dict(color='red', width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=[temp/10 for temp in plant_twin.baseline_history['preheater_temp_c']],
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
                    y=plant_twin.baseline_history['free_lime_pct'],
                    name='Free Lime (%)',
                    line=dict(color='green', width=2),
                    mode='lines'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=[rate/100 for rate in plant_twin.baseline_history['production_rate_tph']],
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
                    y=plant_twin.baseline_history['energy_efficiency_pct'],
                    name='Energy Efficiency (%)',
                    line=dict(color='purple', width=2),
                    mode='lines'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=[nox/10 for nox in plant_twin.baseline_history['nox_mg_nm3']],
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
                    y=plant_twin.baseline_history['vibration_mm_s'],
                    name='Vibration (mm/s)',
                    line=dict(color='red', width=2),
                    mode='lines'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=plant_twin.time_history,
                    y=[power*2 for power in plant_twin.baseline_history['power_consumption_mw']],
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
