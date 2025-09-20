import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta

class UtilityOptimizer:
    """
    Comprehensive utility optimization for compressed air, water, and material handling
    """
    
    def __init__(self):
        # Utility systems configuration
        self.utility_systems = {
            'compressed_air': {
                'total_capacity_cfm': 25000,
                'current_demand_cfm': 18500,
                'pressure_setpoint_bar': 7.5,
                'compressor_efficiency_pct': 82,
                'power_consumption_kw': 1850,
                'leak_rate_pct': 12,
                'maintenance_cost_monthly': 85000
            },
            'process_water': {
                'total_capacity_m3_h': 450,
                'current_consumption_m3_h': 320,
                'recirculation_rate_pct': 65,
                'treatment_efficiency_pct': 88,
                'power_consumption_kw': 280,
                'chemical_cost_monthly': 125000,
                'maintenance_cost_monthly': 45000
            },
            'cooling_water': {
                'total_capacity_m3_h': 2200,
                'current_consumption_m3_h': 1850,
                'cooling_efficiency_pct': 78,
                'power_consumption_kw': 420,
                'chemical_cost_monthly': 65000,
                'evaporation_loss_pct': 8
            },
            'material_handling': {
                'conveyors_total_length_m': 15000,
                'active_conveyors': 28,
                'total_power_consumption_kw': 2200,
                'efficiency_pct': 75,
                'maintenance_cost_monthly': 180000,
                'downtime_hours_monthly': 48
            }
        }
        
        # Optimization targets
        self.optimization_targets = {
            'compressed_air': {
                'power_reduction_pct': 15,
                'leak_reduction_pct': 8,
                'pressure_optimization_bar': 0.5
            },
            'process_water': {
                'consumption_reduction_pct': 12,
                'recirculation_increase_pct': 10,
                'treatment_efficiency_increase_pct': 5
            },
            'cooling_water': {
                'efficiency_improvement_pct': 8,
                'consumption_reduction_pct': 10,
                'chemical_reduction_pct': 15
            },
            'material_handling': {
                'power_reduction_pct': 12,
                'efficiency_improvement_pct': 10,
                'downtime_reduction_pct': 25
            }
        }
        
        # Cost configuration for optimization calculations
        self.cost_config = {
            'electricity_rate_per_kwh': 6.0,  # ₹6/kWh
            'water_rate_per_m3': 25.0,  # ₹25/m³
            'chemical_cost_reduction_factor': 0.15,  # 15% reduction
            'maintenance_cost_reduction_factor': 0.20,  # 20% reduction
            'pressure_optimization_savings_per_bar_day': 50  # ₹50/bar/day
        }
        
        # Generate historical data
        self._generate_historical_data()
    
    def _generate_historical_data(self):
        """Generate synthetic historical utility data"""
        
        # Generate 30 days of hourly data
        hours = 30 * 24
        base_time = datetime.now() - timedelta(hours=hours)
        
        self.historical_data = {
            'timestamps': [base_time + timedelta(hours=i) for i in range(hours)],
            'compressed_air': {
                'demand_cfm': [18500 + np.sin(i/6) * 2000 + random.uniform(-500, 500) for i in range(hours)],
                'pressure_bar': [7.5 + random.uniform(-0.2, 0.3) for i in range(hours)],
                'power_kw': [1850 + random.uniform(-100, 150) for i in range(hours)],
                'efficiency_pct': [82 + random.uniform(-3, 2) for i in range(hours)]
            },
            'process_water': {
                'consumption_m3_h': [320 + np.sin(i/8) * 40 + random.uniform(-20, 20) for i in range(hours)],
                'recirculation_pct': [65 + random.uniform(-5, 5) for i in range(hours)],
                'power_kw': [280 + random.uniform(-20, 30) for i in range(hours)]
            },
            'cooling_water': {
                'consumption_m3_h': [1850 + np.sin(i/12) * 150 + random.uniform(-50, 50) for i in range(hours)],
                'efficiency_pct': [78 + random.uniform(-4, 3) for i in range(hours)],
                'power_kw': [420 + random.uniform(-30, 40) for i in range(hours)]
            },
            'material_handling': {
                'active_conveyors': [28 + random.choice([-2, -1, 0, 0, 0, 1, 2]) for i in range(hours)],
                'total_power_kw': [2200 + random.uniform(-150, 200) for i in range(hours)],
                'efficiency_pct': [75 + random.uniform(-5, 3) for i in range(hours)]
            }
        }
    
    def calculate_optimization_potential(self) -> Dict:
        """Calculate optimization potential across all utilities"""
        
        optimization_results = {}
        
        for utility, current_data in self.utility_systems.items():
            targets = self.optimization_targets[utility]
            
            # Calculate potential savings
            if utility == 'compressed_air':
                power_savings = current_data['power_consumption_kw'] * targets['power_reduction_pct'] / 100
                cost_savings_monthly = power_savings * 24 * 30 * self.cost_config['electricity_rate_per_kwh']
                
                optimization_results[utility] = {
                    'power_savings_kw': power_savings,
                    'cost_savings_monthly': cost_savings_monthly,
                    'leak_reduction_cfm': current_data['current_demand_cfm'] * targets['leak_reduction_pct'] / 100,
                    'pressure_optimization_savings': targets['pressure_optimization_bar'] * self.cost_config['pressure_optimization_savings_per_bar_day'] * 30,
                    'total_monthly_savings': cost_savings_monthly + (targets['pressure_optimization_bar'] * self.cost_config['pressure_optimization_savings_per_bar_day'] * 30)
                }            
            elif utility == 'process_water':
                water_savings = current_data['current_consumption_m3_h'] * targets['consumption_reduction_pct'] / 100
                power_savings = current_data['power_consumption_kw'] * (targets['consumption_reduction_pct'] + 5) / 100
                
                optimization_results[utility] = {
                    'water_savings_m3_h': water_savings,
                    'power_savings_kw': power_savings,
                    'chemical_savings_monthly': current_data['chemical_cost_monthly'] * self.cost_config['chemical_cost_reduction_factor'],
                    'total_monthly_savings': (water_savings * 24 * 30 * self.cost_config['water_rate_per_m3']) + (power_savings * 24 * 30 * self.cost_config['electricity_rate_per_kwh']) + (current_data['chemical_cost_monthly'] * self.cost_config['chemical_cost_reduction_factor'])
                }
            
            elif utility == 'cooling_water':
                water_savings = current_data['current_consumption_m3_h'] * targets['consumption_reduction_pct'] / 100
                power_savings = current_data['power_consumption_kw'] * targets['efficiency_improvement_pct'] / 100
                
                optimization_results[utility] = {
                    'water_savings_m3_h': water_savings,
                    'power_savings_kw': power_savings,
                    'chemical_savings_monthly': current_data['chemical_cost_monthly'] * targets['chemical_reduction_pct'] / 100,
                    'total_monthly_savings': (water_savings * 24 * 30 * self.cost_config['water_rate_per_m3']) + (power_savings * 24 * 30 * self.cost_config['electricity_rate_per_kwh']) + (current_data['chemical_cost_monthly'] * targets['chemical_reduction_pct'] / 100)
                }
            
            elif utility == 'material_handling':
                power_savings = current_data['total_power_consumption_kw'] * targets['power_reduction_pct'] / 100
                maintenance_savings = current_data['maintenance_cost_monthly'] * self.cost_config['maintenance_cost_reduction_factor']
                downtime_savings = 48 * targets['downtime_reduction_pct'] / 100 * 15000  # ₹15,000/hour downtime cost
                
                optimization_results[utility] = {
                    'power_savings_kw': power_savings,
                    'maintenance_savings_monthly': maintenance_savings,
                    'downtime_reduction_hours': 48 * targets['downtime_reduction_pct'] / 100,
                    'downtime_savings_monthly': downtime_savings,
                    'total_monthly_savings': (power_savings * 24 * 30 * 6) + maintenance_savings + downtime_savings
                }
        
        return optimization_results
    
    def generate_optimization_recommendations(self, utility: str) -> List[Dict]:
        """Generate specific optimization recommendations for a utility"""
        
        recommendations_db = {
            'compressed_air': [
                {
                    'title': 'Implement Variable Speed Drive (VSD) Control',
                    'description': 'Install VSD on main compressors to match supply with demand',
                    'savings_potential': '10-15% power reduction',
                    'investment_required': '₹25-35 lakhs',
                    'payback_period': '18-24 months',
                    'priority': 'High'
                },
                {
                    'title': 'Compressed Air Leak Detection & Repair',
                    'description': 'Systematic leak detection using ultrasonic equipment',
                    'savings_potential': '8-12% demand reduction',
                    'investment_required': '₹3-5 lakhs',
                    'payback_period': '6-9 months',
                    'priority': 'High'
                },
                {
                    'title': 'Pressure Optimization Program',
                    'description': 'Reduce system pressure by 0.5 bar through zoning and local boosters',
                    'savings_potential': '5-8% power reduction',
                    'investment_required': '₹8-12 lakhs',
                    'payback_period': '12-18 months',
                    'priority': 'Medium'
                },
                {
                    'title': 'Heat Recovery from Compressors',
                    'description': 'Recover waste heat for space heating or process applications',
                    'savings_potential': '₹2-3 lakhs/year',
                    'investment_required': '₹15-20 lakhs',
                    'payback_period': '6-8 years',
                    'priority': 'Low'
                }
            ],
            
            'process_water': [
                {
                    'title': 'Advanced Water Recirculation System',
                    'description': 'Implement closed-loop cooling with advanced filtration',
                    'savings_potential': '15-20% water consumption reduction',
                    'investment_required': '₹40-55 lakhs',
                    'payback_period': '24-30 months',
                    'priority': 'High'
                },
                {
                    'title': 'Real-time Water Quality Monitoring',
                    'description': 'Install online sensors for pH, conductivity, and turbidity',
                    'savings_potential': '10-15% chemical cost reduction',
                    'investment_required': '₹12-18 lakhs',
                    'payback_period': '18-24 months',
                    'priority': 'Medium'
                },
                {
                    'title': 'Rainwater Harvesting System',
                    'description': 'Capture and treat rainwater for process applications',
                    'savings_potential': '5-8% fresh water reduction',
                    'investment_required': '₹20-30 lakhs',
                    'payback_period': '36-48 months',
                    'priority': 'Medium'
                }
            ],
            
            'cooling_water': [
                {
                    'title': 'Cooling Tower Efficiency Improvement',
                    'description': 'Install high-efficiency fill and drift eliminators',
                    'savings_potential': '8-12% cooling efficiency improvement',
                    'investment_required': '₹18-25 lakhs',
                    'payback_period': '20-28 months',
                    'priority': 'High'
                },
                {
                    'title': 'Automated Chemical Dosing System',
                    'description': 'Implement automated control for water treatment chemicals',
                    'savings_potential': '15-20% chemical cost reduction',
                    'investment_required': '₹8-12 lakhs',
                    'payback_period': '12-18 months',
                    'priority': 'High'
                },
                {
                    'title': 'Variable Speed Drive for Cooling Tower Fans',
                    'description': 'Install VSD control based on cooling demand',
                    'savings_potential': '10-15% fan power reduction',
                    'investment_required': '₹15-20 lakhs',
                    'payback_period': '18-24 months',
                    'priority': 'Medium'
                }
            ],
            
            'material_handling': [
                {
                    'title': 'Conveyor Belt Condition Monitoring',
                    'description': 'Install sensors for belt tension, alignment, and wear monitoring',
                    'savings_potential': '20-25% maintenance cost reduction',
                    'investment_required': '₹30-40 lakhs',
                    'payback_period': '18-24 months',
                    'priority': 'High'
                },
                {
                    'title': 'Energy-Efficient Motor Replacement',
                    'description': 'Replace standard motors with IE3/IE4 efficiency motors',
                    'savings_potential': '8-12% motor power reduction',
                    'investment_required': '₹45-60 lakhs',
                    'payback_period': '30-36 months',
                    'priority': 'Medium'
                },
                {
                    'title': 'Automated Material Flow Optimization',
                    'description': 'Implement AI-based conveyor speed and routing optimization',
                    'savings_potential': '10-15% overall system efficiency',
                    'investment_required': '₹25-35 lakhs',
                    'payback_period': '24-30 months',
                    'priority': 'Medium'
                }
            ]
        }
        
        return recommendations_db.get(utility, [])

def launch_utility_optimization_demo():
    """Launch Utility Optimization Dashboard"""
    
    st.set_page_config(
        page_title="💧 Utility Optimization Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💧 Utility Optimization Dashboard")
    st.markdown("**Comprehensive utility management and optimization platform**")
    
    # Initialize optimizer
    if 'utility_optimizer' not in st.session_state:
        st.session_state.utility_optimizer = UtilityOptimizer()
    
    optimizer = st.session_state.utility_optimizer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Utility Selection")
        
        selected_utility = st.selectbox("Select Utility System", [
            "compressed_air",
            "process_water", 
            "cooling_water",
            "material_handling"
        ], format_func=lambda x: x.replace('_', ' ').title())
        
        st.header("📊 Analysis Period")
        
        analysis_period = st.selectbox("Time Period", [
            "Last 24 Hours",
            "Last 7 Days", 
            "Last 30 Days"
        ])
        
        st.header("🎯 Optimization Focus")
        
        optimization_focus = st.multiselect("Optimization Areas", [
            "Power Consumption",
            "Operational Efficiency",
            "Cost Reduction",
            "Environmental Impact"
        ], default=["Power Consumption", "Cost Reduction"])
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Optimization", "💡 Recommendations", "📈 ROI Analysis"])
    
    with tab1:
        st.subheader("🏭 Utility Systems Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_power = sum([data['power_consumption_kw'] for data in optimizer.utility_systems.values() if 'power_consumption_kw' in data])
        total_monthly_cost = total_power * 24 * 30 * 6 / 1000  # ₹6/kWh converted to thousands
        
        with col1:
            st.metric("Total Power Consumption", f"{total_power:,.0f} kW")
        with col2:
            st.metric("Monthly Power Cost", f"₹{total_monthly_cost:,.0f}k")
        with col3:
            st.metric("Active Systems", "4")
        with col4:
            st.metric("Overall Efficiency", "78%", delta="-2%")
        
        # Individual utility status
        st.subheader("⚙️ Individual Utility Status")
        
        utility_status = []
        for utility_id, data in optimizer.utility_systems.items():
            
            # Calculate efficiency status
            if utility_id == 'compressed_air':
                efficiency = data['compressor_efficiency_pct']
                current_value = f"{data['current_demand_cfm']:,.0f} CFM"
            elif utility_id in ['process_water', 'cooling_water']:
                efficiency = data.get('cooling_efficiency_pct', data.get('treatment_efficiency_pct', 80))
                current_value = f"{data['current_consumption_m3_h']:,.0f} m³/h"
            else:  # material_handling
                efficiency = data['efficiency_pct']
                current_value = f"{data['active_conveyors']} conveyors"
            
            status_color = "🟢" if efficiency > 85 else "🟡" if efficiency > 75 else "🔴"
            
            utility_status.append({
                'System': utility_id.replace('_', ' ').title(),
                'Status': status_color,
                'Current Load': current_value,
                'Efficiency': f"{efficiency:.1f}%",
                'Power': f"{data.get('power_consumption_kw', 0):,.0f} kW",
                'Monthly Cost': f"₹{data.get('power_consumption_kw', 0) * 24 * 30 * 6 / 1000:,.0f}k"
            })
        
        status_df = pd.DataFrame(utility_status)
        st.dataframe(status_df, use_container_width=True)
        
        # Historical trends
        st.subheader("📈 Historical Trends")
        
        # Create multi-subplot chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Compressed Air Demand', 'Water Consumption', 'Power Consumption', 'System Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Compressed air demand
        fig.add_trace(
            go.Scatter(
                x=optimizer.historical_data['timestamps'][-168:],  # Last 7 days
                y=optimizer.historical_data['compressed_air']['demand_cfm'][-168:],
                name='Air Demand (CFM)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Water consumption
        fig.add_trace(
            go.Scatter(
                x=optimizer.historical_data['timestamps'][-168:],
                y=optimizer.historical_data['process_water']['consumption_m3_h'][-168:],
                name='Water Consumption (m³/h)',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=2
        )
        
        # Power consumption
        total_power_history = []
        for i in range(168):
            total = (optimizer.historical_data['compressed_air']['power_kw'][-168:][i] +
                    optimizer.historical_data['process_water']['power_kw'][-168:][i] +
                    optimizer.historical_data['cooling_water']['power_kw'][-168:][i] +
                    optimizer.historical_data['material_handling']['total_power_kw'][-168:][i])
            total_power_history.append(total)
        
        fig.add_trace(
            go.Scatter(
                x=optimizer.historical_data['timestamps'][-168:],
                y=total_power_history,
                name='Total Power (kW)',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # System efficiency
        fig.add_trace(
            go.Scatter(
                x=optimizer.historical_data['timestamps'][-168:],
                y=optimizer.historical_data['compressed_air']['efficiency_pct'][-168:],
                name='Air System Efficiency (%)',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="7-Day Utility Performance Trends",
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"🎯 {selected_utility.replace('_', ' ').title()} Optimization")
        
        # Current vs Optimized comparison
        current_data = optimizer.utility_systems[selected_utility]
        optimization_results = optimizer.calculate_optimization_potential()
        selected_results = optimization_results[selected_utility]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Current Performance**")
            
            if selected_utility == 'compressed_air':
                st.metric("Power Consumption", f"{current_data['power_consumption_kw']:,.0f} kW")
                st.metric("System Efficiency", f"{current_data['compressor_efficiency_pct']:.1f}%")
                st.metric("Leak Rate", f"{current_data['leak_rate_pct']:.1f}%")
                st.metric("Monthly Cost", f"₹{current_data['power_consumption_kw'] * 24 * 30 * 6 / 1000:,.0f}k")
            
            elif selected_utility in ['process_water', 'cooling_water']:
                consumption_key = 'current_consumption_m3_h'
                st.metric("Water Consumption", f"{current_data[consumption_key]:,.0f} m³/h")
                st.metric("Power Consumption", f"{current_data['power_consumption_kw']:,.0f} kW")
                if 'recirculation_rate_pct' in current_data:
                    st.metric("Recirculation Rate", f"{current_data['recirculation_rate_pct']:.1f}%")
                st.metric("Monthly Cost", f"₹{(current_data['power_consumption_kw'] * 24 * 30 * 6 + current_data.get('chemical_cost_monthly', 0)) / 1000:,.0f}k")
            
            else:  # material_handling
                st.metric("Total Power", f"{current_data['total_power_consumption_kw']:,.0f} kW")
                st.metric("System Efficiency", f"{current_data['efficiency_pct']:.1f}%")
                st.metric("Monthly Downtime", f"{current_data['downtime_hours_monthly']:.0f} hours")
                st.metric("Monthly Cost", f"₹{(current_data['total_power_consumption_kw'] * 24 * 30 * 6 + current_data['maintenance_cost_monthly']) / 1000:,.0f}k")
        
        with col2:
            st.markdown("**🎯 Optimization Potential**")
            
            if 'power_savings_kw' in selected_results:
                st.metric(
                    "Power Savings", 
                    f"{selected_results['power_savings_kw']:,.0f} kW",
                    delta=f"-{selected_results['power_savings_kw']/current_data.get('power_consumption_kw', current_data.get('total_power_consumption_kw', 1))*100:.1f}%"
                )
            
            if 'water_savings_m3_h' in selected_results:
                st.metric(
                    "Water Savings",
                    f"{selected_results['water_savings_m3_h']:,.0f} m³/h",
                    delta=f"-{selected_results['water_savings_m3_h']/current_data['current_consumption_m3_h']*100:.1f}%"
                )
            
            if 'leak_reduction_cfm' in selected_results:
                st.metric(
                    "Leak Reduction",
                    f"{selected_results['leak_reduction_cfm']:,.0f} CFM",
                    delta=f"-{selected_results['leak_reduction_cfm']/current_data['current_demand_cfm']*100:.1f}%"
                )
            
            st.metric(
                "Monthly Savings",
                f"₹{selected_results['total_monthly_savings']/1000:,.0f}k",
                delta="Optimization Potential"
            )
        
        # Optimization scenario simulation
        st.subheader("🔬 Optimization Scenario Simulation")
        
        scenario_col1, scenario_col2 = st.columns([1, 2])
        
        with scenario_col1:
            st.markdown("**Scenario Parameters**")
            
            if selected_utility == 'compressed_air':
                pressure_reduction = st.slider("Pressure Reduction (bar)", 0.0, 1.0, 0.5)
                leak_reduction = st.slider("Leak Reduction (%)", 0, 20, 8)
                efficiency_improvement = st.slider("Compressor Efficiency Improvement (%)", 0, 15, 10)
            
            elif selected_utility == 'process_water':
                consumption_reduction = st.slider("Consumption Reduction (%)", 0, 25, 12)
                recirculation_increase = st.slider("Recirculation Increase (%)", 0, 20, 10)
                chemical_reduction = st.slider("Chemical Usage Reduction (%)", 0, 30, 15)
            
            elif selected_utility == 'cooling_water':
                efficiency_improvement = st.slider("Cooling Efficiency Improvement (%)", 0, 15, 8)
                consumption_reduction = st.slider("Water Consumption Reduction (%)", 0, 20, 10)
                chemical_reduction = st.slider("Chemical Usage Reduction (%)", 0, 25, 15)
            
            else:  # material_handling
                power_reduction = st.slider("Power Consumption Reduction (%)", 0, 20, 12)
                efficiency_improvement = st.slider("System Efficiency Improvement (%)", 0, 15, 10)
                downtime_reduction = st.slider("Downtime Reduction (%)", 0, 40, 25)
        
        with scenario_col2:
            # Calculate scenario impact
            if selected_utility == 'compressed_air':
                power_savings = current_data['power_consumption_kw'] * efficiency_improvement / 100
                demand_savings = current_data['current_demand_cfm'] * leak_reduction / 100
                pressure_savings = pressure_reduction * 200  # kW per bar
                
                scenario_results = {
                    'Power Savings (kW)': power_savings + pressure_savings,
                    'Demand Reduction (CFM)': demand_savings,
                    'Monthly Cost Savings (₹)': (power_savings + pressure_savings) * 24 * 30 * 6,
                    'Annual CO₂ Reduction (tons)': (power_savings + pressure_savings) * 24 * 365 * 0.82 / 1000
                }
            
            elif selected_utility == 'process_water':
                water_savings = current_data['current_consumption_m3_h'] * consumption_reduction / 100
                power_savings = current_data['power_consumption_kw'] * consumption_reduction / 100
                chemical_savings = current_data['chemical_cost_monthly'] * chemical_reduction / 100
                
                scenario_results = {
                    'Water Savings (m³/day)': water_savings * 24,
                    'Power Savings (kW)': power_savings,
                    'Monthly Cost Savings (₹)': (water_savings * 24 * 30 * 25) + (power_savings * 24 * 30 * 6) + chemical_savings,
                    'Annual Water Savings (ML)': water_savings * 24 * 365 / 1000
                }
            
            elif selected_utility == 'cooling_water':
                water_savings = current_data['current_consumption_m3_h'] * consumption_reduction / 100
                power_savings = current_data['power_consumption_kw'] * efficiency_improvement / 100
                chemical_savings = current_data['chemical_cost_monthly'] * chemical_reduction / 100
                
                scenario_results = {
                    'Water Savings (m³/day)': water_savings * 24,
                    'Power Savings (kW)': power_savings,
                    'Monthly Cost Savings (₹)': (water_savings * 24 * 30 * 25) + (power_savings * 24 * 30 * 6) + chemical_savings,
                    'Cooling Efficiency Gain (%)': efficiency_improvement
                }
            
            else:  # material_handling
                power_savings = current_data['total_power_consumption_kw'] * power_reduction / 100
                maintenance_savings = current_data['maintenance_cost_monthly'] * 0.20
                downtime_savings = current_data['downtime_hours_monthly'] * downtime_reduction / 100 * 15000
                
                scenario_results = {
                    'Power Savings (kW)': power_savings,
                    'Downtime Reduction (hours/month)': current_data['downtime_hours_monthly'] * downtime_reduction / 100,
                    'Monthly Cost Savings (₹)': (power_savings * 24 * 30 * 6) + maintenance_savings + downtime_savings,
                    'Efficiency Improvement (%)': efficiency_improvement
                }
            
            # Display scenario results
            st.markdown("**📈 Scenario Impact Analysis**")
            
            for metric, value in scenario_results.items():
                if 'Savings' in metric or 'Reduction' in metric:
                    st.metric(metric, f"{value:,.0f}", delta="Improvement")
                else:
                    st.metric(metric, f"{value:.1f}")
    
    with tab3:
        st.subheader(f"💡 Optimization Recommendations - {selected_utility.replace('_', ' ').title()}")
        
        recommendations = optimizer.generate_optimization_recommendations(selected_utility)
        
        for i, rec in enumerate(recommendations):
            
            priority_colors = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }
            
            with st.expander(f"{priority_colors[rec['priority']]} {rec['title']} ({rec['priority']} Priority)", expanded=i < 2):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Savings Potential:** {rec['savings_potential']}")
                    st.markdown(f"**Investment Required:** {rec['investment_required']}")
                    st.markdown(f"**Payback Period:** {rec['payback_period']}")
                
                with col2:
                    if st.button(f"📋 Create Action Plan", key=f"action_{i}"):
                        st.success(f"✅ Action plan created for: {rec['title']}")
                        st.info("📅 Assigned to: Maintenance Team\n🗓️ Target Date: Next planned shutdown")
                    
                    if st.button(f"💰 Detailed ROI", key=f"roi_{i}"):
                        st.info(f"💡 ROI Analysis:\n• Investment: {rec['investment_required']}\n• Annual Savings: {rec['savings_potential']}\n• Payback: {rec['payback_period']}")
    
    with tab4:
        st.subheader("📈 ROI Analysis - All Utilities")
        
        # Calculate comprehensive ROI
        optimization_results = optimizer.calculate_optimization_potential()
        
        # Summary ROI metrics
        total_monthly_savings = sum([result['total_monthly_savings'] for result in optimization_results.values()])
        total_annual_savings = total_monthly_savings * 12
        estimated_investment = 15000000  # ₹1.5 crores total investment
        payback_years = estimated_investment / total_annual_savings
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Annual Savings", f"₹{total_annual_savings/10000000:.1f} Cr")
        with col2:
            st.metric("Investment Required", f"₹{estimated_investment/10000000:.1f} Cr")
        with col3:
            st.metric("Payback Period", f"{payback_years:.1f} years")
        with col4:
            st.metric("10-Year NPV", f"₹{(total_annual_savings * 8 - estimated_investment)/10000000:.1f} Cr")
        
        # ROI breakdown by utility
        st.subheader("💰 ROI Breakdown by Utility")
        
        roi_data = []
        for utility, results in optimization_results.items():
            annual_savings = results['total_monthly_savings'] * 12
            estimated_utility_investment = estimated_investment * (annual_savings / total_annual_savings)
            utility_payback = estimated_utility_investment / annual_savings if annual_savings > 0 else float('inf')            
            roi_data.append({
                'Utility System': utility.replace('_', ' ').title(),
                'Annual Savings (₹L)': f"{annual_savings/100000:.1f}",
                'Investment (₹L)': f"{estimated_utility_investment/100000:.1f}",
                'Payback (years)': f"{utility_payback:.1f}",
                'ROI (%)': f"{(annual_savings/estimated_utility_investment - 1)*100:.1f}" if estimated_utility_investment > 0 else "N/A"
            })
        
        roi_df = pd.DataFrame(roi_data)
        st.dataframe(roi_df, use_container_width=True)
        
        # ROI visualization
        utility_names = [utility.replace('_', ' ').title() for utility in optimization_results.keys()]
        annual_savings = [results['total_monthly_savings'] * 12 / 100000 for results in optimization_results.values()]  # in lakhs
        
        fig = px.bar(
            x=utility_names,
            y=annual_savings,
            title="Annual Savings Potential by Utility (₹ Lakhs)",
            color=annual_savings,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Implementation roadmap
        st.subheader("🗓️ Implementation Roadmap")
        
        roadmap = [
            {"Phase": "Phase 1 (Months 1-3)", "Focus": "Quick Wins", "Utilities": "Compressed Air Leaks, Water Monitoring", "Investment": "₹15L", "Savings": "₹8L/month"},
            {"Phase": "Phase 2 (Months 4-6)", "Focus": "Efficiency Systems", "Utilities": "VSD Installation, Process Optimization", "Investment": "₹45L", "Savings": "₹18L/month"},
            {"Phase": "Phase 3 (Months 7-12)", "Focus": "Advanced Systems", "Utilities": "Full Automation, AI Integration", "Investment": "₹90L", "Savings": "₹35L/month"},
            {"Phase": "Phase 4 (Year 2+)", "Focus": "Continuous Optimization", "Utilities": "Performance Monitoring, Upgrades", "Investment": "₹25L/year", "Savings": "₹45L/month"}
        ]
        
        roadmap_df = pd.DataFrame(roadmap)
        st.dataframe(roadmap_df, use_container_width=True)

if __name__ == "__main__":
    launch_utility_optimization_demo()
