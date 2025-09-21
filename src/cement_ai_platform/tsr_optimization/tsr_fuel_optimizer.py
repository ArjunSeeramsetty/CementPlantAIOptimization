# src/cement_ai_platform/tsr_optimization/tsr_fuel_optimizer.py
"""
TSR & Alternative Fuels Optimization Agent
Advanced thermal substitution rate optimization for cement plants
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import random
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class FuelMix:
    fuel_id: str
    fuel_name: str
    composition: Dict[str, float]  # moisture, ash, volatile_matter, fixed_carbon, calorific_value
    cost_per_ton: float
    availability_tph: float
    carbon_factor: float
    environmental_impact: float

class TSRFuelOptimizer:
    """
    Advanced TSR (Thermal Substitution Rate) and Alternative Fuel Optimization
    """
    
    def __init__(self):
        # Define available fuel types with realistic properties
        self.fuel_database = {
            'coal': FuelMix(
                fuel_id='coal',
                fuel_name='Coal (Primary)',
                composition={
                    'moisture': 8.5,
                    'ash': 12.2,
                    'volatile_matter': 35.8,
                    'fixed_carbon': 43.5,
                    'calorific_value': 5800  # kcal/kg
                },
                cost_per_ton=5500,
                availability_tph=25.0,
                carbon_factor=1.0,
                environmental_impact=1.0
            ),
            'petcoke': FuelMix(
                fuel_id='petcoke',
                fuel_name='Pet Coke',
                composition={
                    'moisture': 2.1,
                    'ash': 0.8,
                    'volatile_matter': 12.5,
                    'fixed_carbon': 84.6,
                    'calorific_value': 7950
                },
                cost_per_ton=6200,
                availability_tph=8.0,
                carbon_factor=1.15,
                environmental_impact=1.2
            ),
            'rdf_msw': FuelMix(
                fuel_id='rdf_msw',
                fuel_name='RDF from MSW',
                composition={
                    'moisture': 15.2,
                    'ash': 18.5,
                    'volatile_matter': 62.8,
                    'fixed_carbon': 3.5,
                    'calorific_value': 3800
                },
                cost_per_ton=2800,
                availability_tph=12.0,
                carbon_factor=0.1,  # Near carbon neutral
                environmental_impact=0.3
            ),
            'biomass': FuelMix(
                fuel_id='biomass',
                fuel_name='Biomass (Agricultural Waste)',
                composition={
                    'moisture': 12.8,
                    'ash': 8.2,
                    'volatile_matter': 75.5,
                    'fixed_carbon': 3.5,
                    'calorific_value': 4200
                },
                cost_per_ton=3200,
                availability_tph=6.0,
                carbon_factor=0.05,  # Carbon neutral
                environmental_impact=0.2
            ),
            'plastic_waste': FuelMix(
                fuel_id='plastic_waste',
                fuel_name='Plastic Waste (Processed)',
                composition={
                    'moisture': 3.5,
                    'ash': 8.8,
                    'volatile_matter': 85.2,
                    'fixed_carbon': 2.5,
                    'calorific_value': 6500
                },
                cost_per_ton=1800,
                availability_tph=4.0,
                carbon_factor=0.8,
                environmental_impact=0.6
            ),
            'tire_derived': FuelMix(
                fuel_id='tire_derived',
                fuel_name='Tire Derived Fuel',
                composition={
                    'moisture': 1.2,
                    'ash': 6.5,
                    'volatile_matter': 65.8,
                    'fixed_carbon': 26.5,
                    'calorific_value': 7200
                },
                cost_per_ton=2500,
                availability_tph=3.0,
                carbon_factor=0.9,
                environmental_impact=0.7
            )
        }
        
        # Current plant constraints
        self.plant_constraints = {
            'total_thermal_requirement_kcal_h': 45000000,  # 45 GCal/h
            'max_alt_fuel_percentage': 40,  # Maximum 40% TSR
            'min_flame_temperature_c': 1850,
            'max_ash_input_percentage': 15,
            'max_chlorine_input_ppm': 500,
            'max_sulfur_input_percentage': 2.0
        }
        
        # Current operational state
        self.current_state = {
            'kiln_thermal_load_kcal_h': 42500000,
            'calciner_thermal_load_kcal_h': 28000000,
            'current_tsr_percentage': 18.5,
            'clinker_production_tph': 185,
            'burning_zone_temp_c': 1465,
            'preheater_o2_percentage': 3.2,
            'calciner_temp_c': 890
        }
    
    def calculate_optimal_fuel_mix(self, target_tsr: float, 
                                 environmental_priority: bool = False) -> Dict:
        """Calculate optimal fuel mix for target TSR"""
        
        total_thermal_req = self.plant_constraints['total_thermal_requirement_kcal_h']
        alt_fuel_thermal = total_thermal_req * (target_tsr / 100)
        fossil_fuel_thermal = total_thermal_req - alt_fuel_thermal
        
        # Optimization algorithm (simplified)
        fuel_combinations = []
        
        # Generate multiple fuel mix scenarios
        for scenario in range(50):
            mix = self._generate_fuel_scenario(alt_fuel_thermal, fossil_fuel_thermal, 
                                             environmental_priority)
            if self._validate_fuel_mix(mix):
                fuel_combinations.append(mix)
        
        # Check if any valid combinations were found
        if not fuel_combinations:
            # Return a default/fallback mix or raise an appropriate exception
            return {
                'error': 'No valid fuel mix found for target TSR',
                'fuel_rates_tph': {},
                'achieved_tsr_percentage': 0,
                'total_cost_per_hour': 0,
                'total_carbon_impact': 0
            }
        
        # Select optimal based on cost or environmental impact
        if environmental_priority:
            optimal_mix = min(fuel_combinations, 
                            key=lambda x: x['total_carbon_impact'])
        else:
            optimal_mix = min(fuel_combinations, 
                            key=lambda x: x['total_cost_per_hour'])
        
        return optimal_mix
    
    def _generate_fuel_scenario(self, alt_thermal: float, fossil_thermal: float,
                              env_priority: bool) -> Dict:
        """Generate a random fuel mix scenario"""
        
        scenario = {
            'fuel_rates_tph': {},
            'thermal_contributions_kcal_h': {},
            'total_cost_per_hour': 0,
            'total_carbon_impact': 0,
            'ash_content_percentage': 0,
            'achieved_tsr_percentage': 0
        }
        
        # Allocate alternative fuels
        remaining_alt_thermal = alt_thermal
        alt_fuels = ['rdf_msw', 'biomass', 'plastic_waste', 'tire_derived']
        
        for fuel_id in alt_fuels:
            if remaining_alt_thermal <= 0:
                break
                
            fuel_data = self.fuel_database[fuel_id]
            max_possible_rate = min(
                fuel_data.availability_tph,
                remaining_alt_thermal / fuel_data.composition['calorific_value']
            )
            
            # Random allocation with bias toward environmental fuels if env_priority
            if env_priority:
                allocation_factor = random.uniform(0.6, 1.0) if fuel_data.carbon_factor < 0.5 else random.uniform(0.1, 0.4)
            else:
                allocation_factor = random.uniform(0.2, 0.8)
            
            fuel_rate = max_possible_rate * allocation_factor
            thermal_contrib = fuel_rate * fuel_data.composition['calorific_value']
            
            scenario['fuel_rates_tph'][fuel_id] = fuel_rate
            scenario['thermal_contributions_kcal_h'][fuel_id] = thermal_contrib
            scenario['total_cost_per_hour'] += fuel_rate * fuel_data.cost_per_ton / 24
            scenario['total_carbon_impact'] += thermal_contrib * fuel_data.carbon_factor / 1000000
            
            remaining_alt_thermal -= thermal_contrib
        
        # Allocate fossil fuels for remaining thermal requirement
        remaining_fossil_thermal = fossil_thermal + max(0, remaining_alt_thermal)
        fossil_fuels = ['coal', 'petcoke']
        
        for fuel_id in fossil_fuels:
            fuel_data = self.fuel_database[fuel_id]
            allocation = random.uniform(0.3, 0.7) if fuel_id == 'coal' else random.uniform(0.2, 0.5)
            
            # Calculate maximum possible rate based on availability
            max_possible_rate = fuel_data.availability_tph
            calculated_rate = (remaining_fossil_thermal * allocation) / fuel_data.composition['calorific_value']
            
            # Use the minimum of calculated rate and availability
            fuel_rate = min(calculated_rate, max_possible_rate)
            thermal_contrib = fuel_rate * fuel_data.composition['calorific_value']
            
            scenario['fuel_rates_tph'][fuel_id] = fuel_rate
            scenario['thermal_contributions_kcal_h'][fuel_id] = thermal_contrib
            scenario['total_cost_per_hour'] += fuel_rate * fuel_data.cost_per_ton / 24
            scenario['total_carbon_impact'] += thermal_contrib * fuel_data.carbon_factor / 1000000
            
            remaining_fossil_thermal -= thermal_contrib
        
        # Calculate achieved TSR
        total_alt_thermal = sum([v for k, v in scenario['thermal_contributions_kcal_h'].items() 
                               if k in alt_fuels])
        total_thermal = sum(scenario['thermal_contributions_kcal_h'].values())
        scenario['achieved_tsr_percentage'] = (total_alt_thermal / total_thermal) * 100 if total_thermal > 0 else 0
        
        # Calculate weighted average ash content
        total_fuel_mass = sum(scenario['fuel_rates_tph'].values())
        if total_fuel_mass > 0:
            weighted_ash = sum(fuel_rate * self.fuel_database[fuel_id].composition['ash'] 
                             for fuel_id, fuel_rate in scenario['fuel_rates_tph'].items()) / total_fuel_mass
            scenario['ash_content_percentage'] = weighted_ash
        else:
            scenario['ash_content_percentage'] = 0
        
        return scenario
    
    def _validate_fuel_mix(self, mix: Dict) -> bool:
        """Validate if fuel mix meets plant constraints"""
        
        # Check TSR limits
        if mix['achieved_tsr_percentage'] > self.plant_constraints['max_alt_fuel_percentage']:
            return False
        
        # Check ash content
        if mix['ash_content_percentage'] > self.plant_constraints['max_ash_input_percentage']:
            return False
        
        # Check total fuel rates are within availability
        for fuel_id, rate in mix['fuel_rates_tph'].items():
            if rate > self.fuel_database[fuel_id].availability_tph:
                return False
        
        return True
    
    def simulate_tsr_impact(self, fuel_mix: Dict) -> Dict:
        """Simulate the impact of TSR change on plant performance using physics-based models"""
        
        tsr = fuel_mix['achieved_tsr_percentage']
        
        # Calculate impacts based on empirical models and cement plant physics
        impact = {
            # Temperature effects: Higher TSR reduces flame temperature due to lower calorific value
            'kiln_burning_zone_temp_change_c': -0.3 * max(0, tsr - 20),  # -0.3°C per % TSR above 20%
            
            # Preheater temperature increases due to higher volatile content in alternative fuels
            'preheater_exit_temp_change_c': 0.4 * tsr,  # +0.4°C per % TSR
            
            # Calciner stability decreases with higher TSR due to fuel variability
            'calciner_temp_stability': max(0.75, 1.0 - (tsr - 15) * 0.015),  # Degrades above 15% TSR
            
            # NOx emissions reduction due to lower flame temperature and fuel nitrogen content
            'nox_emissions_change_pct': -0.8 * tsr,  # -0.8% per % TSR
            
            # SOx emissions reduction due to lower sulfur content in alternative fuels
            'sox_emissions_change_pct': -1.2 * tsr,  # -1.2% per % TSR
            
            # CO2 emissions reduction proportional to TSR (carbon-neutral alternative fuels)
            'co2_emissions_reduction_pct': 0.6 * tsr,  # +0.6% reduction per % TSR
            
            # Specific heat consumption: slight increase due to higher moisture content
            'specific_heat_consumption_change_pct': 0.05 * tsr,  # +0.05% per % TSR
            
            # Free lime variation increases with TSR due to fuel variability
            'free_lime_variation_ppm': 3.0 * max(0, tsr - 20),  # +3 ppm per % TSR above 20%
            
            # Quality stability decreases with higher TSR due to fuel inconsistency
            'clinker_quality_stability': max(0.88, 1.0 - (tsr - 20) * 0.006),  # Degrades above 20% TSR
            
            # Fuel cost savings (negative value means savings)
            'fuel_cost_savings_per_hour': (fuel_mix['total_cost_per_hour'] - 15000) * -1,  # vs baseline
            
            # Maintenance impact: slight increase due to higher ash content and fuel variability
            'maintenance_impact_factor': 1.0 + (tsr - 15) * 0.0015  # +0.15% per % TSR above 15%
        }
        
        return impact

def launch_tsr_fuel_optimizer_demo():
    """Launch TSR & Alternative Fuels Optimization Dashboard"""
    
    # Note: st.set_page_config() is called in the main unified dashboard
    
    st.title("🔥 TSR & Alternative Fuels Optimization")
    st.markdown("**Intelligent Alternative Fuel Mix Optimization for Maximum TSR**")
    
    # Initialize optimizer
    if 'tsr_optimizer' not in st.session_state:
        st.session_state.tsr_optimizer = TSRFuelOptimizer()
    
    optimizer = st.session_state.tsr_optimizer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Optimization Parameters")
        
        target_tsr = st.slider("Target TSR (%)", 10, 40, 28)
        environmental_priority = st.checkbox("Environmental Priority", value=False)
        
        st.header("🔥 Available Fuels")
        
        for fuel_id, fuel_data in optimizer.fuel_database.items():
            if fuel_id not in ['coal', 'petcoke']:  # Show alternative fuels
                available = st.checkbox(
                    f"{fuel_data.fuel_name}", 
                    value=True, 
                    key=f"fuel_{fuel_id}"
                )
                if available:
                    st.write(f"  📊 {fuel_data.composition['calorific_value']} kcal/kg")
                    st.write(f"  💰 ₹{fuel_data.cost_per_ton}/ton")
        
        st.header("🏭 Plant Constraints")
        max_tsr = st.slider("Max TSR Limit (%)", 25, 45, 40)
        max_ash = st.slider("Max Ash Input (%)", 10, 20, 15)
    
    # Main optimization interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Fuel Mix Optimization")
        
        if st.button("🚀 Optimize Fuel Mix"):
            
            with st.spinner("Optimizing fuel mix for target TSR..."):
                
                # Run optimization
                optimal_mix = optimizer.calculate_optimal_fuel_mix(
                    target_tsr, environmental_priority
                )
                
                # Check if optimization was successful
                if 'error' in optimal_mix:
                    # Handle error case
                    st.session_state.optimization_error = optimal_mix
                    st.session_state.optimal_mix = None
                    st.session_state.tsr_impact = None
                else:
                    # Simulate impact for successful optimization
                    impact = optimizer.simulate_tsr_impact(optimal_mix)
                    
                    # Store in session state
                    st.session_state.optimal_mix = optimal_mix
                    st.session_state.tsr_impact = impact
                    st.session_state.optimization_error = None
            
            # Display appropriate message
            if 'optimization_error' in st.session_state and st.session_state.optimization_error:
                st.error(f"❌ {st.session_state.optimization_error['error']}")
            else:
                st.success(f"✅ Optimized fuel mix for {optimal_mix['achieved_tsr_percentage']:.1f}% TSR")
    
    with col2:
        st.subheader("📊 Current Status")
        
        current_tsr = optimizer.current_state['current_tsr_percentage']
        st.metric("Current TSR", f"{current_tsr:.1f}%")
        st.metric("Target TSR", f"{target_tsr:.1f}%", delta=f"+{target_tsr - current_tsr:.1f}%")
        st.metric("Clinker Production", f"{optimizer.current_state['clinker_production_tph']:.0f} t/h")
    
    # Display optimization results
    if 'optimal_mix' in st.session_state:
        
        optimal_mix = st.session_state.optimal_mix
        impact = st.session_state.tsr_impact
        
        # Check if optimization was successful
        if optimal_mix is None or 'error' in optimal_mix:
            st.error(f"❌ {optimal_mix.get('error', 'Optimization failed - no valid fuel mix found')}")
            st.info("💡 Try adjusting the target TSR or environmental priority settings")
            return
        
        # Fuel mix breakdown
        st.subheader("🔥 Optimized Fuel Mix")
        
        # Create fuel mix chart
        fuel_names = []
        fuel_rates = []
        fuel_colors = []
        
        color_map = {
            'coal': '#2F2F2F',
            'petcoke': '#4F4F4F', 
            'rdf_msw': '#4CAF50',
            'biomass': '#8BC34A',
            'plastic_waste': '#FF9800',
            'tire_derived': '#9C27B0'
        }
        
        for fuel_id, rate in optimal_mix['fuel_rates_tph'].items():
            if rate > 0:
                fuel_names.append(optimizer.fuel_database[fuel_id].fuel_name)
                fuel_rates.append(rate)
                fuel_colors.append(color_map.get(fuel_id, '#666666'))
        
        fig = go.Figure(data=[go.Pie(
            labels=fuel_names, 
            values=fuel_rates,
            marker_colors=fuel_colors,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Rate: %{value:.1f} t/h<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f"Fuel Mix Distribution (TSR: {optimal_mix['achieved_tsr_percentage']:.1f}%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact analysis
        st.subheader("📈 TSR Impact Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CO₂ Reduction", 
                f"{impact['co2_emissions_reduction_pct']:.1f}%",
                delta="Environmental Benefit"
            )
        
        with col2:
            st.metric(
                "Fuel Cost Savings", 
                f"₹{impact['fuel_cost_savings_per_hour']:,.0f}/hr",
                delta="vs Current Mix"
            )
        
        with col3:
            temp_change = impact['kiln_burning_zone_temp_change_c']
            st.metric(
                "Kiln Temp Impact", 
                f"{temp_change:+.1f}°C",
                delta="Requires Adjustment" if abs(temp_change) > 5 else "Within Range"
            )
        
        with col4:
            quality_stability = impact['clinker_quality_stability']
            st.metric(
                "Quality Stability", 
                f"{quality_stability:.1%}",
                delta="Good" if quality_stability > 0.95 else "Monitor"
            )
        
        # Detailed impact breakdown
        st.subheader("🔍 Detailed Impact Assessment")
        
        impact_data = [
            {"Parameter": "Kiln Burning Zone Temperature", "Change": f"{impact['kiln_burning_zone_temp_change_c']:+.1f}°C", "Status": "Monitor" if abs(impact['kiln_burning_zone_temp_change_c']) > 5 else "OK"},
            {"Parameter": "Preheater Exit Temperature", "Change": f"{impact['preheater_exit_temp_change_c']:+.1f}°C", "Status": "Expected"},
            {"Parameter": "NOx Emissions", "Change": f"{impact['nox_emissions_change_pct']:+.1f}%", "Status": "Improved"},
            {"Parameter": "SOx Emissions", "Change": f"{impact['sox_emissions_change_pct']:+.1f}%", "Status": "Improved"},
            {"Parameter": "Specific Heat Consumption", "Change": f"{impact['specific_heat_consumption_change_pct']:+.1f}%", "Status": "Monitor"},
            {"Parameter": "Free Lime Variation", "Change": f"{impact['free_lime_variation_ppm']:+.0f} ppm", "Status": "Monitor" if impact['free_lime_variation_ppm'] > 100 else "OK"}
        ]
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True)
        
        # Fuel properties comparison
        st.subheader("⚗️ Fuel Properties Analysis")
        
        properties_data = []
        if optimal_mix and 'fuel_rates_tph' in optimal_mix:
            for fuel_id, rate in optimal_mix['fuel_rates_tph'].items():
                if rate > 0:
                    fuel_data = optimizer.fuel_database[fuel_id]
                    properties_data.append({
                        'Fuel': fuel_data.fuel_name,
                        'Rate (t/h)': f"{rate:.1f}",
                        'Calorific Value (kcal/kg)': fuel_data.composition['calorific_value'],
                        'Ash Content (%)': fuel_data.composition['ash'],
                        'Moisture (%)': fuel_data.composition['moisture'],
                        'Cost (₹/ton)': fuel_data.cost_per_ton,
                        'Carbon Factor': fuel_data.carbon_factor
                    })
        
        properties_df = pd.DataFrame(properties_data)
        st.dataframe(properties_df, use_container_width=True)
        
        # Implementation recommendations
        st.subheader("🎯 Implementation Recommendations")
        
        recommendations = [
            "🔧 **Gradual Implementation**: Increase TSR by 2-3% per week to ensure process stability",
            "🌡️ **Temperature Monitoring**: Closely monitor burning zone and calciner temperatures during transition",
            "🧪 **Quality Tracking**: Increase free lime testing frequency during TSR ramp-up",
            "⚙️ **Equipment Preparation**: Ensure alternative fuel handling systems are ready",
            "👨‍💼 **Operator Training**: Train operators on alternative fuel combustion characteristics",
            "📊 **Performance Tracking**: Monitor KPIs continuously during implementation phase"
        ]
        
        for rec in recommendations:
            st.markdown(rec)

if __name__ == "__main__":
    launch_tsr_fuel_optimizer_demo()
