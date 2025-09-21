# src/cement_ai_platform/dashboard/unified_dashboard.py
"""
Unified Dashboard for JK Cement Digital Twin Platform POC
Provides a single navigation interface for all seven enhancement modules
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import each dashboard module
try:
    from cement_ai_platform.streaming.realtime_dashboard import launch_streaming_demo
    STREAMING_AVAILABLE = True
except ImportError as e:
    STREAMING_AVAILABLE = False
    print(f"Streaming dashboard not available: {e}")

try:
    from cement_ai_platform.multi_plant.multi_plant_dashboard import launch_multi_plant_demo
    MULTI_PLANT_AVAILABLE = True
except ImportError as e:
    MULTI_PLANT_AVAILABLE = False
    print(f"Multi-plant dashboard not available: {e}")

try:
    from cement_ai_platform.mobile.mobile_dashboard import launch_mobile_demo
    MOBILE_AVAILABLE = True
except ImportError as e:
    MOBILE_AVAILABLE = False
    print(f"Mobile dashboard not available: {e}")

try:
    from cement_ai_platform.maintenance.maintenance_dashboard import launch_predictive_maintenance_demo
    MAINTENANCE_AVAILABLE = True
except ImportError as e:
    MAINTENANCE_AVAILABLE = False
    print(f"Maintenance dashboard not available: {e}")

try:
    from cement_ai_platform.validation.validation_dashboard import launch_data_validation_demo
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"Validation dashboard not available: {e}")

try:
    from cement_ai_platform.dwsim.dwsim_dashboard import launch_dwsim_integration_demo
    DWSIM_AVAILABLE = True
except ImportError as e:
    DWSIM_AVAILABLE = False
    print(f"DWSIM dashboard not available: {e}")

try:
    from cement_ai_platform.dashboard.dynamic_plant_twin import launch_dynamic_plant_twin
    DYNAMIC_TWIN_AVAILABLE = True
except ImportError as e:
    DYNAMIC_TWIN_AVAILABLE = False
    print(f"Dynamic Plant Twin not available: {e}")

try:
    from cement_ai_platform.tsr_optimization.tsr_fuel_optimizer import launch_tsr_fuel_optimizer_demo
    TSR_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    TSR_OPTIMIZER_AVAILABLE = False
    print(f"TSR Optimizer not available: {e}")

try:
    from cement_ai_platform.copilot.plant_ai_assistant import launch_plant_ai_assistant
    PLANT_COPILOT_AVAILABLE = True
except ImportError as e:
    PLANT_COPILOT_AVAILABLE = False
    print(f"Plant Copilot not available: {e}")

try:
    from cement_ai_platform.utilities.utility_optimizer import launch_utility_optimization_demo
    UTILITY_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    UTILITY_OPTIMIZER_AVAILABLE = False
    print(f"Utility Optimizer not available: {e}")

try:
    from cement_ai_platform.lims.lims_integration import launch_lims_integration_demo
    LIMS_AVAILABLE = True
except ImportError as e:
    LIMS_AVAILABLE = False
    print(f"LIMS Integration not available: {e}")

try:
    from cement_ai_platform.analytics.historical_analytics import launch_historical_analytics_demo
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    print(f"Historical Analytics not available: {e}")

# New Enhanced Modules
try:
    from cement_ai_platform.config.enhanced_plant_config import EnhancedPlantConfigManager
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError as e:
    ENHANCED_CONFIG_AVAILABLE = False
    print(f"Enhanced Plant Config not available: {e}")

try:
    from cement_ai_platform.models.physics_models import DynamicProcessDataGenerator, PhysicsBasedProcessModel
    PHYSICS_MODELS_AVAILABLE = True
except ImportError as e:
    PHYSICS_MODELS_AVAILABLE = False
    print(f"Physics Models not available: {e}")

try:
    from cement_ai_platform.optimization.tsr_optimizer import AdvancedTSROptimizer, FuelOptimizationResult
    ADVANCED_TSR_AVAILABLE = True
except ImportError as e:
    ADVANCED_TSR_AVAILABLE = False
    print(f"Advanced TSR Optimizer not available: {e}")

def show_module_status():
    """Display the status of each module"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Module Status")
    
    modules = [
        ("Live Plant Twin", DYNAMIC_TWIN_AVAILABLE),
        ("🔬 Enhanced Plant Config", ENHANCED_CONFIG_AVAILABLE),
        ("⚛️ Physics-Based Models", PHYSICS_MODELS_AVAILABLE),
        ("🔥 Advanced TSR Optimizer", ADVANCED_TSR_AVAILABLE),
        ("TSR & Fuel Optimizer", TSR_OPTIMIZER_AVAILABLE),
        ("Plant AI Copilot", PLANT_COPILOT_AVAILABLE),
        ("Utility Optimization", UTILITY_OPTIMIZER_AVAILABLE),
        ("LIMS Integration", LIMS_AVAILABLE),
        ("Historical Analytics", ANALYTICS_AVAILABLE),
        ("Real-Time Streaming", STREAMING_AVAILABLE),
        ("Multi-Plant Support", MULTI_PLANT_AVAILABLE),
        ("Mobile Dashboard", MOBILE_AVAILABLE),
        ("Predictive Maintenance", MAINTENANCE_AVAILABLE),
        ("Data Validation", VALIDATION_AVAILABLE),
        ("DWSIM Integration", DWSIM_AVAILABLE)
    ]
    
    for module_name, available in modules:
        status = "✅" if available else "❌"
        st.sidebar.markdown(f"{status} {module_name}")

def show_platform_overview():
    """Display platform overview and capabilities"""
    st.title("🏭 JK Cement Digital Twin Platform - POC Dashboard")
    
    st.markdown("""
    ## 🚀 Platform Overview
    
    Welcome to the **JK Cement Digital Twin Platform** Proof of Concept (POC) demonstration.
    This unified dashboard provides access to all seven enhancement modules developed for 
    the 6-month POC program.
    
    ### 📊 Available Modules:
    """)
    
    # Create columns for module cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏭 Live Plant Twin**
        - Real-time plant simulation
        - Dynamic process visualization
        - AI-powered recommendations
        - Scenario injection for demos
        """)

        st.markdown("""
        **🔬 Enhanced Plant Config**
        - Physics-based plant configurations
        - Real JK Cement plant data (Nimbahera, Mangrol, Muddapur)
        - Multi-tenant support
        - Dynamic configuration management
        """)

        st.markdown("""
        **⚛️ Physics-Based Models**
        - Real cement chemistry calculations
        - Free lime prediction using Lea-Parker equations
        - Thermal energy optimization
        - NOx emission modeling
        """)

        st.markdown("""
        **🔥 Advanced TSR Optimizer**
        - Linear programming optimization
        - Multi-objective fuel mix optimization
        - Cost, emissions, and efficiency balancing
        - Physics-informed constraints
        """)

        st.markdown("""
        **🔥 TSR & Fuel Optimizer**
        - Alternative fuel mix optimization
        - TSR impact simulation
        - Cost-benefit analysis
        - Environmental optimization
        """)
        
        st.markdown("""
        **🤖 Plant AI Copilot**
        - Gemini-powered assistant
        - Intelligent troubleshooting
        - Optimization recommendations
        - Natural language interface
        """)
        
        st.markdown("""
        **💧 Utility Optimization**
        - Compressed air optimization
        - Water system management
        - Material handling efficiency
        - ROI analysis & recommendations
        """)
        
        st.markdown("""
        **🧪 LIMS Integration**
        - Robotic lab automation
        - Real-time quality monitoring
        - Quality prediction algorithms
        - Sample tracking & analysis
        """)
        
        st.markdown("""
        **📊 Historical Analytics**
        - Large-scale data analysis (10+ years)
        - Parameter correlation detection
        - Performance benchmarking
        - Optimization opportunities
        """)
    
    with col2:
        st.markdown("""
        **🔄 Real-Time Streaming**
        - Pub/Sub sensor data simulation
        - Live process monitoring
        - Real-time alerts and notifications
        """)
        
        st.markdown("""
        **🏭 Multi-Plant Support**
        - Enterprise-scale plant management
        - Cross-plant analytics
        - Tenant isolation and security
        """)
        
        st.markdown("""
        **📱 Mobile Dashboard**
        - Mobile-optimized interface
        - PWA capabilities
        - Push notifications
        """)
    
    with col3:
        st.markdown("""
        **🔧 Predictive Maintenance**
        - Time-to-failure models
        - Maintenance scheduling
        - Equipment health monitoring
        """)
        
        st.markdown("""
        **🔬 Data Validation**
        - Drift detection algorithms
        - Data quality assessment
        - Model retraining triggers
        """)
        
        st.markdown("""
        **⚗️ DWSIM Integration**
        - Physics-based simulation
        - Process scenario execution
        - Chemical engineering models
        """)
    
    st.markdown("---")
    
    # Show technical capabilities
    st.markdown("### 🛠️ Technical Capabilities")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Cloud Infrastructure:**
        - Google Cloud Platform (GCP)
        - Vertex AI (Gemini Pro)
        - BigQuery ML
        - Cloud Run & Kubernetes
        - Pub/Sub & Firestore
        """)
    
    with tech_col2:
        st.markdown("""
        **AI & ML Features:**
        - Advanced TSR Optimization (Linear Programming)
        - Physics-Based Process Models
        - Real Cement Chemistry Calculations
        - Multi-Objective Optimization
        - Enhanced Plant Configuration Management
        - Quality Prediction Models
        - Anomaly Detection
        - Process Control (PID)
        - Utility Optimization
        """)

def main():
    """Main unified dashboard application"""
    
    # Configure page
    st.set_page_config(
        page_title="🚀 Cement Plant POC Unified Dashboard",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🏭 Cement Plant POC")
    st.sidebar.markdown("**Unified Dashboard**")
    
    # Navigation options
    nav_options = [
        "📊 Platform Overview",
        "🏭 Live Plant Twin",
        "🔬 Enhanced Plant Config",
        "⚛️ Physics-Based Models",
        "🔥 Advanced TSR Optimizer",
        "🔥 TSR & Fuel Optimizer",
        "🤖 Plant AI Copilot",
        "💧 Utility Optimization",
        "🧪 LIMS Integration",
        "📊 Historical Analytics",
        "🔄 Real-Time Streaming",
        "🏭 Multi-Plant Support",
        "📱 Mobile Dashboard",
        "🔧 Predictive Maintenance",
        "🔬 Data Validation",
        "⚗️ DWSIM Integration"
    ]
    
    choice = st.sidebar.radio("Select Module", nav_options)
    
    # Show module status
    show_module_status()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🚀 Quick Start
    1. Select a module from the menu
    2. Explore the dashboard features
    3. Test real-time capabilities
    4. Review AI/ML predictions
    
    ### 📞 Support
    For technical support or questions about the POC, 
    contact the JK Cement AI Team.
    """)
    
    # Main content area
    if choice == "📊 Platform Overview":
        show_platform_overview()

    elif choice == "🏭 Live Plant Twin":
        if DYNAMIC_TWIN_AVAILABLE:
            launch_dynamic_plant_twin()
        else:
            st.error("❌ Live Plant Twin module is not available")
            st.info("Please ensure the dynamic plant twin module is properly installed and configured.")

    elif choice == "🔬 Enhanced Plant Config":
        if ENHANCED_CONFIG_AVAILABLE:
            launch_enhanced_plant_config_demo()
        else:
            st.error("❌ Enhanced Plant Configuration module is not available")
            st.info("Please ensure the enhanced plant config module is properly installed and configured.")

    elif choice == "⚛️ Physics-Based Models":
        if PHYSICS_MODELS_AVAILABLE:
            launch_physics_models_demo()
        else:
            st.error("❌ Physics-Based Models module is not available")
            st.info("Please ensure the physics models module is properly installed and configured.")

    elif choice == "🔥 Advanced TSR Optimizer":
        if ADVANCED_TSR_AVAILABLE:
            launch_advanced_tsr_demo()
        else:
            st.error("❌ Advanced TSR Optimizer module is not available")
            st.info("Please ensure the advanced TSR optimizer module is properly installed and configured.")
    
    elif choice == "🔥 TSR & Fuel Optimizer":
        if TSR_OPTIMIZER_AVAILABLE:
            launch_tsr_fuel_optimizer_demo()
        else:
            st.error("❌ TSR & Fuel Optimizer module is not available")
            st.info("Please ensure the TSR optimizer module is properly installed and configured.")
    
    elif choice == "🤖 Plant AI Copilot":
        if PLANT_COPILOT_AVAILABLE:
            launch_plant_ai_assistant()
        else:
            st.error("❌ Plant AI Copilot module is not available")
            st.info("Please ensure the plant AI assistant module is properly installed and configured.")
    
    elif choice == "💧 Utility Optimization":
        if UTILITY_OPTIMIZER_AVAILABLE:
            launch_utility_optimization_demo()
        else:
            st.error("❌ Utility Optimization module is not available")
            st.info("Please ensure the utility optimizer module is properly installed and configured.")
    
    elif choice == "🧪 LIMS Integration":
        if LIMS_AVAILABLE:
            launch_lims_integration_demo()
        else:
            st.error("❌ LIMS Integration module is not available")
            st.info("Please ensure the LIMS integration module is properly installed and configured.")
    
    elif choice == "📊 Historical Analytics":
        if ANALYTICS_AVAILABLE:
            launch_historical_analytics_demo()
        else:
            st.error("❌ Historical Analytics module is not available")
            st.info("Please ensure the historical analytics module is properly installed and configured.")
        
    elif choice == "🔄 Real-Time Streaming":
        if STREAMING_AVAILABLE:
            st.title("🔄 Real-Time Streaming Dashboard")
            launch_streaming_demo()
        else:
            st.error("❌ Real-Time Streaming module is not available")
            st.info("Please ensure the streaming module is properly installed and configured.")
            
    elif choice == "🏭 Multi-Plant Support":
        if MULTI_PLANT_AVAILABLE:
            st.title("🏭 Multi-Plant Support Dashboard")
            launch_multi_plant_demo()
        else:
            st.error("❌ Multi-Plant Support module is not available")
            st.info("Please ensure the multi-plant module is properly installed and configured.")
            
    elif choice == "📱 Mobile Dashboard":
        if MOBILE_AVAILABLE:
            st.title("📱 Mobile Dashboard")
            launch_mobile_demo()
        else:
            st.error("❌ Mobile Dashboard module is not available")
            st.info("Please ensure the mobile dashboard module is properly installed and configured.")
            
    elif choice == "🔧 Predictive Maintenance":
        if MAINTENANCE_AVAILABLE:
            st.title("🔧 Predictive Maintenance Dashboard")
            launch_predictive_maintenance_demo()
        else:
            st.error("❌ Predictive Maintenance module is not available")
            st.info("Please ensure the maintenance module is properly installed and configured.")
            
    elif choice == "🔬 Data Validation":
        if VALIDATION_AVAILABLE:
            st.title("🔬 Data Validation Dashboard")
            launch_data_validation_demo()
        else:
            st.error("❌ Data Validation module is not available")
            st.info("Please ensure the validation module is properly installed and configured.")
            
    elif choice == "⚗️ DWSIM Integration":
        if DWSIM_AVAILABLE:
            st.title("⚗️ DWSIM Integration Dashboard")
            launch_dwsim_integration_demo()
        else:
            st.error("❌ DWSIM Integration module is not available")
            st.info("Please ensure the DWSIM module is properly installed and configured.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏭 JK Cement Digital Twin Platform POC | Powered by AI & Cloud Technologies</p>
        <p>Version 1.0.0 | Production-Ready Enhancement Modules</p>
    </div>
    """, unsafe_allow_html=True)

def launch_enhanced_plant_config_demo():
    """Launch enhanced plant configuration demo"""
    st.title("🔬 Enhanced Plant Configuration System")
    st.markdown("**Physics-based plant configurations for JK Cement facilities**")

    try:
        # Initialize plant config manager
        config_manager = EnhancedPlantConfigManager()

        # Plant selection
        st.subheader("🏭 Select Plant Configuration")

        plants = config_manager.get_all_plants()
        plant_names = [f"{plant.plant_name} ({plant.location})" for plant in plants]
        selected_plant_idx = st.selectbox(
            "Choose Plant:",
            range(len(plants)),
            format_func=lambda x: plant_names[x]
        )

        selected_plant = plants[selected_plant_idx]

        # Display plant overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Plant Capacity", f"{selected_plant.capacity_tpd:,} TPD")
            st.metric("Technology Level", selected_plant.technology_level.title())
            st.metric("Commissioning Year", selected_plant.commissioning_year)

        with col2:
            st.metric("Thermal Energy", f"{selected_plant.energy['thermal']} kcal/kg")
            st.metric("Electrical Energy", f"{selected_plant.energy['electrical']} kWh/t")
            st.metric("NOx Emissions", f"{selected_plant.environmental['nox_mg_nm3']} mg/Nm³")

        with col3:
            st.metric("Free Lime Target", f"{selected_plant.quality['free_lime_pct']}%")
            st.metric("Kiln Temperature", f"{selected_plant.process['kiln_temperature_c']}°C")
            st.metric("Preheater Stages", selected_plant.process['preheater_stages'])

        # Process parameters
        st.subheader("⚙️ Process Parameters")
        st.json(selected_plant.process)

        # Fuel properties
        st.subheader("⛽ Fuel Mix & Properties")
        st.dataframe(pd.DataFrame(selected_plant.fuel_properties).T)

        # Quality targets
        st.subheader("🎯 Quality Targets")
        quality_df = pd.DataFrame(list(selected_plant.quality.items()),
                                columns=['Parameter', 'Target'])
        st.dataframe(quality_df)

        # Raw materials
        st.subheader("🪨 Raw Materials Composition")
        raw_materials_df = pd.DataFrame(list(selected_plant.raw_materials.items()),
                                       columns=['Material', 'kg/t'])
        st.bar_chart(raw_materials_df.set_index('Material'))

        # Environmental limits
        st.subheader("🌱 Environmental Limits")
        env_df = pd.DataFrame(list(selected_plant.environmental.items()),
                             columns=['Parameter', 'Limit'])
        st.dataframe(env_df)

        # Plant comparison
        st.subheader("📊 Multi-Plant Comparison")
        comparison_df = config_manager.get_plant_comparison_matrix()
        st.dataframe(comparison_df)

        # Export options
        st.subheader("💾 Export Configuration")
        export_format = st.selectbox("Export Format:", ["JSON", "YAML"])
        if st.button("Export Configuration"):
            config_data = config_manager.export_configurations(export_format.lower())
            st.download_button(
                label=f"Download {export_format}",
                data=config_data,
                file_name=f"jk_cement_plant_config.{export_format.lower()}",
                mime=f"text/{export_format.lower()}"
            )

    except Exception as e:
        st.error(f"❌ Error loading plant configuration: {e}")
        st.info("Please ensure all dependencies are installed correctly.")

def launch_physics_models_demo():
    """Launch physics-based process models demo"""
    st.title("⚛️ Physics-Based Process Models")
    st.markdown("**Real cement chemistry and process calculations**")

    try:
        # Initialize plant configuration
        config_manager = EnhancedPlantConfigManager()
        plants = config_manager.get_all_plants()

        # Plant selection
        plant_names = [f"{plant.plant_name} ({plant.location})" for plant in plants]
        selected_plant_idx = st.selectbox(
            "Choose Plant for Physics Models:",
            range(len(plants)),
            format_func=lambda x: plant_names[x]
        )

        selected_plant = plants[selected_plant_idx]

        # Initialize physics model
        physics_model = PhysicsBasedProcessModel(selected_plant)
        data_generator = DynamicProcessDataGenerator(selected_plant)

        # Model demonstration
        st.subheader("🔬 Physics Model Calculations")

        # Current KPIs
        st.markdown("**📊 Current Plant KPIs (Physics-Based)**")
        current_kpis = data_generator.generate_current_kpis()

        kpi_cols = st.columns(4)
        kpis_to_show = [
            ("Kiln Temperature", "kiln_temperature_c", "°C"),
            ("Free Lime", "free_lime_pct", "%"),
            ("Thermal Energy", "thermal_energy_kcal_kg", "kcal/kg"),
            ("Electrical Energy", "electrical_energy_kwh_t", "kWh/t"),
            ("OEE", "oee_percentage", "%"),
            ("Energy Efficiency", "energy_efficiency_pct", "%"),
            ("NOx Emissions", "nox_emissions_mg_nm3", "mg/Nm³"),
            ("Production Rate", "production_rate_tph", "t/h")
        ]

        for i, (name, key, unit) in enumerate(kpis_to_show):
            with kpi_cols[i % 4]:
                st.metric(f"{name}", f"{current_kpis.get(key, 0):.2f} {unit}")

        # Physics calculations demo
        st.subheader("🧮 Physics Calculations")

        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            st.markdown("**Lea-Parker Free Lime Calculation**")
            st.latex(r"F_{lime} = f(T, \tau, LSF) = LSF \times \exp\left(-\frac{6000}{T}\right) \times \frac{\tau}{45}")

            # Manual calculation inputs
            temp_input = st.number_input("Kiln Temperature (°C)", 1400, 1500, 1450)
            residence_time_input = st.number_input("Residence Time (min)", 30, 60, 45)
            lsf_input = st.number_input("LSF Value", 90, 100, 95)

            calculated_free_lime = physics_model.calculate_free_lime(
                temp_input,
                residence_time_input,
                {'limestone': 1200, 'clay': 200}
            )

            st.metric("Calculated Free Lime", f"{calculated_free_lime:.2f}%")

        with calc_col2:
            st.markdown("**Cement Strength Prediction (Bogue)**")
            st.latex(r"Strength_{28d} = 0.65 \times C_3S + 0.25 \times C_2S + 0.10 \times C_3A")

            c3s_input = st.slider("C3S Content (%)", 50, 70, 62)
            c2s_input = st.slider("C2S Content (%)", 10, 20, 15)
            c3a_input = st.slider("C3A Content (%)", 5, 12, 8)
            fineness_input = st.slider("Fineness (cm²/g)", 3000, 4000, 3450)

            calculated_strength = physics_model.calculate_cement_strength(
                {'c3s_content_pct': c3s_input, 'c2s_content_pct': c2s_input, 'c3a_content_pct': c3a_input},
                fineness_input
            )

            st.metric("Predicted Strength (28d)", f"{calculated_strength:.1f} MPa")

        # Process scenarios
        st.subheader("🎯 Process Scenarios")

        scenario_type = st.selectbox(
            "Select Scenario:",
            ["high_temperature", "low_temperature", "high_feed_rate", "quality_issue", "fuel_efficiency"]
        )

        if st.button("Run Scenario Simulation"):
            with st.spinner("Running physics simulation..."):
                scenario_data = data_generator.simulate_scenario(scenario_type, 8)

                st.markdown(f"**{scenario_type.replace('_', ' ').title()} Scenario Results**")

                # Display key metrics
                final_metrics = scenario_data.iloc[-1]
                scenario_cols = st.columns(4)

                metrics_to_show = [
                    ("Free Lime", "free_lime_pct", "%"),
                    ("Thermal Energy", "thermal_energy_kcal_kg", "kcal/kg"),
                    ("NOx Emissions", "nox_emissions_mg_nm3", "mg/Nm³"),
                    ("Energy Efficiency", "energy_efficiency_pct", "%")
                ]

                for i, (name, key, unit) in enumerate(metrics_to_show):
                    with scenario_cols[i]:
                        value = final_metrics.get(key, 0)
                        st.metric(name, f"{value:.2f} {unit}")

                # Plot scenario trends
                st.markdown("**📈 Scenario Trends**")
                fig = px.line(scenario_data, x='timestamp', y=['free_lime_pct', 'thermal_energy_kcal_kg', 'nox_emissions_mg_nm3'],
                            title=f"{scenario_type.replace('_', ' ').title()} Scenario")
                st.plotly_chart(fig, use_container_width=True)

        # Historical data generation
        st.subheader("📊 Generate Historical Data")

        hours = st.slider("Historical Data Period (hours)", 1, 168, 24)  # 1 hour to 1 week

        if st.button("Generate Historical KPIs"):
            with st.spinner("Generating physics-based historical data..."):
                historical_data = data_generator.generate_historical_data(hours)

                st.markdown(f"**📈 Generated {hours} Hours of Historical Data**")

                # Display summary statistics
                summary_stats = historical_data.describe()
                st.dataframe(summary_stats)

                # Plot historical trends
                st.markdown("**📊 Historical Trends**")
                trend_fig = px.line(historical_data, x='timestamp',
                                  y=['kiln_temperature_c', 'free_lime_pct', 'thermal_energy_kcal_kg', 'oee_percentage'],
                                  title="Historical Process Trends")
                st.plotly_chart(trend_fig, use_container_width=True)

                # Download historical data
                csv_data = historical_data.to_csv(index=False)
                st.download_button(
                    label="Download Historical Data (CSV)",
                    data=csv_data,
                    file_name=f"physics_based_historical_data_{hours}h.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error in physics models demo: {e}")
        st.info("Please ensure all dependencies are installed correctly.")

def launch_advanced_tsr_demo():
    """Launch advanced TSR optimization demo"""
    st.title("🔥 Advanced TSR Optimization Engine")
    st.markdown("**Multi-objective fuel optimization with physics constraints**")

    try:
        # Initialize plant configuration
        config_manager = EnhancedPlantConfigManager()
        plants = config_manager.get_all_plants()

        # Plant selection
        plant_names = [f"{plant.plant_name} ({plant.location})" for plant in plants]
        selected_plant_idx = st.selectbox(
            "Choose Plant for TSR Optimization:",
            range(len(plants)),
            format_func=lambda x: plant_names[x]
        )

        selected_plant = plants[selected_plant_idx]

        # Initialize TSR optimizer
        tsr_optimizer = AdvancedTSROptimizer(selected_plant)

        # Optimization parameters
        st.subheader("🎯 Optimization Parameters")

        col1, col2 = st.columns(2)

        with col1:
            target_tsr = st.slider("Target TSR (%)", 15, 35, 25)
            cost_weight = st.slider("Cost Weight", 0.0, 1.0, 0.4)

        with col2:
            emissions_weight = st.slider("Emissions Weight", 0.0, 1.0, 0.3)
            energy_weight = st.slider("Energy Weight", 0.0, 1.0, 0.2)

        # Constraints
        st.subheader("🔒 Operational Constraints")

        constraints = {}
        constraint_cols = st.columns(2)

        with constraint_cols[0]:
            max_ash = st.number_input("Max Ash Content (%)", 10.0, 20.0, 15.0)
            constraints['max_ash_content_pct'] = max_ash

        with constraint_cols[1]:
            max_alt_fuel_rate = st.number_input("Max Alternative Fuel Rate (t/h)", 0.0, 50.0, 40.0)
            constraints['max_alt_fuel_rate'] = max_alt_fuel_rate

        # Run optimization
        if st.button("🚀 Run TSR Optimization", type="primary"):
            with st.spinner("Running advanced optimization..."):
                # Single objective optimization
                single_result = tsr_optimizer.optimize_fuel_mix(target_tsr, constraints)

                # Multi-objective optimization
                objectives = {
                    'cost_weight': cost_weight,
                    'emissions_weight': emissions_weight,
                    'energy_weight': energy_weight,
                    'quality_weight': 0.1
                }
                multi_result = tsr_optimizer.multi_objective_optimization(objectives)

            # Display results
            st.subheader("📊 Optimization Results")

            # Results comparison
            results_col1, results_col2 = st.columns(2)

            with results_col1:
                st.markdown("**🎯 Single Objective Optimization**")
                st.metric("Achieved TSR", f"{single_result.tsr_achieved:.1f}%")
                st.metric("Cost Savings/Month", f"₹{single_result.cost_savings_monthly/100000:.1f}L")
                st.metric("CO2 Reduction", f"{single_result.co2_reduction_tons:.1f} tons")
                st.metric("Energy Efficiency", f"{single_result.energy_efficiency_gain:+.1f}%")

            with results_col2:
                st.markdown("**⚖️ Multi-Objective Optimization**")
                st.metric("Achieved TSR", f"{multi_result.tsr_achieved:.1f}%")
                st.metric("Cost Savings/Month", f"₹{multi_result.cost_savings_monthly/100000:.1f}L")
                st.metric("CO2 Reduction", f"{multi_result.co2_reduction_tons:.1f} tons")
                st.metric("Energy Efficiency", f"{multi_result.energy_efficiency_gain:+.1f}%")

            # Fuel mix comparison
            st.subheader("⛽ Optimal Fuel Mix")

            fuel_mix_data = []
            current_fuel_mix = selected_plant.fuel_mix.copy()

            for fuel in set(list(single_result.optimal_fuel_mix.keys()) + list(current_fuel_mix.keys())):
                fuel_mix_data.append({
                    'Fuel Type': fuel.replace('_', ' ').title(),
                    'Current Mix (t/h)': current_fuel_mix.get(fuel, 0),
                    'Optimized Mix (t/h)': single_result.optimal_fuel_mix.get(fuel, 0),
                    'Multi-Obj Mix (t/h)': multi_result.optimal_fuel_mix.get(fuel, 0)
                })

            fuel_mix_df = pd.DataFrame(fuel_mix_data)
            st.dataframe(fuel_mix_df)

            # Operational impact
            st.subheader("⚙️ Operational Impact Assessment")

            impact_cols = st.columns(2)

            with impact_cols[0]:
                st.markdown("**Single Objective Impact**")
                for key, value in single_result.operational_impact.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")

            with impact_cols[1]:
                st.markdown("**Multi-Objective Impact**")
                for key, value in multi_result.operational_impact.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")

            # Implementation feasibility
            st.subheader("📋 Implementation Feasibility")

            feasibility_col1, feasibility_col2 = st.columns(2)

            with feasibility_col1:
                st.markdown(f"**Single Objective: {single_result.implementation_feasibility}**")
                st.progress(0.9 if "High" in single_result.implementation_feasibility else
                           0.6 if "Medium" in single_result.implementation_feasibility else 0.3)

            with feasibility_col2:
                st.markdown(f"**Multi-Objective: {multi_result.implementation_feasibility}**")
                st.progress(0.85 if "High" in multi_result.implementation_feasibility else
                           0.55 if "Medium" in multi_result.implementation_feasibility else 0.25)

            # Sensitivity analysis
            st.subheader("📊 Sensitivity Analysis")

            tsr_range = np.linspace(15, 35, 5)
            sensitivity_data = []

            for tsr_target in tsr_range:
                try:
                    result = tsr_optimizer.optimize_fuel_mix(tsr_target, constraints)
                    sensitivity_data.append({
                        'TSR_Target': tsr_target,
                        'TSR_Achieved': result.tsr_achieved,
                        'Cost_Savings_Lakh': result.cost_savings_monthly / 100000,
                        'CO2_Reduction': result.co2_reduction_tons
                    })
                except:
                    continue

            if sensitivity_data:
                sensitivity_df = pd.DataFrame(sensitivity_data)
                fig = px.line(sensitivity_df, x='TSR_Target',
                            y=['Cost_Savings_Lakh', 'CO2_Reduction'],
                            title="TSR Sensitivity Analysis")
                st.plotly_chart(fig, use_container_width=True)

        # Optimization constraints explanation
        with st.expander("🔬 Understanding Optimization Constraints"):
            st.markdown("""
            **Linear Programming Constraints:**

            1. **Thermal Energy Balance:**
               ```
               Σ(fuel_rate_i × CV_i) = Total Thermal Requirement
               ```

            2. **TSR Requirement:**
               ```
               Σ(alt_fuel_rate_i × CV_i) / Σ(all_fuel_rate_i × CV_i) = TSR_target
               ```

            3. **Ash Content Limit:**
               ```
               Σ(fuel_rate_i × ash_i) ≤ Max Ash Content × Total Fuel Rate
               ```

            4. **Fuel Availability:**
               ```
               fuel_rate_i ≤ Max Available Rate_i
               ```

            **Multi-Objective Optimization:**
            - **Cost Minimization:** Reduce fuel costs while meeting TSR targets
            - **Emissions Reduction:** Maximize CO2 reduction from alternative fuels
            - **Energy Efficiency:** Optimize thermal efficiency
            - **Quality Impact:** Minimize quality variation
            """)

    except Exception as e:
        st.error(f"❌ Error in TSR optimization demo: {e}")
        st.info("Please ensure all dependencies are installed correctly.")

if __name__ == "__main__":
    main()
