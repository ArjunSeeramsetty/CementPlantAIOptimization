# src/cement_ai_platform/dashboard/unified_dashboard.py
"""
Unified Dashboard for JK Cement Digital Twin Platform POC
Provides a single navigation interface for all seven enhancement modules
"""

import logging
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

MODULE_IMPORT_ERRORS = {}


def show_runtime_environment_notice():
    """Warn when the dashboard is launched outside the project virtual environment."""
    executable_path = sys.executable.lower()
    in_project_venv = "\\.venv\\" in executable_path or "/.venv/" in executable_path
    running_in_container = executable_path.startswith("/usr/local/bin/python")

    if not in_project_venv and not running_in_container:
        st.warning(
            "Dashboard is not running from the project virtual environment. "
            "This commonly causes many modules to appear unavailable."
        )
        st.code(
            ".venv\\Scripts\\streamlit.exe run src/cement_ai_platform/dashboard/unified_dashboard.py",
            language="powershell",
        )
        st.caption(f"Current interpreter: {sys.executable}")

# Import each dashboard module
try:
    from cement_ai_platform.streaming.realtime_dashboard import launch_streaming_demo
    STREAMING_AVAILABLE = True
except ImportError as e:
    STREAMING_AVAILABLE = False
    MODULE_IMPORT_ERRORS["streaming"] = str(e)
    logger.warning("Streaming dashboard not available: %s", e)

try:
    from cement_ai_platform.multi_plant.multi_plant_dashboard import launch_multi_plant_demo
    MULTI_PLANT_AVAILABLE = True
except ImportError as e:
    MULTI_PLANT_AVAILABLE = False
    MODULE_IMPORT_ERRORS["multi_plant"] = str(e)
    logger.warning("Multi-plant dashboard not available: %s", e)

try:
    from cement_ai_platform.mobile.mobile_dashboard import launch_mobile_demo
    MOBILE_AVAILABLE = True
except ImportError as e:
    MOBILE_AVAILABLE = False
    MODULE_IMPORT_ERRORS["mobile"] = str(e)
    logger.warning("Mobile dashboard not available: %s", e)

try:
    from cement_ai_platform.maintenance.maintenance_dashboard import launch_predictive_maintenance_demo
    MAINTENANCE_AVAILABLE = True
except ImportError as e:
    MAINTENANCE_AVAILABLE = False
    MODULE_IMPORT_ERRORS["maintenance"] = str(e)
    logger.warning("Maintenance dashboard not available: %s", e)

try:
    from cement_ai_platform.validation.validation_dashboard import launch_data_validation_demo
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    MODULE_IMPORT_ERRORS["validation"] = str(e)
    logger.warning("Validation dashboard not available: %s", e)

try:
    from cement_ai_platform.dwsim.dwsim_dashboard import launch_dwsim_integration_demo
    DWSIM_AVAILABLE = True
except ImportError as e:
    DWSIM_AVAILABLE = False
    MODULE_IMPORT_ERRORS["dwsim"] = str(e)
    logger.warning("DWSIM dashboard not available: %s", e)

try:
    from cement_ai_platform.dashboard.dynamic_plant_twin import launch_dynamic_plant_twin
    DYNAMIC_TWIN_AVAILABLE = True
except ImportError as e:
    DYNAMIC_TWIN_AVAILABLE = False
    MODULE_IMPORT_ERRORS["dynamic_twin"] = str(e)
    logger.warning("Dynamic Plant Twin not available: %s", e)

try:
    from cement_ai_platform.tsr_optimization.tsr_fuel_optimizer import launch_tsr_fuel_optimizer_demo
    TSR_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    TSR_OPTIMIZER_AVAILABLE = False
    MODULE_IMPORT_ERRORS["tsr_optimizer"] = str(e)
    logger.warning("TSR Optimizer not available: %s", e)

try:
    from cement_ai_platform.copilot.plant_ai_assistant import launch_plant_ai_assistant
    PLANT_COPILOT_AVAILABLE = True
except ImportError as e:
    PLANT_COPILOT_AVAILABLE = False
    MODULE_IMPORT_ERRORS["plant_copilot"] = str(e)
    logger.warning("Plant Copilot not available: %s", e)

try:
    from cement_ai_platform.utilities.utility_optimizer import launch_utility_optimization_demo
    UTILITY_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    UTILITY_OPTIMIZER_AVAILABLE = False
    MODULE_IMPORT_ERRORS["utility_optimizer"] = str(e)
    logger.warning("Utility Optimizer not available: %s", e)

try:
    from cement_ai_platform.lims.lims_integration import launch_lims_integration_demo
    LIMS_AVAILABLE = True
except ImportError as e:
    LIMS_AVAILABLE = False
    MODULE_IMPORT_ERRORS["lims"] = str(e)
    logger.warning("LIMS Integration not available: %s", e)

try:
    from cement_ai_platform.analytics.historical_analytics import launch_historical_analytics_demo
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    MODULE_IMPORT_ERRORS["analytics"] = str(e)
    logger.warning("Historical Analytics not available: %s", e)

# New Enhanced Modules
try:
    from cement_ai_platform.config.enhanced_plant_config import EnhancedPlantConfigManager
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError as e:
    ENHANCED_CONFIG_AVAILABLE = False
    MODULE_IMPORT_ERRORS["enhanced_config"] = str(e)
    logger.warning("Enhanced Plant Config not available: %s", e)

try:
    from cement_ai_platform.models.physics_models import DynamicProcessDataGenerator, PhysicsBasedProcessModel
    PHYSICS_MODELS_AVAILABLE = True
except ImportError as e:
    PHYSICS_MODELS_AVAILABLE = False
    MODULE_IMPORT_ERRORS["physics_models"] = str(e)
    logger.warning("Physics Models not available: %s", e)

try:
    from cement_ai_platform.optimization.tsr_optimizer import AdvancedTSROptimizer, FuelOptimizationResult
    ADVANCED_TSR_AVAILABLE = True
except ImportError as e:
    ADVANCED_TSR_AVAILABLE = False
    MODULE_IMPORT_ERRORS["advanced_tsr"] = str(e)
    logger.warning("Advanced TSR Optimizer not available: %s", e)

try:
    from cement_ai_platform.analytics.multi_plant_analytics import create_enhanced_multi_plant_dashboard
    MULTI_PLANT_ANALYTICS_AVAILABLE = True
except ImportError as e:
    MULTI_PLANT_ANALYTICS_AVAILABLE = False
    MODULE_IMPORT_ERRORS["multi_plant_analytics"] = str(e)
    logger.warning("Multi-Plant Analytics not available: %s", e)

try:
    from cement_ai_platform.copilot.enhanced_ai_assistant import CementPlantCopilot
    ENHANCED_COPILOT_AVAILABLE = True
except ImportError as e:
    ENHANCED_COPILOT_AVAILABLE = False
    MODULE_IMPORT_ERRORS["enhanced_copilot"] = str(e)
    logger.warning("Enhanced AI Copilot not available: %s", e)


def show_module_unavailable(module_label: str, error_key: str) -> None:
    """Render a consistent unavailable-module message with the root import error."""
    st.error(f"{module_label} module is not available")
    error_message = MODULE_IMPORT_ERRORS.get(error_key)
    if error_message:
        st.code(error_message)
    else:
        st.info("Please ensure the module is properly installed and configured.")

def show_module_status():
    """Display the status of each module"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Module Status")
    
    modules = [
        ("Live Plant Twin", DYNAMIC_TWIN_AVAILABLE),
        ("🛡️ Perimeter & Vision Safety", True),
        ("⚛️ Process Drift Simulation", True),
        ("🔬 Enhanced Plant Config", ENHANCED_CONFIG_AVAILABLE),
        ("⚛️ Physics-Based Models", PHYSICS_MODELS_AVAILABLE),
        ("🔥 Advanced TSR Optimizer", ADVANCED_TSR_AVAILABLE),
        ("📊 Multi-Plant Analytics", MULTI_PLANT_ANALYTICS_AVAILABLE),
        ("🤖 Enhanced AI Copilot", ENHANCED_COPILOT_AVAILABLE),
        ("Utility Optimization", UTILITY_OPTIMIZER_AVAILABLE),
        ("LIMS Integration", LIMS_AVAILABLE),
        ("Historical Analytics", ANALYTICS_AVAILABLE),
        ("Real-Time Streaming", STREAMING_AVAILABLE),
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
        **📊 Multi-Plant Analytics**
        - Cross-plant performance benchmarking
        - Industry benchmark comparison
        - Improvement opportunity identification
        - Sustainability score calculation
        """)

        st.markdown("""
        **🤖 Enhanced AI Copilot**
        - Cement domain expertise
        - Plant-specific recommendations
        - Real-time operational guidance
        - Knowledge-based troubleshooting
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
    show_runtime_environment_notice()
    
    # Navigation options
    nav_options = [
        "📊 Platform Overview",
        "🏭 Live Plant Twin",
        "🔬 Enhanced Plant Config",
        "⚛️ Physics-Based Models",
        "🔥 Advanced TSR Optimizer",
        "📊 Multi-Plant Analytics",
        "🤖 Enhanced AI Copilot",
        "🛡️ Perimeter & Vision Safety",
        "⚛️ Process Drift Simulation",
        "💧 Utility Optimization",
        "🧪 LIMS Integration",
        "📊 Historical Analytics",
        "🔄 Real-Time Streaming",
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
            show_module_unavailable("Live Plant Twin", "dynamic_twin")

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
            show_module_unavailable("Advanced TSR Optimizer", "advanced_tsr")

    elif choice == "📊 Multi-Plant Analytics":
        if MULTI_PLANT_ANALYTICS_AVAILABLE:
            launch_multi_plant_analytics_demo()
        else:
            st.error("❌ Multi-Plant Analytics module is not available")
            st.info("Please ensure the multi-plant analytics module is properly installed and configured.")

    elif choice == "🤖 Enhanced AI Copilot":
        if ENHANCED_COPILOT_AVAILABLE:
            launch_enhanced_ai_copilot_demo()
        else:
            st.error("❌ Enhanced AI Copilot module is not available")
            st.info("Please ensure the enhanced AI copilot module is properly installed and configured.")

    elif choice == "🛡️ Perimeter & Vision Safety":
        launch_vision_safety_demo()

    elif choice == "⚛️ Process Drift Simulation":
        launch_process_drift_demo()
    
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
            show_module_unavailable("Utility Optimization", "utility_optimizer")
    
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
            show_module_unavailable("Data Validation", "validation")
            
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

def launch_multi_plant_analytics_demo():
    """Launch multi-plant analytics dashboard"""
    st.title("📊 Multi-Plant Performance Analytics")
    st.markdown("**Advanced cross-plant benchmarking and performance comparison**")

    try:
        # Generate analytics data
        comparison_data, performance_data = create_enhanced_multi_plant_dashboard()

        # Plant performance summary
        st.subheader("🏭 Plant Performance Summary")
        
        plant_summary = comparison_data['plant_summary']
        
        # Display key metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Total Plants", len(plant_summary))
            best_energy = plant_summary.loc[plant_summary['thermal_energy_kcal_kg'].idxmin()]
            st.metric("Best Energy Efficiency", f"{best_energy['thermal_energy_kcal_kg']:.0f} kcal/kg", 
                     delta=f"{best_energy['plant_name']}")

        with metric_cols[1]:
            best_sustainability = plant_summary.loc[plant_summary['sustainability_score'].idxmax()]
            st.metric("Best Sustainability", f"{best_sustainability['sustainability_score']:.1f}/100",
                     delta=f"{best_sustainability['plant_name']}")
            
        with metric_cols[2]:
            best_oee = plant_summary.loc[plant_summary['oee_percentage'].idxmax()]
            st.metric("Best OEE", f"{best_oee['oee_percentage']:.1f}%",
                     delta=f"{best_oee['plant_name']}")

        with metric_cols[3]:
            avg_tsr = plant_summary['tsr_percentage'].mean()
            st.metric("Average TSR", f"{avg_tsr:.1f}%")

        # Performance comparison table
        st.subheader("📊 Performance Comparison Table")
        
        # Select metrics to display
        display_metrics = st.multiselect(
            "Select Metrics to Display:",
            ['plant_name', 'thermal_energy_kcal_kg', 'electrical_energy_kwh_t', 
             'energy_cost_per_ton', 'free_lime_pct', 'cement_strength_28d',
             'nox_emissions_mg_nm3', 'co2_emissions_kg_t', 'oee_percentage',
             'availability_pct', 'total_cost_per_ton', 'tsr_percentage',
             'sustainability_score'],
            default=['plant_name', 'thermal_energy_kcal_kg', 'energy_cost_per_ton', 
                    'oee_percentage', 'tsr_percentage', 'sustainability_score']
        )
        
        display_df = plant_summary[display_metrics]
        st.dataframe(display_df, use_container_width=True)

        # Rankings
        st.subheader("🏆 Plant Rankings")
        
        rankings = comparison_data['rankings']
        
        ranking_cols = st.columns(3)
        
        with ranking_cols[0]:
            st.markdown("**⚡ Energy Efficiency Leaders**")
            for i, plant in enumerate(rankings['energy_efficiency']):
                st.write(f"{i+1}. {plant['plant_name']} - {plant['thermal_energy_kcal_kg']:.0f} kcal/kg")

        with ranking_cols[1]:
            st.markdown("**🌱 Sustainability Champions**")
            for i, plant in enumerate(rankings['sustainability']):
                st.write(f"{i+1}. {plant['plant_name']} - {plant['sustainability_score']:.1f}/100")

        with ranking_cols[2]:
            st.markdown("**💰 Cost Efficiency Leaders**")
            for i, plant in enumerate(rankings['cost_efficiency']):
                st.write(f"{i+1}. {plant['plant_name']} - ₹{plant['total_cost_per_ton']:.0f}/ton")

        # Performance insights
        st.subheader("💡 Key Performance Insights")
        
        insights = comparison_data['insights']
        for insight in insights:
            st.info(insight)

        # Improvement opportunities
        st.subheader("🚀 Improvement Opportunities")
        
        improvement_opportunities = comparison_data['improvement_opportunities']
        
        for plant_name, opportunities in improvement_opportunities.items():
            if opportunities:
                with st.expander(f"📋 {plant_name} - Improvement Opportunities"):
                    for opp in opportunities:
                        st.markdown(f"**{opp['category']}:** {opp['opportunity']}")
                        st.markdown(f"- *Impact:* {opp['impact']}")
                        st.markdown(f"- *Benchmark:* {opp['benchmark']}")
                        st.markdown("---")

        # Benchmark analysis
        st.subheader("📈 Industry Benchmark Analysis")
        
        benchmark_analysis = comparison_data['benchmark_analysis']
        
        benchmark_cols = st.columns(2)
        
        with benchmark_cols[0]:
            st.markdown("**Energy Performance vs Industry**")
            thermal_benchmark = benchmark_analysis['thermal_energy_kcal_kg']
            st.metric(
                "Thermal Energy vs Industry",
                f"{thermal_benchmark['jk_cement_average']:.0f} kcal/kg",
                delta=f"{thermal_benchmark['performance_vs_benchmark_pct']:+.1f}% vs {thermal_benchmark['industry_benchmark']:.0f} benchmark"
            )

        with benchmark_cols[1]:
            st.markdown("**Environmental Performance vs Industry**")
            nox_benchmark = benchmark_analysis['nox_emissions_mg_nm3']
            st.metric(
                "NOx Emissions vs Industry",
                f"{nox_benchmark['jk_cement_average']:.0f} mg/Nm³",
                delta=f"{nox_benchmark['performance_vs_benchmark_pct']:+.1f}% vs {nox_benchmark['industry_benchmark']:.0f} benchmark"
            )

        # Performance trends over time
        st.subheader("📊 Performance Trends (Last 30 Days)")
        
        # Create trend charts
        trend_cols = st.columns(2)
        
        with trend_cols[0]:
            # Energy efficiency trend
            fig_energy = px.line(
                performance_data,
                x='date',
                y='thermal_energy_kcal_kg',
                color='plant_name',
                title='Thermal Energy Consumption Trends'
            )
            st.plotly_chart(fig_energy, use_container_width=True)

        with trend_cols[1]:
            # OEE trend
            fig_oee = px.line(
                performance_data,
                x='date',
                y='oee_percentage',
                color='plant_name',
                title='OEE Performance Trends'
            )
            st.plotly_chart(fig_oee, use_container_width=True)

        # Technology level analysis
        st.subheader("🔬 Technology Level Impact Analysis")
        
        tech_analysis = plant_summary.groupby('technology_level').agg({
            'thermal_energy_kcal_kg': 'mean',
            'sustainability_score': 'mean',
            'oee_percentage': 'mean',
            'total_cost_per_ton': 'mean'
        }).round(2)
        
        st.dataframe(tech_analysis)

        # Download performance data
        st.subheader("💾 Export Performance Data")
        
        csv_data = plant_summary.to_csv(index=False)
        st.download_button(
            label="Download Performance Summary (CSV)",
            data=csv_data,
            file_name="multi_plant_performance_summary.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error in multi-plant analytics demo: {e}")
        st.info("Please ensure all dependencies are installed correctly.")

def launch_enhanced_ai_copilot_demo():
    """Launch enhanced AI copilot dashboard"""
    st.title("🤖 Enhanced AI Copilot")
    st.markdown("**Cement domain expertise with plant-specific recommendations**")

    try:
        # Load plant configurations
        from cement_ai_platform.config.enhanced_plant_config import EnhancedPlantConfigManager
        from cement_ai_platform.models.physics_models import DynamicProcessDataGenerator
        
        config_manager = EnhancedPlantConfigManager()
        plants = config_manager.get_all_plants()

        # Plant selection
        st.subheader("🏭 Select Plant for AI Analysis")
        
        plant_names = [f"{plant.plant_name} ({plant.location})" for plant in plants]
        selected_plant_idx = st.selectbox(
            "Choose Plant:",
            range(len(plants)),
            format_func=lambda x: plant_names[x]
        )

        selected_plant = plants[selected_plant_idx]

        # Initialize AI copilot
        data_generator = DynamicProcessDataGenerator(selected_plant)
        current_kpis = data_generator.generate_current_kpis()
        
        copilot = CementPlantCopilot(selected_plant, current_kpis)

        # Display plant context
        st.subheader("📊 Current Plant Status")
        
        context_cols = st.columns(4)
        
        with context_cols[0]:
            st.metric("Plant", selected_plant.plant_name)
            st.metric("Technology", selected_plant.technology_level.title())
            
        with context_cols[1]:
            st.metric("Kiln Temperature", f"{current_kpis.get('kiln_temperature_c', 0):.1f}°C")
            st.metric("Free Lime", f"{current_kpis.get('free_lime_pct', 0):.2f}%")
            
        with context_cols[2]:
            st.metric("Thermal Energy", f"{current_kpis.get('thermal_energy_kcal_kg', 0):.0f} kcal/kg")
            st.metric("NOx Emissions", f"{current_kpis.get('nox_emissions_mg_nm3', 0):.0f} mg/Nm³")
            
        with context_cols[3]:
            st.metric("OEE", f"{current_kpis.get('oee_percentage', 0):.1f}%")
            st.metric("Production Rate", f"{current_kpis.get('production_rate_tph', 0):.1f} t/h")

        # AI Chat Interface
        st.subheader("💬 AI Assistant Chat")
        
        # Initialize confidence variable to avoid NameError
        confidence = st.session_state.get('ai_confidence', 0.0)
        
        # Display knowledge base summary
        kb_summary = copilot.get_knowledge_base_summary()
        
        with st.expander("🧠 Knowledge Base Overview"):
            st.markdown(f"**Available Expertise Areas:** {', '.join(kb_summary['process_expertise_areas'])}")
            st.markdown(f"**Fuel Optimization Areas:** {', '.join(kb_summary['fuel_optimization_areas'])}")
            st.markdown(f"**Gemini AI Available:** {'✅' if kb_summary['gemini_available'] else '❌'}")
            st.markdown(f"**Current KPIs Available:** {kb_summary['current_kpis_available']}/15")

        # Chat interface
        user_query = st.text_area(
            "Ask the AI Assistant:",
            placeholder="e.g., 'How can I reduce my thermal energy consumption?' or 'What's causing high free lime?'",
            height=100
        )

        if st.button("🚀 Get AI Recommendation", type="primary"):
            if user_query.strip():
                with st.spinner("AI is analyzing your query..."):
                    response = copilot.generate_contextual_response(user_query.strip())

                # Display AI response
                st.subheader("🤖 AI Response")
                
                # Confidence indicator
                confidence = response['confidence']
                st.session_state['ai_confidence'] = confidence  # Store in session state
                confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                st.markdown(f"**Confidence Level:** :{confidence_color}[{confidence:.0%}]")

                # Main answer
                st.markdown("**Answer:**")
                st.write(response['answer'])

                # Plant-specific actions
                if response['plant_specific_actions']:
                    st.markdown("**🎯 Plant-Specific Actions:**")
                    for action in response['plant_specific_actions']:
                        st.markdown(f"• {action}")

                # Related KPIs
                if response['related_kpis']:
                    st.markdown("**📊 Related KPIs to Monitor:**")
                    kpi_cols = st.columns(len(response['related_kpis']))
                    for i, kpi in enumerate(response['related_kpis']):
                        with kpi_cols[i]:
                            kpi_value = current_kpis.get(kpi, 0)
                            kpi_display = kpi.replace('_', ' ').title()
                            st.metric(kpi_display, f"{kpi_value:.2f}")

                # Recommendations
                if response['recommendations']:
                    st.markdown("**💡 Implementation Recommendations:**")
                    for rec in response['recommendations']:
                        with st.expander(f"📋 {rec['action']}"):
                            st.markdown(f"**Expected Impact:** {rec['impact']}")
                            st.markdown(f"**Timeline:** {rec['timeline']}")
                            st.markdown(f"**Investment Required:** {rec['investment']}")
            else:
                st.warning("Please enter a question for the AI assistant.")

        # Pre-defined query examples
        st.subheader("💡 Example Queries")
        
        example_cols = st.columns(2)
        
        with example_cols[0]:
            st.markdown("**🔥 Kiln Operations:**")
            if st.button("How to optimize kiln temperature?"):
                st.session_state.example_query = "How can I optimize my kiln temperature for better efficiency?"
            
            if st.button("Reduce free lime issues?"):
                st.session_state.example_query = "What causes high free lime and how to fix it?"

        with example_cols[1]:
            st.markdown("**⚡ Energy & Environment:**")
            if st.button("Reduce thermal energy consumption?"):
                st.session_state.example_query = "How can I reduce thermal energy consumption by 5%?"
            
            if st.button("Increase TSR safely?"):
                st.session_state.example_query = "How can I safely increase TSR to 30%?"

        # Auto-fill example query
        if hasattr(st.session_state, 'example_query'):
            st.text_area(
                "Selected Example Query:",
                value=st.session_state.example_query,
                height=80,
                key="example_query_display"
            )
            if st.button("Use This Query"):
                user_query = st.session_state.example_query
                # Clear the session state
                del st.session_state.example_query
                st.rerun()

        # AI Performance Metrics
        st.subheader("📈 AI Assistant Performance")
        
        perf_cols = st.columns(3)
        
        with perf_cols[0]:
            st.metric("Response Confidence", f"{confidence:.0%}")
            
        with perf_cols[1]:
            st.metric("Plant Context Integration", "High" if kb_summary['current_kpis_available'] > 10 else "Medium")
            
        with perf_cols[2]:
            st.metric("Domain Expertise", "Advanced" if kb_summary['gemini_available'] else "Basic")

    except Exception as e:
        st.error(f"❌ Error in enhanced AI copilot demo: {e}")
        st.info("Please ensure all dependencies are installed correctly.")


def launch_vision_safety_demo():
    """Launch Perimeter & Vision Safety monitoring dashboard"""
    st.title("🛡️ Perimeter & Vision Safety Monitoring")
    st.markdown("**Real-Time Computer Vision Alerts & PPE Compliance**")

    # Import modules lazily
    from cement_ai_platform.vision.safety_monitor import SafetyMonitor
    import time

    if 'safety_monitor' not in st.session_state:
        st.session_state.safety_monitor = SafetyMonitor()
    
    monitor = st.session_state.safety_monitor

    # Video stream state
    if 'frame_id' not in st.session_state:
        st.session_state.frame_id = 0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📸 Camera Streams")
        camera = st.radio("Select Camera Feed:", [
            "CAM-NORTH-FENCE - Perimeter Limit Check",
            "CAM-LOADING-BAY-3 - PPE Compliance Check"
        ])
        
        feed_type = 'boundary' if "FENCE" in camera else 'ppe'
        camera_id = "CAM-NORTH-FENCE" if feed_type == 'boundary' else "CAM-LOADING-BAY-3"
        
        st.markdown("---")
        st.subheader("🔔 Live Alert Center")
        
        # Pull live status
        if feed_type == 'boundary':
            status = monitor.check_boundary_violation(st.session_state.frame_id)
            if status["is_violation"]:
                st.error("🚨 **CRITICAL**: Perimeter intrusion detected! Intruder crossed north fence boundary.")
                st.session_state.last_incident_obs = "Unidentified individual crossed the designated yellow fence zone."
            else:
                st.success("✅ **STATUS**: Secure zone clear. No anomalies.")
        else:
            status = monitor.check_ppe_compliance(st.session_state.frame_id)
            detection = status["detections"][0]
            if not detection["compliant"]:
                st.warning(f"⚠️ **WARNING**: Safety breach! {detection['label']} is missing {detection['violation'].lower()}.")
                st.session_state.last_incident_obs = f"Worker detected in loading zone without safety equipment: {detection['violation']}."
            else:
                st.success("✅ **STATUS**: All workers fully compliant with PPE regulations.")

        # Interactive controls
        st.markdown("---")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            if st.button("🔄 Next Frame"):
                st.session_state.frame_id += 1
                st.rerun()
        with col_ctrl2:
            auto_play = st.checkbox("Auto-play Feed", value=True)

    with col2:
        st.subheader("📺 Video Monitor")
        
        # Loop animation if auto-play is checked
        frame_placeholder = st.empty()
        
        if auto_play:
            # Play a short sequence of 15 frames per interaction to keep Streamlit responsive
            for i in range(15):
                st.session_state.frame_id += 1
                frame_bytes = monitor.generate_mock_frame_bytes(feed_type, st.session_state.frame_id)
                if frame_bytes:
                    frame_placeholder.image(frame_bytes, use_column_width=True)
                time.sleep(0.08)
        else:
            frame_bytes = monitor.generate_mock_frame_bytes(feed_type, st.session_state.frame_id)
            if frame_bytes:
                frame_placeholder.image(frame_bytes, use_column_width=True)

    st.markdown("---")
    st.subheader("📝 Gemini Incident Reporting")
    
    col_rep1, col_rep2 = st.columns([1, 2])
    with col_rep1:
        st.markdown("Generates a formal, compliance-grade safety report (ISO 45001) using Gemini.")
        report_button = st.button("📋 Generate Incident Report via Gemini", type="primary")
    
    with col_rep2:
        if report_button:
            with st.spinner("Invoking Gemini for compliance audit..."):
                obs = getattr(st.session_state, 'last_incident_obs', "No active breaches in current frame.")
                incident_type = "Restricted Boundary Breach" if feed_type == 'boundary' else "Missing PPE Violation"
                report_md = monitor.generate_safety_incident_report(camera_id, incident_type, obs)
                st.markdown(report_md)


def launch_process_drift_demo():
    """Launch Tennessee Eastman process drift simulator dashboard"""
    st.title("⚛️ TEP Process Anomaly & Drift Simulator")
    st.markdown("**Dynamic Chemical Reactor Mass/Energy Balance & Process Control**")

    # Import modules lazily
    from cement_ai_platform.simulation.tep_simulator import TennesseeEastmanSimulator
    import time

    if 'tep_sim' not in st.session_state:
        st.session_state.tep_sim = TennesseeEastmanSimulator()
    
    sim = st.session_state.tep_sim

    # Active drift tracker
    if 'active_drift_type' not in st.session_state:
        st.session_state.active_drift_type = None

    # Sidebar parameters
    st.sidebar.subheader("🎛️ TEP Control Setpoints")
    cooling_valve = st.sidebar.slider("Reactor Cooling Valve (%)", 20.0, 80.0, float(sim.controls["reactor_cooling_valve"]))
    steam_valve = st.sidebar.slider("Stripper Steam Valve (%)", 10.0, 50.0, float(sim.controls["stripper_steam_valve"]))
    
    # Apply slider controls to simulator state
    sim.controls["reactor_cooling_valve"] = cooling_valve
    sim.controls["stripper_steam_valve"] = steam_valve

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🚨 Anomaly Drift Injector")
        st.markdown("Inject operational deviations to evaluate drift-detection thresholds.")

        # Drift trigger buttons
        if st.button("🔥 Inject Catalyst Decay (Slow Anomaly)"):
            sim.inject_drift("catalyst_decay")
            st.session_state.active_drift_type = "catalyst_decay"
            st.warning("Catalyst Decay anomaly injected (Slow loss of product yield)")

        if st.button("🌡️ Inject Cooling Valve Stiction (Oscillations)"):
            sim.inject_drift("reactor_valve_stiction")
            st.session_state.active_drift_type = "reactor_valve_stiction"
            st.warning("Cooling Water Valve Stiction anomaly injected (Temperature oscillations)")

        if st.button("❄️ Inject Feed D Temperature Step"):
            sim.inject_drift("feed_d_temp_step")
            st.session_state.active_drift_type = "feed_d_temp_step"
            st.warning("Feed D Temperature Step anomaly injected (Sudden ambient heat increase)")

        if st.button("💨 Inject Feed A Composition Shift (Pressure Buildup)"):
            sim.inject_drift("feed_a_composition")
            st.session_state.active_drift_type = "feed_a_composition"
            st.warning("Feed A Composition Shift anomaly injected (Rapid pressure climb)")

        st.markdown("---")
        if st.button("🔄 Reset Simulator & Clear Drifts", type="primary"):
            sim.clear_drifts()
            sim.reset()
            st.session_state.active_drift_type = None
            st.success("Simulator reset to stable steady-state conditions!")

        st.markdown("---")
        st.subheader("💡 TEP Process State")
        st.metric("Reactor Temperature", f"{sim.state['reactor_temp']:.2f} °C", delta="Trip Limit: 135°C")
        st.metric("Reactor Pressure", f"{sim.state['reactor_pressure']:.1f} kPa", delta="Trip Limit: 3100 kPa")
        st.metric("Separator Level", f"{sim.state['separator_level']:.2f} %", delta="Target: 50%")

    with col2:
        st.subheader("📈 Real-Time Process Telemetry")
        
        # Run dynamic simulation history
        history = sim.generate_history(num_steps=100, drift_to_inject=st.session_state.active_drift_type)
        df = pd.DataFrame(history)
        
        # Check safety trip
        if sim.safety_tripped:
            st.error(f"🚨 **SYSTEM CRITICALLY TRIPPED**: {sim.trip_reason}")
            st.info("System is cooling down passively. Reset the simulator to restore operations.")

        # Plotly charts
        # Temperature
        fig_temp = px.line(df, x="step_idx", y="reactor_temp", 
                          title="Reactor Temperature Profile (°C)", 
                          color_discrete_sequence=["#FF5722"])
        fig_temp.add_hline(y=135.0, line_dash="dash", line_color="red", annotation_text="Safety Limit (135°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

        # Pressure
        fig_press = px.line(df, x="step_idx", y="reactor_pressure", 
                           title="Reactor Pressure Profile (kPa)",
                           color_discrete_sequence=["#2196F3"])
        fig_press.add_hline(y=3100.0, line_dash="dash", line_color="red", annotation_text="Safety Limit (3100 kPa)")
        st.plotly_chart(fig_press, use_container_width=True)

        # Levels
        fig_level = px.line(df, x="step_idx", y=["separator_level", "stripper_level"], 
                           title="Vessel Levels (%)")
        st.plotly_chart(fig_level, use_container_width=True)


if __name__ == "__main__":
    main()
