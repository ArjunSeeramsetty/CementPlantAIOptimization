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

def show_module_status():
    """Display the status of each module"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Module Status")
    
    modules = [
        ("Live Plant Twin", DYNAMIC_TWIN_AVAILABLE),
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
        - Alternative Fuel Optimization
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

if __name__ == "__main__":
    main()
