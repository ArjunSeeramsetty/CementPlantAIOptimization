# NEW FILE: src/cement_ai_platform/multi_plant/multi_plant_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List
import time
from datetime import datetime

from .plant_manager import MultiPlantManager
from .multi_plant_supervisor import MultiPlantSupervisor

class MultiPlantDashboard:
    """Dashboard for multi-plant management and monitoring"""
    
    def __init__(self):
        self.plant_manager = MultiPlantManager()
        self.supervisor = MultiPlantSupervisor()
    
    def launch_multi_plant_dashboard(self):
        """Launch multi-plant management dashboard"""
        
        st.set_page_config(
            page_title="🏭 Multi-Plant Management Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏭 Multi-Plant Cement Operations Dashboard")
        st.markdown("**Enterprise-Scale Plant Management System**")
        
        # Sidebar - Tenant and Plant Selection
        with st.sidebar:
            st.header("🏢 Tenant Selection")
            
            # Get available tenants
            all_plants = list(self.plant_manager.plant_registry.values())
            available_tenants = list(set(p.tenant_id for p in all_plants))
            
            if available_tenants:
                selected_tenant = st.selectbox("Select Tenant", available_tenants)
                
                # Get plants for selected tenant
                tenant_plants = self.plant_manager.get_tenant_plants(selected_tenant)
                plant_options = {p.plant_name: p.plant_id for p in tenant_plants}
                
                st.header("🏭 Plant Selection")
                selected_plants = st.multiselect(
                    "Select Plants", 
                    options=list(plant_options.keys()),
                    default=list(plant_options.keys())[:3]  # Select first 3
                )
                
                selected_plant_ids = [plant_options[name] for name in selected_plants]
            else:
                st.error("No plants registered in system")
                return
        
        # Main dashboard
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏭 Plant Overview", 
            "📊 Cross-Plant Analytics", 
            "⚙️ Configuration", 
            "🚀 Model Deployment",
            "🎼 Supervisor Control"
        ])
        
        with tab1:
            self._render_plant_overview(selected_tenant, selected_plant_ids)
        
        with tab2:
            self._render_cross_plant_analytics(selected_tenant)
        
        with tab3:
            self._render_configuration_management(selected_tenant)
        
        with tab4:
            self._render_model_deployment(selected_tenant, selected_plant_ids)
        
        with tab5:
            self._render_supervisor_control()
    
    def _render_plant_overview(self, tenant_id: str, plant_ids: List[str]):
        """Render plant overview dashboard"""
        
        st.subheader(f"🏭 Plant Overview - {tenant_id.upper()}")
        
        if not plant_ids:
            st.info("No plants selected")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_capacity = 0
        total_plants = len(plant_ids)
        
        for plant_id in plant_ids:
            config = self.plant_manager.get_plant_config(plant_id)
            if config:
                total_capacity += config.capacity_tpd
        
        with col1:
            st.metric("Total Plants", total_plants)
        with col2:
            st.metric("Total Capacity", f"{total_capacity:,.0f} TPD")
        with col3:
            st.metric("Avg Capacity", f"{total_capacity/total_plants:,.0f} TPD")
        with col4:
            st.metric("Tenant Status", "ACTIVE")
        
        # Plant details
        st.subheader("📋 Plant Details")
        
        plant_data = []
        for plant_id in plant_ids:
            config = self.plant_manager.get_plant_config(plant_id)
            if config:
                # Simulate operational data
                import random
                plant_data.append({
                    'Plant Name': config.plant_name,
                    'Location': config.location,
                    'Capacity (TPD)': f"{config.capacity_tpd:,.0f}",
                    'Status': random.choice(['Running', 'Running', 'Running', 'Maintenance']),
                    'Efficiency': f"{random.uniform(88, 96):.1f}%",
                    'Free Lime': f"{random.uniform(0.8, 1.8):.1f}%",
                    'Energy (kcal/kg)': f"{random.uniform(680, 750):.0f}"
                })
        
        if plant_data:
            df = pd.DataFrame(plant_data)
            st.dataframe(df, use_container_width=True)
    
    def _render_cross_plant_analytics(self, tenant_id: str):
        """Render cross-plant analytics and benchmarking"""
        
        st.subheader(f"📊 Cross-Plant Analytics - {tenant_id.upper()}")
        
        # Get benchmarks
        benchmarks = self.plant_manager.get_cross_plant_benchmarks(tenant_id)
        
        if not benchmarks:
            st.info("No benchmark data available")
            return
        
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Energy Efficiency Ranking**")
            
            energy_data = benchmarks['energy_efficiency']['plants']
            energy_df = pd.DataFrame([
                {'Plant': data['plant_name'], 'Efficiency': data['value']}  
                for plant_id, data in energy_data.items()
            ]).sort_values('Efficiency', ascending=False)
            
            fig = px.bar(energy_df, x='Plant', y='Efficiency', 
                        title="Energy Efficiency by Plant")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Quality Consistency Ranking**")
            
            quality_data = benchmarks['quality_consistency']['plants']
            quality_df = pd.DataFrame([
                {'Plant': data['plant_name'], 'Quality Score': data['value']}
                for plant_id, data in quality_data.items()
            ]).sort_values('Quality Score', ascending=False)
            
            fig = px.bar(quality_df, x='Plant', y='Quality Score',
                        title="Quality Consistency by Plant")
            st.plotly_chart(fig, use_container_width=True)
        
        # Best practices sharing
        st.subheader("💡 Best Practices Sharing")
        
        best_energy_plant = benchmarks['energy_efficiency']['best']
        best_quality_plant = benchmarks['quality_consistency']['best']
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.info(f"""
            **🏆 Energy Efficiency Leader:**
            {best_energy_plant[1]['plant_name']}
            
            **Score:** {best_energy_plant[1]['value']:.1%}
            
            **Best Practices:**
            • Advanced kiln control algorithms
            • Waste heat recovery optimization
            • Preventive maintenance scheduling
            """)
        
        with col4:
            st.success(f"""
            **🎯 Quality Consistency Leader:**
            {best_quality_plant[1]['plant_name']}
            
            **Score:** {best_quality_plant[1]['value']:.1%}
            
            **Best Practices:**
            • Real-time quality monitoring
            • Automated raw meal blending
            • Advanced process control
            """)
    
    def _render_configuration_management(self, tenant_id: str):
        """Render configuration management interface"""
        
        st.subheader(f"⚙️ Configuration Management - {tenant_id.upper()}")
        
        # Plant configuration templates
        st.markdown("**📋 Plant Configuration Templates**")
        
        template_options = ["Standard Indian Plant", "High Capacity Plant", "UAE Plant", "Custom"]
        selected_template = st.selectbox("Select Template", template_options)
        
        if st.button("🚀 Deploy Template to Selected Plants"):
            st.success(f"Template '{selected_template}' deployed successfully!")
        
        # Configuration validation
        st.markdown("**✅ Configuration Validation**")
        
        validation_results = {
            "DCS Tag Mappings": "✅ Valid",
            "Safety Interlocks": "✅ Valid", 
            "Quality Targets": "⚠️ Review Required",
            "Environmental Limits": "✅ Valid"
        }
        
        for item, status in validation_results.items():
            st.write(f"{status} {item}")
    
    def _render_model_deployment(self, tenant_id: str, plant_ids: List[str]):
        """Render AI model deployment interface"""
        
        st.subheader(f"🚀 AI Model Deployment - {tenant_id.upper()}")
        
        # Available models
        available_models = [
            "Quality Prediction Model v2.1",
            "Energy Optimization Model v1.8", 
            "Anomaly Detection Model v3.0",
            "Predictive Maintenance Model v1.5"
        ]
        
        selected_model = st.selectbox("Select Model to Deploy", available_models)
        
        # Deployment options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Deployment Targets**")
            
            if plant_ids:
                for plant_id in plant_ids:
                    config = self.plant_manager.get_plant_config(plant_id)
                    if config:
                        st.checkbox(config.plant_name, value=True, key=f"deploy_{plant_id}")
        
        with col2:
            st.markdown("**⚙️ Deployment Settings**")
            
            deployment_mode = st.radio("Deployment Mode", ["Staged Rollout", "Immediate", "Canary"])
            auto_rollback = st.checkbox("Auto Rollback on Failure", value=True)
            
        if st.button("🚀 Deploy Model"):
            
            # Simulate deployment
            with st.spinner("Deploying model to selected plants..."):
                time.sleep(3)
                
                # Show results
                st.success("✅ Model deployment completed!")
                
                for plant_id in plant_ids:
                    config = self.plant_manager.get_plant_config(plant_id)
                    st.write(f"✅ Success {config.plant_name if config else plant_id}")
    
    def _render_supervisor_control(self):
        """Render supervisor control interface"""
        
        st.subheader("🎼 MultiPlantSupervisor Control")
        
        # Supervisor status
        supervisor_status = self.supervisor.get_supervisor_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢 Running" if supervisor_status['supervisor_status']['running'] else "🔴 Stopped"
            st.metric("Supervisor Status", status)
        
        with col2:
            st.metric("Active Plants", supervisor_status['supervisor_status']['active_plants'])
        
        with col3:
            health_score = supervisor_status['supervisor_status']['tenant_health_score']
            st.metric("Tenant Health", f"{health_score:.1%}")
        
        with col4:
            st.metric("Active Alerts", supervisor_status['active_alerts'])
        
        # Control buttons
        st.markdown("**🎮 Supervisor Controls**")
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            if supervisor_status['supervisor_status']['running']:
                if st.button("⏹️ Stop Orchestration"):
                    self.supervisor.stop_orchestration()
                    st.success("Orchestration stopped")
                    st.rerun()
            else:
                if st.button("▶️ Start Orchestration"):
                    self.supervisor.start_orchestration()
                    st.success("Orchestration started")
                    st.rerun()
        
        with col6:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        with col7:
            if st.button("📊 Generate Report"):
                st.info("Generating cross-plant report...")
        
        # Tenant summary
        st.subheader("📊 Tenant Summary")
        
        tenant_summary = supervisor_status.get('tenant_summary', {})
        
        if tenant_summary:
            summary_data = []
            for tenant_id, metrics in tenant_summary.items():
                summary_data.append({
                    'Tenant': tenant_id.upper(),
                    'Plants': metrics['total_plants'],
                    'Active': metrics['active_plants'],
                    'Capacity (TPD)': f"{metrics['total_capacity']:,.0f}",
                    'Health Score': f"{metrics['avg_health_score']:.1%}",
                    'Performance': f"{metrics['performance_score']:.1%}",
                    'Critical Alerts': metrics['critical_alerts']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Recent alerts
        st.subheader("🚨 Recent Cross-Plant Alerts")
        
        recent_alerts = supervisor_status.get('recent_alerts', [])
        
        if recent_alerts:
            for alert in recent_alerts:
                severity_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(alert.get('severity', 'Low'), '🟢')
                
                st.markdown(f"""
                **{severity_color} {alert.get('type', 'Alert')}** - {alert.get('severity', 'Low')}
                
                {alert.get('message', 'No message')}
                
                *{alert.get('timestamp', 'Unknown time')}*
                """)
        else:
            st.info("No recent cross-plant alerts")
        
        # Cross-plant benchmarks
        st.subheader("📈 Cross-Plant Benchmarks")
        
        benchmarks = supervisor_status.get('cross_plant_benchmarks', {})
        
        if benchmarks:
            for tenant_id, tenant_benchmarks in benchmarks.items():
                st.markdown(f"**{tenant_id.upper()} Benchmarks**")
                
                col8, col9, col10 = st.columns(3)
                
                with col8:
                    energy_bench = tenant_benchmarks.get('energy_efficiency', {})
                    if energy_bench:
                        st.metric("Energy Efficiency", f"{energy_bench.get('average', 0):.1%}")
                
                with col9:
                    quality_bench = tenant_benchmarks.get('quality_consistency', {})
                    if quality_bench:
                        st.metric("Quality Consistency", f"{quality_bench.get('average', 0):.1%}")
                
                with col10:
                    capacity_bench = tenant_benchmarks.get('capacity_utilization', {})
                    if capacity_bench:
                        st.metric("Capacity Utilization", f"{capacity_bench.get('average', 0):.1%}")

# Demo launcher
def launch_multi_plant_demo():
    """Launch multi-plant management demo"""
    dashboard = MultiPlantDashboard()
    dashboard.launch_multi_plant_dashboard()

if __name__ == "__main__":
    launch_multi_plant_demo()
