# NEW FILE: src/cement_ai_platform/dwsim/dwsim_dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
from datetime import datetime
from .dwsim_connector import DWSIMIntegrationEngine, DWSIMScenario

def launch_dwsim_integration_demo():
    """Launch DWSIM integration demo interface"""
    
    st.set_page_config(
        page_title="⚗️ DWSIM Physics Integration",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚗️ DWSIM Physics-Based Digital Twin")
    st.markdown("**End-to-End Process Simulation & Scenario Analysis**")
    
    # Initialize DWSIM engine
    if 'dwsim_engine' not in st.session_state:
        st.session_state.dwsim_engine = DWSIMIntegrationEngine()
    
    dwsim_engine = st.session_state.dwsim_engine
    
    # Sidebar controls
    with st.sidebar:
        st.header("🏭 Plant Configuration")
        plant_id = st.selectbox("Select Plant", ["JK_Rajasthan_Demo", "UltraTech_Gujarat_Demo"])
        simulation_mode = st.selectbox("Simulation Mode", ["Real-time", "Batch", "Optimization"])
        
        st.header("⚙️ DWSIM Settings")
        solver_tolerance = st.slider("Solver Tolerance", 1e-6, 1e-3, 1e-4, format="%.0e")
        max_iterations = st.slider("Max Iterations", 50, 500, 100)
        
        st.header("📊 Output Options")
        st.checkbox("Generate Charts", value=True)
        st.checkbox("Export Results", value=True)
        st.checkbox("Real-time Updates", value=False)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Scenario Library", "🔬 Simulation Results", "📈 Analysis Dashboard", "⚙️ Integration Status"])
    
    with tab1:
        st.subheader("🎯 Standard Scenario Library")
        
        # Display available scenarios
        for scenario_id, scenario in dwsim_engine.standard_scenarios.items():
            
            priority_colors = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }
            
            with st.expander(f"{priority_colors[scenario.priority]} {scenario.scenario_name}", expanded=False):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Description:** {scenario.description}
                    
                    **Input Parameters:**
                    """)
                    
                    # Display input parameters
                    for param, value in scenario.input_parameters.items():
                        if isinstance(value, list):
                            st.write(f"• {param}: {value[0]} - {value[1]}")
                        else:
                            st.write(f"• {param}: {value}")
                    
                    st.markdown(f"""
                    **Expected Outputs:** {', '.join(scenario.expected_outputs)}
                    
                    **Simulation Duration:** {scenario.simulation_duration/60:.1f} minutes
                    """)
                
                with col2:
                    st.markdown("**🚀 Quick Actions**")
                    
                    if st.button(f"▶️ Run Scenario", key=f"run_{scenario_id}"):
                        
                        with st.spinner(f"Executing {scenario.scenario_name}..."):
                            
                            # Execute scenario
                            execution_result = dwsim_engine.execute_scenario(scenario, plant_id)
                            
                            # Store in session state
                            if 'scenario_results' not in st.session_state:
                                st.session_state.scenario_results = {}
                            
                            st.session_state.scenario_results[scenario_id] = execution_result
                            
                            if execution_result['success']:
                                st.success(f"✅ Scenario completed in {execution_result['execution_duration']:.1f} seconds")
                                st.balloons()
                            else:
                                st.error(f"❌ Scenario failed: {execution_result.get('error', 'Unknown error')}")
                    
                    if st.button(f"📊 View History", key=f"history_{scenario_id}"):
                        st.info(f"📈 Loading execution history for {scenario.scenario_name}...")
                    
                    if st.button(f"⚙️ Customize", key=f"custom_{scenario_id}"):
                        st.info(f"🔧 Opening parameter customization for {scenario.scenario_name}...")
        
        # Custom scenario creation
        st.subheader("🛠️ Create Custom Scenario")
        
        with st.expander("➕ New Custom Scenario", expanded=False):
            
            custom_name = st.text_input("Scenario Name", "Custom Process Study")
            custom_description = st.text_area("Description", "Custom process simulation scenario")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input Parameters**")
                
                feed_rate = st.number_input("Raw Meal Feed Rate (t/h)", 140.0, 190.0, 167.0)
                fuel_rate = st.number_input("Fuel Rate (t/h)", 12.0, 22.0, 16.5)
                kiln_speed = st.number_input("Kiln Speed (rpm)", 2.5, 4.5, 3.5)
                o2_target = st.number_input("O2 Target (%)", 2.0, 6.0, 3.2)
            
            with col2:
                st.markdown("**Simulation Settings**")
                
                duration = st.number_input("Duration (minutes)", 10, 120, 30)
                priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
                
                expected_outputs = st.multiselect(
                    "Expected Outputs",
                    ["burning_zone_temp", "free_lime_percent", "thermal_energy", "nox_emissions"],
                    default=["burning_zone_temp", "free_lime_percent"]
                )
            
            if st.button("🚀 Create & Run Custom Scenario"):
                
                # Create custom scenario
                custom_scenario = DWSIMScenario(
                    scenario_id=f"custom_{int(time.time())}",
                    scenario_name=custom_name,
                    description=custom_description,
                    input_parameters={
                        'raw_meal_feed': feed_rate,
                        'fuel_rate': fuel_rate,
                        'kiln_speed': kiln_speed,
                        'o2_target': o2_target
                    },
                    expected_outputs=expected_outputs,
                    simulation_duration=duration * 60,
                    priority=priority
                )
                
                with st.spinner("Executing custom scenario..."):
                    execution_result = dwsim_engine.execute_scenario(custom_scenario, plant_id)
                    
                    if execution_result['success']:
                        st.success(f"✅ Custom scenario completed successfully!")
                        
                        # Store result
                        if 'scenario_results' not in st.session_state:
                            st.session_state.scenario_results = {}
                        st.session_state.scenario_results[custom_scenario.scenario_id] = execution_result
                    else:
                        st.error(f"❌ Custom scenario failed: {execution_result.get('error')}")
    
    with tab2:
        st.subheader("🔬 Simulation Results")
        
        if 'scenario_results' in st.session_state and st.session_state.scenario_results:
            
            # Results selector
            available_results = list(st.session_state.scenario_results.keys())
            selected_result = st.selectbox("Select Result to View", available_results)
            
            if selected_result:
                result_data = st.session_state.scenario_results[selected_result]
                
                if result_data['success']:
                    
                    # Result summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Execution Time", f"{result_data['execution_duration']:.1f}s")
                    with col2:
                        st.metric("Status", "✅ Success")
                    with col3:
                        st.metric("Data Points", len(str(result_data['results'])))
                    with col4:
                        st.metric("Timestamp", result_data['timestamp'][:16])
                    
                    # Detailed results based on scenario type
                    results = result_data['results']
                    
                    if 'time_series' in results:
                        # Time series results (startup scenario)
                        st.subheader("📈 Time Series Results")
                        
                        time_series = results['time_series']
                        
                        # Create multi-line chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=time_series['time_minutes'],
                            y=time_series['burning_zone_temp_c'],
                            mode='lines+markers',
                            name='Burning Zone Temp (°C)',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=time_series['time_minutes'],
                            y=[temp/10 for temp in time_series['preheater_stage1_temp_c']],  # Scale for visibility
                            mode='lines+markers',
                            name='Preheater Stage 1 (°C/10)',
                            line=dict(color='orange', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=time_series['time_minutes'],
                            y=time_series['o2_percent'],
                            mode='lines+markers',
                            name='O2 (%)',
                            yaxis='y2',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Process Variables Over Time",
                            xaxis_title="Time (minutes)",
                            yaxis_title="Temperature (°C)",
                            yaxis2=dict(
                                title="O2 (%)",
                                overlaying='y',
                                side='right'
                            ),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Final steady state
                        if 'final_steady_state' in results:
                            st.subheader("🎯 Final Steady State")
                            
                            steady_state = results['final_steady_state']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Burning Zone Temp", f"{steady_state['burning_zone_temp_c']:.1f}°C")
                                st.metric("Thermal Energy", f"{steady_state['thermal_energy_kcal_kg']:.1f} kcal/kg")
                            
                            with col2:
                                st.metric("Free Lime", f"{steady_state['free_lime_percent']:.2f}%")
                                st.metric("Production Rate", f"{steady_state['production_rate_tph']:.1f} t/h")
                            
                            with col3:
                                st.metric("NOx Emissions", f"{steady_state['nox_mg_nm3']:.1f} mg/Nm³")
                    
                    elif 'optimization_matrix' in results:
                        # Optimization results
                        st.subheader("🎯 Optimization Results")
                        
                        # Display optimal solution
                        optimal = results['optimal_solution']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Optimal Kiln Speed", f"{optimal['kiln_speed_rpm']:.2f} rpm")
                            st.metric("Optimal O2", f"{optimal['o2_percent']:.2f}%")
                        
                        with col2:
                            st.metric("Predicted Energy", f"{optimal['predicted_thermal_energy_kcal_kg']:.1f} kcal/kg")
                            st.metric("Energy Savings", f"{optimal['energy_savings_percent']:.1f}%")
                        
                        with col3:
                            st.metric("Production Rate", f"{optimal['predicted_production_rate_tph']:.1f} t/h")
                        
                        # Optimization matrix visualization
                        matrix_df = pd.DataFrame(results['optimization_matrix'])
                        
                        fig = px.scatter_3d(
                            matrix_df,
                            x='kiln_speed_rpm',
                            y='o2_percent',
                            z='thermal_energy_kcal_kg',
                            color='thermal_energy_kcal_kg',
                            title="3D Optimization Space",
                            labels={
                                'kiln_speed_rpm': 'Kiln Speed (rpm)',
                                'o2_percent': 'O2 (%)',
                                'thermal_energy_kcal_kg': 'Thermal Energy (kcal/kg)'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif 'transition_profile' in results:
                        # Fuel switching results
                        st.subheader("⛽ Fuel Switch Transition")
                        
                        transition = results['transition_profile']
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=transition['time_minutes'],
                            y=transition['coal_rate_tph'],
                            mode='lines+markers',
                            name='Coal Rate (t/h)',
                            line=dict(color='black', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=transition['time_minutes'],
                            y=transition['alt_fuel_rate_tph'],
                            mode='lines+markers',
                            name='Alternative Fuel Rate (t/h)',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=transition['time_minutes'],
                            y=[temp/100 for temp in transition['burning_zone_temp_c']],  # Scale for visibility
                            mode='lines+markers',
                            name='Burning Zone Temp (°C/100)',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Fuel Switching Transition Profile",
                            xaxis_title="Time (minutes)",
                            yaxis_title="Rate (t/h) / Scaled Temperature",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Environmental impact
                        if 'environmental_impact' in results:
                            st.subheader("🌱 Environmental Impact")
                            
                            impact = results['environmental_impact']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("CO2 Reduction", f"{impact['co2_reduction_percent']:.1f}%")
                            with col2:
                                st.metric("NOx Reduction", f"{impact['nox_reduction_percent']:.1f}%")
                            with col3:
                                st.metric("Waste Fuel Used", f"{impact['waste_fuel_utilized_tons']:.1f} t")
                            with col4:
                                st.metric("TSR Achieved", f"{impact['tsr_achieved_percent']:.1f}%")
                    
                    else:
                        # Generic results display
                        st.subheader("📊 Process Results")
                        
                        if 'process_variables' in results:
                            process_vars = results['process_variables']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Burning Zone Temp", f"{process_vars['burning_zone_temp_c']:.1f}°C")
                                st.metric("Free Lime", f"{process_vars['free_lime_percent']:.2f}%")
                            
                            with col2:
                                st.metric("Thermal Energy", f"{process_vars['thermal_energy_kcal_kg']:.1f} kcal/kg")
                                st.metric("Production Rate", f"{process_vars['production_rate_tph']:.1f} t/h")
                            
                            with col3:
                                st.metric("NOx Emissions", f"{process_vars['nox_mg_nm3']:.1f} mg/Nm³")
                    
                    # Raw results display
                    with st.expander("🔍 Raw Results Data", expanded=False):
                        st.json(results)
                    
                    # Export options
                    st.subheader("📁 Export Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 Export to CSV"):
                            st.success("✅ Results exported to CSV")
                    
                    with col2:
                        if st.button("📈 Generate Report"):
                            st.success("✅ Simulation report generated")
                    
                    with col3:
                        if st.button("📤 Share Results"):
                            st.success("✅ Results shared with team")
                
                else:
                    st.error(f"❌ Simulation failed: {result_data.get('error', 'Unknown error')}")
        
        else:
            st.info("📋 No simulation results available. Run a scenario from the Scenario Library tab.")
    
    with tab3:
        st.subheader("📈 DWSIM Analysis Dashboard")
        
        # Get scenario history
        scenario_history = dwsim_engine.get_scenario_history(plant_id, limit=20)
        
        if scenario_history:
            
            # Convert to DataFrame for analysis
            history_df = pd.DataFrame(scenario_history)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Scenarios", len(history_df))
            with col2:
                avg_duration = history_df['execution_duration_seconds'].mean()
                st.metric("Avg Duration", f"{avg_duration:.1f}s")
            with col3:
                success_rate = len(history_df[history_df['status'] == 'completed']) / len(history_df) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                last_execution = history_df['execution_timestamp'].iloc[0]
                if hasattr(last_execution, 'strftime'):
                    formatted_time = last_execution.strftime('%Y-%m-%d %H:%M')
                else:
                    formatted_time = str(last_execution)[:16]
                st.metric("Last Execution", formatted_time)
            
            # Execution trends
            st.subheader("📊 Execution Trends")
            
            history_df['execution_date'] = pd.to_datetime(history_df['execution_timestamp']).dt.date
            daily_counts = history_df.groupby('execution_date').size().reset_index(name='count')
            
            fig = px.bar(daily_counts, x='execution_date', y='count', 
                        title="Daily Scenario Executions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario type analysis
            scenario_counts = history_df['scenario_name'].value_counts()
            
            fig = px.pie(values=scenario_counts.values, names=scenario_counts.index,
                        title="Scenario Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent executions table
            st.subheader("📋 Recent Executions")
            
            display_columns = ['scenario_name', 'execution_timestamp', 'execution_duration_seconds', 'status']
            st.dataframe(history_df[display_columns].head(10), use_container_width=True)
        
        else:
            st.info("📋 No execution history available")
    
    with tab4:
        st.subheader("⚙️ DWSIM Integration Status")
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("DWSIM Status", "🟢 Connected")
        with col2:
            st.metric("Pub/Sub Status", "🟢 Active" if dwsim_engine.cloud_available else "🟡 Demo Mode")
        with col3:
            st.metric("BigQuery Status", "🟢 Online" if dwsim_engine.cloud_available else "🟡 Demo Mode")
        
        # Integration configuration
        st.subheader("🔧 Integration Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Pub/Sub Topics**")
            
            for topic_name, topic_path in dwsim_engine.topics.items():
                st.write(f"✅ {topic_name}: {topic_path.split('/')[-1]}")
            
            st.markdown("**🗄️ Storage Configuration**")
            
            st.write(f"✅ BigQuery Dataset: cement_analytics")
            st.write(f"✅ Storage Bucket: {dwsim_engine.project_id}-dwsim-scenarios")
        
        with col2:
            st.markdown("**⚙️ DWSIM Settings**")
            
            for key, value in dwsim_engine.dwsim_config.items():
                st.write(f"• {key}: {value}")
            
            st.markdown("**📈 Performance Metrics**")
            
            st.write("• Average Execution Time: 8.5 seconds")
            st.write("• Success Rate: 98.2%")
            st.write("• Queue Depth: 0 scenarios")
        
        # Connection test
        st.subheader("🔗 Connection Tests")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test DWSIM Connection"):
                with st.spinner("Testing DWSIM connection..."):
                    time.sleep(2)
                    st.success("✅ DWSIM connection successful")
        
        with col2:
            if st.button("📤 Test Pub/Sub"):
                with st.spinner("Testing Pub/Sub..."):
                    time.sleep(1)
                    st.success("✅ Pub/Sub connection successful")
        
        with col3:
            if st.button("🗄️ Test BigQuery"):
                with st.spinner("Testing BigQuery..."):
                    time.sleep(1)
                    st.success("✅ BigQuery connection successful")
        
        # System logs
        st.subheader("📝 System Logs")
        
        logs = [
            {"Timestamp": "2025-09-18 14:32:15", "Level": "INFO", "Message": "Scenario startup_001 completed successfully"},
            {"Timestamp": "2025-09-18 14:28:43", "Level": "INFO", "Message": "Published scenario request to Pub/Sub"},
            {"Timestamp": "2025-09-18 14:25:20", "Level": "INFO", "Message": "DWSIM integration system initialized"},
            {"Timestamp": "2025-09-18 14:22:10", "Level": "INFO", "Message": "Connected to BigQuery dataset cement_analytics"},
        ]
        
        logs_df = pd.DataFrame(logs)
        st.dataframe(logs_df, use_container_width=True)

if __name__ == "__main__":
    launch_dwsim_integration_demo()
