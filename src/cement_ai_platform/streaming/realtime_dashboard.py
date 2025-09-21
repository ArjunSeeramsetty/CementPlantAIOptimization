import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import threading
import queue
import random

class RealTimeStreamingDashboard:
    """Real-time streaming dashboard for POC demonstration"""
    
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.historical_data = []
        
    def start_demo_dashboard(self):
        """Launch real-time streaming dashboard"""
        
        # Note: st.set_page_config() is called in the main unified dashboard
        
        st.title("🏭 Cement Plant Real-Time Data Streaming")
        st.markdown("**Live sensor data via Google Cloud Pub/Sub simulation**")
        
        # Initialize streaming simulator
        if 'streaming_simulator' not in st.session_state:
            try:
                from cement_ai_platform.streaming.pubsub_simulator import CementPlantPubSubSimulator, RealTimeDataProcessor
                st.session_state.streaming_simulator = CementPlantPubSubSimulator()
                st.session_state.data_processor = RealTimeDataProcessor()
                st.session_state.streaming_active = False
            except ImportError as e:
                st.error(f"❌ Error importing streaming modules: {e}")
                st.stop()
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Streaming Controls")
            
            if st.button("🚀 Start Streaming"):
                if not st.session_state.streaming_active:
                    st.session_state.streaming_simulator.start_streaming_simulation(interval_seconds=2)
                    st.session_state.streaming_active = True
                    st.success("🚀 Streaming started!")
            
            if st.button("⏹️ Stop Streaming"):
                if st.session_state.streaming_active:
                    st.session_state.streaming_simulator.stop_streaming()
                    st.session_state.streaming_active = False
                    st.info("⏹️ Streaming stopped")
            
            streaming_interval = st.slider("Streaming Interval (seconds)", 1, 10, 2)
            
            st.header("📊 Data Filters")
            show_anomalies_only = st.checkbox("Show Anomalies Only")
            show_alerts = st.checkbox("Show AI Alerts", value=True)
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Live Process Variables")
            chart_placeholder = st.empty()
            
            st.subheader("🤖 AI Controller Response")
            controller_placeholder = st.empty()
        
        with col2:
            st.subheader("🚨 Live Alerts")
            alerts_placeholder = st.empty()
            
            st.subheader("📊 Streaming Statistics")
            stats_placeholder = st.empty()
        
        # Simulate real-time updates
        self._simulate_realtime_updates(
            chart_placeholder, 
            controller_placeholder, 
            alerts_placeholder, 
            stats_placeholder
        )
    
    def _simulate_realtime_updates(self, chart_placeholder, controller_placeholder, 
                                 alerts_placeholder, stats_placeholder):
        """Simulate real-time dashboard updates"""
        
        # Generate simulated data for demo
        if len(self.historical_data) < 50:  # Generate initial data
            for i in range(50):
                data_point = {
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=50-i),
                    'free_lime_percent': random.uniform(0.8, 2.3),
                    'burning_zone_temp_c': random.uniform(1430, 1470),
                    'fuel_rate_tph': random.uniform(14.5, 18.2),
                    'kiln_speed_rpm': random.uniform(2.9, 4.0)
                }
                self.historical_data.append(data_point)
        
        # Add new data point
        new_data = {
            'timestamp': pd.Timestamp.now(),
            'free_lime_percent': random.uniform(0.8, 2.3),
            'burning_zone_temp_c': random.uniform(1430, 1470),
            'fuel_rate_tph': random.uniform(14.5, 18.2),
            'kiln_speed_rpm': random.uniform(2.9, 4.0)
        }
        self.historical_data.append(new_data)
        
        # Keep only last 50 data points
        if len(self.historical_data) > 50:
            self.historical_data = self.historical_data[-50:]
        
        # Update live charts
        if chart_placeholder:
            self._update_live_charts(chart_placeholder)
        
        # Update controller status
        if controller_placeholder:
            self._update_controller_status(controller_placeholder, new_data)
        
        # Update alerts
        if alerts_placeholder:
            self._update_alerts(alerts_placeholder, new_data)
        
        # Update statistics
        if stats_placeholder:
            self._update_statistics(stats_placeholder)
        
        # Auto-refresh every 3 seconds
        time.sleep(3)
        st.rerun()
    
    def _update_live_charts(self, placeholder):
        """Update live process variable charts"""
        
        if not self.historical_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        
        # Create subplots
        fig = go.Figure()
        
        # Add traces for key variables
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['free_lime_percent'],
            mode='lines+markers',
            name='Free Lime %',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['burning_zone_temp_c']/100,  # Scale for display
            mode='lines+markers',
            name='Burning Zone Temp (°C/100)',
            line=dict(color='orange', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Real-Time Process Variables",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            showlegend=True
        )
        
        placeholder.plotly_chart(fig, use_container_width=True)
    
    def _update_controller_status(self, placeholder, latest_data):
        """Update AI controller status"""
        
        # Simulate controller response
        placeholder.markdown(f"""
        **🤖 Latest AI Recommendations:**
        - **Fuel Rate**: {latest_data['fuel_rate_tph']:.2f} t/h (Current)
        - **Kiln Speed**: {latest_data['kiln_speed_rpm']:.2f} rpm (Current)
        - **Free Lime**: {latest_data['free_lime_percent']:.2f}% (Target: <1.5%)
        - **Control Health**: {"Good" if latest_data['free_lime_percent'] < 2.0 else "Alert"}
        
        **🔮 AI Predictions:**
        - **Next Free Lime**: {latest_data['free_lime_percent'] * 0.95:.2f}% (in 15 min)
        - **Energy Efficiency**: 92.3%
        - **Quality Status**: {"✅ On Target" if latest_data['free_lime_percent'] < 2.0 else "⚠️ Needs Attention"}
        """)
    
    def _update_alerts(self, placeholder, latest_data):
        """Update alerts panel"""
        
        alerts = []
        
        # Check for alerts
        if latest_data['free_lime_percent'] > 2.0:
            alerts.append("🚨 HIGH: Free lime above 2.0%")
        
        if latest_data['burning_zone_temp_c'] < 1430:
            alerts.append("⚠️ WARNING: Low burning zone temperature")
        
        if latest_data['fuel_rate_tph'] > 18.0:
            alerts.append("ℹ️ INFO: High fuel consumption")
        
        # Display alerts
        if alerts:
            for alert in alerts:
                placeholder.markdown(alert)
        else:
            placeholder.markdown("✅ No active alerts")
    
    def _update_statistics(self, placeholder):
        """Update streaming statistics"""
        
        placeholder.metric("Data Points", len(self.historical_data))
        placeholder.metric("Streaming Status", "🟢 Active" if st.session_state.get('streaming_active', False) else "🔴 Inactive")
        placeholder.metric("Last Update", pd.Timestamp.now().strftime('%H:%M:%S'))

# Demo launcher
def launch_streaming_demo():
    """Launch the real-time streaming demo"""
    dashboard = RealTimeStreamingDashboard()
    dashboard.start_demo_dashboard()

if __name__ == "__main__":
    launch_streaming_demo()
