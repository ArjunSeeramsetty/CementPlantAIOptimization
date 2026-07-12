# NEW FILE: src/cement_ai_platform/mobile/mobile_dashboard.py
import streamlit as st
from streamlit_javascript import st_javascript
import json
import logging
from google.cloud import firestore
import random
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class MobileCementDashboard:
    """
    Mobile-optimized cement plant dashboard with Firebase push notifications
    """
    
    def __init__(self, project_id: str = None):
        import os
        self.project_id = project_id or os.getenv("CEMENT_GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "cement-ai-opt-38517"
        
        try:
            self.firestore_client = firestore.Client(project=self.project_id)
            self.cloud_available = True
        except Exception as e:
            logger.warning("Firestore initialization warning: %s", e)
            self.firestore_client = None
            self.cloud_available = False
    
    def launch_mobile_dashboard(self):
        """Launch mobile-optimized dashboard"""
        
        # Note: st.set_page_config() is called in the main unified dashboard
        
        # Mobile CSS styling
        self._inject_mobile_css()
        
        # Register service worker for PWA functionality
        self._register_service_worker()
        
        # Title is handled by unified dashboard
        st.markdown("**Real-time cement plant monitoring on mobile**")
        
        # Mobile navigation
        page = st.selectbox("📋 Navigate", [
            "🏠 Dashboard", 
            "🚨 Alerts", 
            "📊 KPIs", 
            "⚙️ Settings"
        ])
        
        if page == "🏠 Dashboard":
            self._render_mobile_dashboard()
        elif page == "🚨 Alerts":
            self._render_mobile_alerts()
        elif page == "📊 KPIs":
            self._render_mobile_kpis()
        elif page == "⚙️ Settings":
            self._render_mobile_settings()
    
    def _inject_mobile_css(self):
        """Inject mobile-optimized CSS"""
        
        mobile_css = """
        <style>
        /* Mobile-first responsive design */
        .main > div {
            padding: 1rem;
        }
        
        /* Large, touch-friendly buttons */
        .stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
            margin: 0.5rem 0;
        }
        
        /* Mobile-optimized metrics */
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            color: white;
        }
        
        /* Alert cards */
        .alert-card {
            border-left: 4px solid #ff4444;
            padding: 1rem;
            margin: 0.5rem 0;
            background: #fff;
            color: #1a1a1a;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Status indicators */
        .status-good { color: #4CAF50; font-weight: bold; }
        .status-warning { color: #FF9800; font-weight: bold; }
        .status-critical { color: #F44336; font-weight: bold; }
        </style>
        """
        
        st.markdown(mobile_css, unsafe_allow_html=True)
    
    def _register_service_worker(self):
        """Register service worker for PWA and push notifications"""
        
        service_worker_js = """
        // Register service worker for PWA functionality
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('SW registered: ', registration);
                })
                .catch(registrationError => {
                    console.log('SW registration failed: ', registrationError);
                });
        }
        
        // Request notification permission
        if ('Notification' in window) {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    console.log('Notification permission granted');
                }
            });
        }
        """
        
        try:
            st_javascript(service_worker_js)
        except Exception as e:
            # Fallback if streamlit_javascript is not available
            st.info("📱 PWA features require streamlit_javascript package")
    
    def _render_mobile_dashboard(self):
        """Render mobile-optimized dashboard"""
        
        # Quick status overview
        st.markdown("### 🏭 Plant Status")
        
        # Simulate real-time data
        status_data = {
            'plant_status': random.choice(['Running', 'Running', 'Running', 'Maintenance']),
            'free_lime': random.uniform(0.8, 2.2),
            'energy_efficiency': random.uniform(88, 96),
            'production_rate': random.uniform(155, 175)
        }
        
        # Mobile-optimized metrics cards
        col1, col2 = st.columns(2)
        
        with col1:
            status_class = "status-good" if status_data['plant_status'] == 'Running' else "status-warning"
            st.markdown(f"""
            <div class="metric-container">
                <h3>🏭 Plant Status</h3>
                <p class="{status_class}">{status_data['plant_status']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            lime_class = "status-good" if status_data['free_lime'] < 1.5 else "status-critical"
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Free Lime</h3>
                <p class="{lime_class}">{status_data['free_lime']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Energy and production
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>⚡ Efficiency</h3>
                <p class="status-good">{status_data['energy_efficiency']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h3>📈 Production</h3>
                <p class="status-good">{status_data['production_rate']:.0f} t/h</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("📊 View Trends"):
                st.info("📈 Opening trend analysis...")
        
        with col6:
            if st.button("🤖 AI Insights"):
                st.info("🧠 Loading AI recommendations...")
        
        # Recent alerts summary
        st.markdown("### 🚨 Recent Alerts")
        
        recent_alerts = [
            {"time": "14:32", "type": "Quality", "message": "Free lime above target", "severity": "High"},
            {"time": "14:15", "type": "Energy", "message": "Kiln fuel efficiency optimized", "severity": "Info"},
            {"time": "13:58", "type": "Equipment", "message": "Raw mill vibration normal", "severity": "Good"}
        ]
        
        for alert in recent_alerts[:3]:  # Show only top 3 on mobile
            severity_class = {
                "High": "status-critical",
                "Info": "status-warning", 
                "Good": "status-good"
            }.get(alert['severity'], "status-good")
            
            st.markdown(f"""
            <div class="alert-card">
                <strong>{alert['time']} - {alert['type']}</strong><br>
                <span class="{severity_class}">{alert['message']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Push notification test
        st.markdown("### 📱 Push Notification Test")
        
        if st.button("🔔 Send Test Notification"):
            st.success("📱 Test notification sent!")
            st.balloons()
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto-refresh (30s)", value=True)
        
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()
    
    def _render_mobile_alerts(self):
        """Render mobile alerts interface"""
        
        st.markdown("### 🚨 Plant Alerts")
        
        # Alert filters
        alert_filter = st.selectbox("Filter by:", ["All Alerts", "Critical", "High", "Medium", "Low"])
        
        # Simulated alerts
        alerts = [
            {
                "id": "AL001",
                "timestamp": "14:32:15",
                "severity": "Critical",
                "type": "Quality",
                "message": "Free lime exceeded 2.0% threshold",
                "equipment": "Kiln #1",
                "action": "Increase fuel rate recommended"
            },
            {
                "id": "AL002", 
                "timestamp": "14:28:43",
                "severity": "High",
                "type": "Equipment",
                "message": "Raw mill vibration trending upward",
                "equipment": "Raw Mill #1",
                "action": "Schedule inspection"
            },
            {
                "id": "AL003",
                "timestamp": "14:15:20",
                "severity": "Medium",
                "type": "Energy", 
                "message": "Kiln energy efficiency below target",
                "equipment": "Kiln #1",
                "action": "AI optimization applied"
            }
        ]
        
        # Filter alerts
        if alert_filter != "All Alerts":
            alerts = [a for a in alerts if a['severity'] == alert_filter]
        
        # Display alerts
        for alert in alerts:
            severity_colors = {
                "Critical": "#F44336",
                "High": "#FF9800",
                "Medium": "#2196F3",
                "Low": "#4CAF50"
            }
            
            color = severity_colors.get(alert['severity'], "#666")
            
            st.markdown(f"""
            <div class="alert-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{alert['id']}</strong>
                    <span style="color: {color};">{alert['severity']}</span>
                </div>
                <div style="margin: 0.5rem 0;">
                    <strong>{alert['type']} - {alert['equipment']}</strong><br>
                    {alert['message']}
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    🕒 {alert['timestamp']} | 🔧 {alert['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Acknowledge", key=f"ack_{alert['id']}"):
                    st.success(f"Alert {alert['id']} acknowledged")
            with col2:
                if st.button("📞 Call Expert", key=f"call_{alert['id']}"):
                    st.info("Connecting to expert...")
            with col3:
                if st.button("📝 Add Note", key=f"note_{alert['id']}"):
                    note = st.text_input("Note:", key=f"note_input_{alert['id']}")
    
    def _render_mobile_kpis(self):
        """Render mobile KPIs interface"""
        
        st.markdown("### 📊 Key Performance Indicators")
        
        # Time period selector
        period = st.selectbox("📅 Period", ["Last Hour", "Last 4 Hours", "Last 12 Hours", "Today"])
        
        # Simulated KPI data
        kpis = {
            "Production": {
                "value": f"{random.uniform(155, 175):.0f} t/h",
                "target": "167 t/h",
                "trend": "↗️ +2.3%",
                "status": "Good"
            },
            "Energy Efficiency": {
                "value": f"{random.uniform(88, 96):.1f}%",
                "target": "90%",
                "trend": "↗️ +1.2%",
                "status": "Good"
            },
            "Free Lime": {
                "value": f"{random.uniform(0.8, 2.2):.1f}%",
                "target": "<1.5%",
                "trend": "↘️ -0.3%",
                "status": "Warning" if random.random() > 0.7 else "Good"
            },
            "Equipment Health": {
                "value": f"{random.uniform(85, 98):.0f}%",
                "target": ">90%", 
                "trend": "→ 0%",
                "status": "Good"
            }
        }
        
        # Display KPIs in mobile-friendly format
        for kpi_name, kpi_data in kpis.items():
            
            status_color = {
                "Good": "#4CAF50",
                "Warning": "#FF9800", 
                "Critical": "#F44336"
            }.get(kpi_data['status'], "#4CAF50")
            
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, {status_color}33 0%, {status_color}66 100%);">
                <h4>{kpi_name}</h4>
                <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                    {kpi_data['value']}
                </div>
                <div style="font-size: 0.9rem;">
                    Target: {kpi_data['target']} | {kpi_data['trend']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_mobile_settings(self):
        """Render mobile settings interface"""
        
        st.markdown("### ⚙️ Mobile App Settings")
        
        # Notification preferences
        st.markdown("**📱 Notifications**")
        
        notifications = {
            "Push Notifications": st.toggle("Push notifications", value=True),
            "Critical Alerts": st.toggle("Critical alerts (immediate)", value=True), 
            "Daily Summary": st.toggle("Daily summary report", value=True),
            "Weekly Report": st.toggle("Weekly performance report", value=False),
            "Maintenance Alerts": st.toggle("Maintenance reminders", value=True)
        }
        
        # Display preferences
        st.markdown("**🎨 Display**")
        
        display_prefs = {
            "Dark Mode": st.toggle("Dark mode", value=False),
            "Auto Refresh": st.toggle("Auto-refresh data", value=True),
            "Compact View": st.toggle("Compact view", value=True)
        }
        
        refresh_interval = st.selectbox("Refresh interval", ["30 seconds", "1 minute", "5 minutes"])
        
        # Language and region
        st.markdown("**🌐 Regional Settings**")
        
        language = st.selectbox("Language", ["English", "हिंदी", "ગુજરાતી"])
        timezone = st.selectbox("Timezone", ["Asia/Kolkata", "Asia/Dubai", "UTC"])
        units = st.selectbox("Units", ["Metric", "Imperial"])
        
        # Security
        st.markdown("**🔒 Security**")
        
        biometric_auth = st.toggle("Biometric authentication", value=False)
        auto_logout = st.selectbox("Auto-logout after", ["15 minutes", "30 minutes", "1 hour", "Never"])
        
        # About section
        st.markdown("**ℹ️ About**")
        
        st.info(f"""
        **Cement Plant Mobile Monitor**
        Version: 2.1.0
        Build: 2025.09.18
        
        Connected to: {self.project_id}
        Status: Online ✅
        
        © 2025 Cement AI Platform
        """)

# Demo launcher
def launch_mobile_demo():
    """Launch mobile dashboard demo"""
    mobile_app = MobileCementDashboard()
    mobile_app.launch_mobile_dashboard()

if __name__ == "__main__":
    launch_mobile_demo()
