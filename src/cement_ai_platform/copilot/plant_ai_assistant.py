# src/cement_ai_platform/copilot/plant_ai_assistant.py
"""
Plant AI Assistant (Copilot) with Gemini Integration
Intelligent cement plant operations assistant with troubleshooting and optimization capabilities
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any
import json
import time
from datetime import datetime
import random
import html
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.getLogger(__name__).warning("Google Generative AI not available - using simulation mode")

logger = logging.getLogger(__name__)

class PlantAIAssistant:
    """
    Gemini-powered Plant AI Assistant for cement operations
    """
    
    def __init__(self, api_key: str = None):
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GOOGLE_GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        # Initialize Gemini API
        if self.api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                # Test the API key with a simple request
                test_response = self.model.generate_content("Hello")
                logger.info("Gemini API initialized successfully with key prefix: %s...", self.api_key[:10])
            except Exception as e:
                logger.error("Failed to initialize Gemini with API key: %s", e)
                logger.warning("Falling back to simulation mode")
                self.model = None
                self.api_key = None
        else:
            # Fallback simulation for demo
            self.model = None
            if GEMINI_AVAILABLE:
                logger.warning("No Gemini API key found. Using simulation mode.")
                logger.info("Set GOOGLE_GEMINI_API_KEY or GOOGLE_API_KEY in your .env file")
                
        # Initialize Vertex AI as alternative/fallback
        try:
            from cement_ai_platform.gcp.production_services import get_production_services
            self.gcp_services = get_production_services()
            self.vertex_available = self.gcp_services.gcp_available
        except Exception:
            self.gcp_services = None
            self.vertex_available = False
        
        # Plant knowledge base
        self.plant_knowledge = {
            "process_parameters": {
                "kiln_temperature": {"normal_range": [1430, 1470], "unit": "°C", "impact": "clinker_quality"},
                "free_lime": {"normal_range": [0.8, 1.5], "unit": "%", "impact": "cement_strength"},
                "o2_percentage": {"normal_range": [2.5, 4.0], "unit": "%", "impact": "combustion_efficiency"},
                "fuel_rate": {"normal_range": [14, 18], "unit": "t/h", "impact": "thermal_energy"},
                "feed_rate": {"normal_range": [150, 180], "unit": "t/h", "impact": "production_rate"}
            },
            
            "troubleshooting": {
                "high_free_lime": {
                    "causes": ["Low burning zone temperature", "Insufficient residence time", "Poor fuel distribution"],
                    "solutions": ["Increase fuel rate", "Reduce kiln speed", "Check burner alignment"],
                    "prevention": ["Monitor coal fineness", "Maintain consistent feed rate", "Regular kiln shell temperature checks"]
                },
                "low_kiln_temperature": {
                    "causes": ["Insufficient fuel", "Air/fuel ratio imbalance", "Heat loss through shell"],
                    "solutions": ["Increase fuel rate", "Reduce excess air", "Check refractory condition"],
                    "prevention": ["Regular fuel quality analysis", "Maintain burner in good condition", "Monitor shell temperature"]
                },
                "high_vibration": {
                    "causes": ["Bearing wear", "Misalignment", "Imbalanced load"],
                    "solutions": ["Schedule bearing inspection", "Check alignment", "Balance material distribution"],
                    "prevention": ["Regular lubrication", "Vibration monitoring", "Proper maintenance schedule"]
                }
            },
            
            "optimization_strategies": {
                "energy_efficiency": [
                    "Optimize air/fuel ratio for complete combustion",
                    "Minimize excess air to reduce heat loss",
                    "Use waste heat recovery systems",
                    "Maintain optimal kiln speed for residence time"
                ],
                "quality_improvement": [
                    "Maintain consistent raw meal chemistry",
                    "Control burning zone temperature within optimal range",
                    "Optimize clinker cooling rate",
                    "Monitor and control SO3 content"
                ],
                "alternative_fuel_usage": [
                    "Gradually increase TSR to maintain stability",
                    "Match fuel characteristics to process requirements",
                    "Monitor impact on emissions and quality",
                    "Optimize fuel distribution between kiln and calciner"
                ]
            },
            
            "safety_procedures": {
                "emergency_shutdown": [
                    "Immediately stop fuel feeding",
                    "Maintain kiln rotation at reduced speed",
                    "Monitor shell temperatures during cooldown",
                    "Ensure proper ventilation"
                ],
                "high_temperature_alarm": [
                    "Reduce fuel rate gradually",
                    "Increase cooling air if available",
                    "Check for process irregularities",
                    "Prepare for potential emergency shutdown"
                ]
            }
        }
        
        # Conversation history
        self.conversation_history = []
    
    def generate_response(self, query: str, context: Dict = None) -> str:
        """Generate AI response using Gemini or fallback simulation"""
        
        if self.model or (hasattr(self, 'vertex_available') and self.vertex_available):
            return self._generate_gemini_response(query, context)
        else:
            return self._generate_simulated_response(query, context)
    
    def _filter_sensitive_context(self, context: Dict) -> Dict:
        """Filter out sensitive operational data from context"""
        safe_keys = ['kiln_temp_c', 'free_lime_pct', 'energy_efficiency_pct']
        return {k: v for k, v in context.items() if k in safe_keys}
    
    def _generate_gemini_response(self, query: str, context: Dict = None) -> str:
        """Generate response using Gemini API"""
        
        # Filter sensitive data from context
        safe_context = self._filter_sensitive_context(context) if context else None
        
        # Build enhanced prompt with plant context
        system_prompt = f"""
        You are an expert cement plant AI assistant with deep knowledge of cement manufacturing processes.
        
        Current plant context:
        {json.dumps(safe_context, indent=2) if safe_context else "No current context available"}
        
        Plant knowledge base includes:
        - Process parameters and their normal ranges
        - Common troubleshooting scenarios and solutions
        - Optimization strategies for energy, quality, and alternative fuels
        - Safety procedures and emergency protocols
        
        Provide detailed, technical responses that are practical and actionable for cement plant operators and engineers.
        
        User query: {query}
        """
        
        # Try Vertex AI first if available
        if hasattr(self, 'gcp_services') and self.gcp_services and self.vertex_available:
            try:
                res = self.gcp_services.query_gemini_pro(system_prompt)
                if res and res.get('success'):
                    return res['response']
            except Exception as e:
                logger.warning("Vertex AI query failed in plant assistant: %s", e)

        # Try public API key
        if self.model:
            try:
                response = self.model.generate_content(system_prompt)
                return response.text
            except Exception as e:
                logger.error("Error generating Gemini response: %s", e)
                return self._generate_simulated_response(query, context)
        else:
            return self._generate_simulated_response(query, context)
    
    def _generate_simulated_response(self, query: str, context: Dict = None) -> str:
        """Generate simulated intelligent response for demo purposes"""
        
        query_lower = query.lower()
        
        # Pattern matching for common queries
        if any(keyword in query_lower for keyword in ['free lime', 'lime']):
            return self._generate_free_lime_response(context)
        elif any(keyword in query_lower for keyword in ['temperature', 'temp', 'hot', 'cold']):
            return self._generate_temperature_response(context)
        elif any(keyword in query_lower for keyword in ['fuel', 'tsr', 'alternative']):
            return self._generate_fuel_response(context)
        elif any(keyword in query_lower for keyword in ['vibration', 'maintenance']):
            return self._generate_maintenance_response(context)
        elif any(keyword in query_lower for keyword in ['optimize', 'efficiency', 'improve']):
            return self._generate_optimization_response(context)
        elif any(keyword in query_lower for keyword in ['quality', 'strength', 'blaine']):
            return self._generate_quality_response(context)
        else:
            return self._generate_general_response(query, context)
    
    def _generate_free_lime_response(self, context: Dict) -> str:
        current_lime = context.get('free_lime_pct', 1.2) if context else 1.2
        
        if current_lime > 1.8:
            return f"""🔍 **Free Lime Analysis (Current: {current_lime:.2f}%)**

**Status**: ⚠️ ELEVATED - Immediate attention required

**Root Cause Analysis**:
• Burning zone temperature may be insufficient for complete calcination
• Possible fuel distribution issues in the kiln
• Raw meal fineness or chemistry variations

**Immediate Actions**:
1. **Increase fuel rate by 0.3-0.5 t/h** to raise burning zone temperature
2. **Check burner condition** and flame characteristics  
3. **Verify raw meal chemistry** - particularly LSF and silica modulus
4. **Monitor kiln shell temperatures** for heat distribution

**Expected Results**:
• Free lime should reduce by 0.3-0.5% within 30-45 minutes
• Monitor clinker temperature and adjust cooling accordingly
• Quality impact on 28-day strength expected to improve by 3-5 MPa

**Preventive Measures**:
• Implement continuous raw meal blending optimization
• Schedule burner maintenance to ensure optimal flame shape
• Consider alternative fuel gradual reduction if TSR is high"""

        else:
            return f"""✅ **Free Lime Analysis (Current: {current_lime:.2f}%)**

**Status**: GOOD - Within optimal range

**Current Performance**:
• Free lime is well-controlled and within target range
• Indicates good burning conditions and fuel efficiency
• Clinker quality should be stable

**Optimization Opportunities**:
1. **Fine-tune fuel rate** - Potential for slight reduction while maintaining quality
2. **Monitor trends** - Watch for any gradual drift patterns
3. **Alternative fuel increase** - Current conditions favorable for TSR optimization

**Recommendations**:
• Maintain current kiln speed and fuel distribution
• Consider gradual TSR increase by 2-3% if alternative fuels available
• Continue monitoring every 30 minutes for trend analysis"""
    
    def _generate_temperature_response(self, context: Dict) -> str:
        current_temp = context.get('kiln_temp_c', 1450) if context else 1450
        
        return f"""🌡️ **Temperature Analysis (Current: {current_temp:.0f}°C)**

**Assessment**: {'⚠️ HIGH' if current_temp > 1465 else '✅ OPTIMAL' if current_temp >= 1440 else '🔴 LOW'}

**Impact Analysis**:
• **Clinker Quality**: {'Risk of overburning' if current_temp > 1470 else 'Good burnability' if current_temp >= 1440 else 'Incomplete burning risk'}
• **Refractory Life**: {'Accelerated wear' if current_temp > 1470 else 'Normal wear rate'}
• **Energy Efficiency**: {'High thermal energy' if current_temp > 1465 else 'Optimal energy use'}

**Recommended Actions**:
{self._get_temperature_recommendations(current_temp)}

**Process Correlation**:
• Each 10°C change typically affects free lime by ±0.1-0.2%
• Fuel rate adjustment of ±0.2 t/h needed per 10°C target change
• Monitor preheater exit temperature for overall system balance"""
    
    def _get_temperature_recommendations(self, temp: float) -> str:
        if temp > 1470:
            return """1. **URGENT**: Reduce fuel rate by 0.3-0.5 t/h immediately
2. **Check**: Verify air/fuel distribution and excess air levels  
3. **Monitor**: Shell temperatures and refractory condition
4. **Prepare**: For potential emergency actions if temp continues rising"""
        elif temp > 1465:
            return """1. **Reduce fuel rate** by 0.2-0.3 t/h
2. **Optimize air distribution** to improve heat transfer
3. **Monitor closely** for stabilization within 20-30 minutes
4. **Check** for any process disturbances"""
        elif temp < 1440:
            return """1. **Increase fuel rate** by 0.2-0.4 t/h
2. **Check fuel quality** and burner condition
3. **Verify air/fuel ratio** optimization
4. **Monitor free lime** for potential increase"""
        else:
            return """1. **Maintain current settings** - temperature is optimal
2. **Fine-tune** minor adjustments based on quality trends
3. **Monitor stability** and prepare for fuel/feed changes"""
    
    def _generate_fuel_response(self, context: Dict) -> str:
        current_tsr = context.get('current_tsr_percentage', 18.5) if context else 18.5
        
        return f"""⛽ **Fuel & TSR Analysis (Current TSR: {current_tsr:.1f}%)**

**Alternative Fuel Status**: {'🟢 Good potential for increase' if current_tsr < 30 else '⚠️ Monitor stability closely'}

**Optimization Strategy**:
1. **Target TSR**: {min(35, current_tsr + 8):.1f}% (gradual increase recommended)
2. **Fuel Mix Recommendation**:
   • RDF from MSW: High availability, good for calciner
   • Biomass: Excellent environmental benefit, stable combustion
   • Plastic waste: High calorific value, requires careful handling

**Implementation Plan**:
• **Week 1-2**: Increase TSR by 3-4% while monitoring quality
• **Week 3-4**: Fine-tune fuel distribution between kiln and calciner  
• **Week 5+**: Optimize to target TSR with quality confirmation

**Key Monitoring Points**:
• Calciner temperature stability (±15°C)
• Free lime variation (<100 ppm increase)
• NOx emissions (typically reduce by 5-10%)
• Preheater O2 levels for combustion optimization

**Economic Impact**:
• Estimated fuel cost saving: ₹8,000-12,000/hour at target TSR
• CO2 reduction: 15-20% per unit thermal energy
• Payback period: 3-6 months including handling infrastructure"""
    
    def _generate_maintenance_response(self, context: Dict) -> str:
        vibration = context.get('vibration_mm_s', 4.5) if context else 4.5
        
        return f"""🔧 **Maintenance Analysis (Current Vibration: {vibration:.1f} mm/s)**

**Equipment Health**: {'🔴 CRITICAL' if vibration > 7 else '⚠️ ATTENTION NEEDED' if vibration > 6 else '🟡 MONITOR' if vibration > 5 else '✅ GOOD'}

**Predictive Analysis**:
• **Current Status**: {'Schedule immediate inspection' if vibration > 6.5 else 'Plan maintenance within 7 days' if vibration > 5.5 else 'Normal monitoring schedule'}
• **Failure Risk**: {f'{min(85, (vibration - 3) * 20):.0f}%' if vibration > 3 else '15%'}
• **Estimated Time to Action**: {'< 24 hours' if vibration > 7 else '< 7 days' if vibration > 6 else '< 30 days'}

**Recommended Actions**:
{self._get_maintenance_recommendations(vibration)}

**Cost Impact Analysis**:
• **Preventive Cost**: ₹2-5 lakhs (bearing replacement + labor)
• **Failure Cost**: ₹25-50 lakhs (equipment damage + downtime)  
• **Production Impact**: 8-24 hours downtime if emergency repair needed

**Scheduling Recommendation**:
• **Optimal Window**: Next planned shutdown or within 7 days
• **Required Resources**: Mechanical team + vibration specialist
• **Spare Parts**: Ensure bearing inventory availability"""
    
    def _get_maintenance_recommendations(self, vibration: float) -> str:
        if vibration > 7:
            return """🚨 **CRITICAL - IMMEDIATE ACTION**:
1. Reduce load if possible while maintaining minimum kiln speed
2. Schedule emergency bearing inspection within 24 hours
3. Prepare for potential unplanned shutdown
4. Alert maintenance team and ensure spare parts availability"""
        elif vibration > 6:
            return """⚠️ **HIGH PRIORITY**:
1. Schedule detailed vibration analysis within 48 hours
2. Check lubrication system and oil analysis
3. Plan bearing replacement during next available window
4. Monitor continuously with 4-hour frequency"""
        elif vibration > 5:
            return """🟡 **MONITOR CLOSELY**:
1. Increase monitoring frequency to daily
2. Schedule oil analysis and bearing inspection
3. Plan maintenance within next 2-4 weeks  
4. Check for any operational changes causing increase"""
        else:
            return """✅ **NORMAL MAINTENANCE**:
1. Continue standard monitoring schedule
2. Routine lubrication and inspection as per schedule
3. Good opportunity to focus on other equipment
4. Consider this as baseline for future comparisons"""
    
    def _generate_optimization_response(self, context: Dict) -> str:
        return """🎯 **Plant Optimization Strategy**

**Multi-Parameter Optimization Approach**:

**1. Energy Efficiency (Target: 5-8% reduction)**
• **Thermal Energy**: Optimize air/fuel ratio for complete combustion
• **Electrical Energy**: Variable speed drives on major motors
• **Waste Heat Recovery**: Implement AQC system if not already present
• **Process Integration**: Balance kiln-preheater-cooler system

**2. Quality Consistency (Target: ±2% variation)**
• **Raw Material**: Advanced blending with predictive control
• **Burning Process**: Maintain burning zone temperature ±5°C
• **Cooling**: Optimize cooler performance for consistent temperature
• **Cement Grinding**: Integrate pyro-process feedback

**3. Alternative Fuel Usage (Target: 10-15% TSR increase)**
• **Gradual Implementation**: 2-3% TSR increase per month
• **Fuel Characterization**: Match fuel properties to process needs
• **Distribution Optimization**: Kiln vs calciner fuel split
• **Quality Impact**: Monitor and compensate for variations

**4. Production Optimization**
• **Throughput**: Optimize residence time vs production rate
• **Stability**: Minimize process variations and interruptions
• **Equipment Reliability**: Predictive maintenance implementation
• **Operational Excellence**: Advanced process control integration

**Implementation Priority**:
1. **Phase 1 (Month 1-2)**: Energy monitoring and basic optimization
2. **Phase 2 (Month 3-4)**: Quality control system upgrade
3. **Phase 3 (Month 5-6)**: Alternative fuel gradual increase
4. **Phase 4 (Month 6+)**: Advanced process integration

**Expected ROI**: 15-25% within 12 months through energy savings and quality improvements"""
    
    def _generate_quality_response(self, context: Dict) -> str:
        return """🏆 **Quality Management Strategy**

**Key Quality Parameters**:

**1. Compressive Strength Optimization**
• **Target**: 28-day strength 45-52 MPa
• **Control**: Clinker mineral composition (C3S, C2S, C3A)
• **Factors**: Burning zone temperature, residence time, cooling rate
• **Monitoring**: Daily testing with 4-hour online prediction

**2. Fineness Control (Blaine)**
• **Target**: 3200-3800 cm²/g depending on cement grade
• **Control**: Cement mill operation and separator efficiency
• **Optimization**: Grinding media gradation and mill loading
• **Integration**: Feedback from pyro-process to grinding

**3. Chemical Composition**
• **LSF (Lime Saturation Factor)**: 92-98%
• **Silica Modulus**: 2.0-2.6
• **Alumina Modulus**: 1.2-2.0
• **SO3 Content**: 2.5-3.5% for optimal strength development

**Quality Prediction System**:
• **Free Lime Prediction**: Online calciner sensors + AI model
• **Strength Prediction**: Combine process data + lab results  
• **Early Warning**: Detect quality drift 2-4 hours in advance
• **Automatic Correction**: Integrate with process control system

**Benefits**:
• Reduce off-spec cement by 60-80%
• Improve customer satisfaction and premium pricing
• Reduce quality testing costs by 30-40%
• Enable proactive rather than reactive quality management"""
    
    def _generate_general_response(self, query: str, context: Dict) -> str:
        return f"""🤖 **AI Assistant Response**

I understand you're asking about: "{query}"

Based on my analysis of cement plant operations, here are some general insights:

**Current Plant Status** (if context available):
{self._format_context_summary(context) if context else "No current plant data available for analysis."}

**How I can help you**:
• **Process Troubleshooting**: Ask about specific issues like high free lime, temperature problems, or equipment issues
• **Optimization Guidance**: Get recommendations for energy, quality, or production improvements  
• **Maintenance Planning**: Predictive maintenance insights and scheduling recommendations
• **Alternative Fuels**: TSR optimization and fuel mix strategies
• **Quality Management**: Strength, fineness, and chemical composition guidance
• **Safety Procedures**: Emergency protocols and safety best practices

**Example questions you can ask**:
• "Why is my free lime high and how can I fix it?"
• "How can I optimize my alternative fuel usage?"
• "What's the best maintenance strategy for high vibration?"
• "How do I improve energy efficiency in my kiln?"

Feel free to ask me anything about cement plant operations - I'm here to help optimize your process!"""
    
    def _format_context_summary(self, context: Dict) -> str:
        summary_parts = []
        
        if 'kiln_temp_c' in context:
            summary_parts.append(f"• Kiln Temperature: {context['kiln_temp_c']:.0f}°C")
        if 'free_lime_pct' in context:
            summary_parts.append(f"• Free Lime: {context['free_lime_pct']:.2f}%")
        if 'production_rate_tph' in context:
            summary_parts.append(f"• Production Rate: {context['production_rate_tph']:.0f} t/h")
        if 'energy_efficiency_pct' in context:
            summary_parts.append(f"• Energy Efficiency: {context['energy_efficiency_pct']:.1f}%")
        
        return "\n".join(summary_parts) if summary_parts else "Basic plant monitoring data available."

def launch_plant_ai_assistant():
    """Launch Plant AI Assistant Interface"""
    
    # Note: st.set_page_config() is called in the main unified dashboard
    
    st.title("🤖 Cement Plant AI Assistant")
    st.markdown("**Your intelligent copilot for cement plant operations**")
    
    # Initialize assistant
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = PlantAIAssistant()
        st.session_state.conversation_history = []
    
    assistant = st.session_state.ai_assistant
    
    # Sidebar with plant context
    with st.sidebar:
        st.header("🏭 Current Plant Context")
        
        # Simulate current plant data
        plant_context = {
            'kiln_temp_c': st.slider("Kiln Temperature (°C)", 1400, 1500, 1450),
            'free_lime_pct': st.slider("Free Lime (%)", 0.5, 3.0, 1.2),
            'production_rate_tph': st.slider("Production Rate (t/h)", 140, 200, 165),
            'energy_efficiency_pct': st.slider("Energy Efficiency (%)", 80, 100, 92),
            'vibration_mm_s': st.slider("Equipment Vibration (mm/s)", 2.0, 10.0, 4.5),
            'current_tsr_percentage': st.slider("Current TSR (%)", 10.0, 40.0, 18.5)
        }
        
        st.header("💡 Quick Questions")
        
        quick_questions = [
            "Why is my free lime high?",
            "How to optimize fuel efficiency?", 
            "When should I schedule maintenance?",
            "How to increase TSR safely?",
            "What's causing temperature variations?",
            "How to improve cement quality?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.current_query = question
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat with Plant AI")
        
        # Display conversation history
        if st.session_state.conversation_history:
            for i, (query, response, timestamp) in enumerate(st.session_state.conversation_history):
                
                # User message
                st.markdown(f"""
                <div style="background: #e3f2fd; color: #1a1a1a; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <strong>You ({timestamp}):</strong><br>
                    {html.escape(query)}
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                st.markdown(f"""
                <div style="background: #f5f5f5; color: #1a1a1a; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <strong>🤖 Plant AI Assistant:</strong><br>
                    {html.escape(response)}
                </div>
                """, unsafe_allow_html=True)
        
        # Query input
        query_input = st.text_input(
            "Ask me anything about cement plant operations:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., Why is my free lime high and how can I reduce it?",
            key="query_input"
        )
        
        col_send, col_clear = st.columns([1, 4])
        
        with col_send:
            if st.button("🚀 Send") or st.session_state.get('current_query'):
                if query_input or st.session_state.get('current_query'):
                    
                    final_query = query_input or st.session_state.get('current_query', '')
                    
                    with st.spinner("🤔 AI Assistant is thinking..."):
                        response = assistant.generate_response(final_query, plant_context)
                        
                        # Add to conversation history
                        timestamp = datetime.now().strftime("%H:%M")
                        st.session_state.conversation_history.append((final_query, response, timestamp))
                        
                        # Clear current query
                        if 'current_query' in st.session_state:
                            del st.session_state.current_query
                    
                    st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()
    
    with col2:
        st.subheader("📊 Plant Status")
        
        # Status indicators based on context
        temp_status = "🔥" if plant_context['kiln_temp_c'] > 1465 else "🌡️"
        lime_status = "⚠️" if plant_context['free_lime_pct'] > 1.8 else "✅"
        efficiency_status = "⚡" if plant_context['energy_efficiency_pct'] > 90 else "🔋"
        vibration_status = "🚨" if plant_context['vibration_mm_s'] > 6.5 else "📳"
        
        st.markdown(f"""
        **Current Status:**
        
        {temp_status} **Temperature**: {plant_context['kiln_temp_c']:.0f}°C
        
        {lime_status} **Free Lime**: {plant_context['free_lime_pct']:.2f}%
        
        {efficiency_status} **Efficiency**: {plant_context['energy_efficiency_pct']:.1f}%
        
        {vibration_status} **Vibration**: {plant_context['vibration_mm_s']:.1f} mm/s
        
        🔥 **TSR**: {plant_context['current_tsr_percentage']:.1f}%
        
        📦 **Production**: {plant_context['production_rate_tph']:.0f} t/h
        """)
        
        st.subheader("🎯 AI Capabilities")
        
        capabilities = [
            "🔍 Process Troubleshooting",
            "⚡ Energy Optimization", 
            "🏆 Quality Management",
            "🔧 Predictive Maintenance",
            "🔥 Alternative Fuel Strategy",
            "📈 Performance Analytics",
            "🚨 Safety Recommendations",
            "📊 KPI Tracking"
        ]
        
        for cap in capabilities:
            st.markdown(f"• {cap}")
        
        st.subheader("📚 Knowledge Areas")
        
        st.markdown("""
        • **Process Parameters** - Temperature, pressure, flow rates
        • **Quality Control** - Strength, fineness, chemical composition  
        • **Energy Management** - Thermal & electrical optimization
        • **Maintenance** - Predictive analytics & scheduling
        • **Alternative Fuels** - TSR optimization strategies
        • **Safety Protocols** - Emergency procedures & best practices
        • **Troubleshooting** - Root cause analysis & solutions
        • **Regulations** - Environmental compliance & reporting
        """)

if __name__ == "__main__":
    launch_plant_ai_assistant()
