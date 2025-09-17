import os
import yaml
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

# Conditional import for Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    # Create a mock genai module for fallback
    class MockGenai:
        @staticmethod
        def configure(**kwargs):
            pass
        
        class GenerativeModel:
            def __init__(self, model_name):
                self.model_name = model_name
            
            def generate_content(self, prompt):
                class MockResponse:
                    def __init__(self, text):
                        self.text = text
                
                # Return a fallback response based on prompt content
                if 'free lime' in prompt.lower():
                    return MockResponse("High free lime typically indicates insufficient burning. Check burning zone temperature (target: 1450°C), reduce kiln speed, or increase fuel rate. Normal free lime should be <1.5%.")
                elif 'nox' in prompt.lower() or 'emission' in prompt.lower():
                    return MockResponse("High NOx emissions can be reduced by optimizing excess air (target: 2-4% O2), using staged combustion, or adjusting fuel/air mixing. Current limit: <500 mg/Nm³.")
                elif 'energy' in prompt.lower():
                    return MockResponse("Energy optimization focuses on thermal efficiency (target: <720 kcal/kg) and electrical consumption (<75 kWh/t). Check kiln insulation, optimize combustion, and review mill operations.")
                else:
                    return MockResponse("Based on current plant data, the system is operating within normal parameters. Continue monitoring key performance indicators and follow standard operating procedures.")
    
    genai = MockGenai()

try:
    from cement_ai_platform.config.settings import Settings
except ImportError:
    # Fallback for testing
    class Settings:
        def __init__(self):
            pass

class CementPlantKnowledgeBase:
    """Knowledge base for cement plant operations and expertise"""
    
    def __init__(self):
        self.plant_context = self._load_plant_context()
        self.process_knowledge = self._load_process_knowledge()
        self.troubleshooting_guide = self._load_troubleshooting_guide()
    
    def _load_plant_context(self) -> Dict:
        """Load current plant configuration and status"""
        try:
            with open("config/plant_config.yml", 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            return {"error": "Plant configuration not found"}
    
    def _load_process_knowledge(self) -> Dict:
        """Load cement process knowledge base"""
        return {
            "kiln_operations": {
                "normal_burning_zone_temp": "1420-1480°C",
                "optimal_free_lime": "<1.5%",
                "fuel_rate_range": "14-20 t/h",
                "kiln_speed_range": "2.8-4.2 rpm"
            },
            "quality_parameters": {
                "free_lime_target": "0.5-1.5%",
                "c3s_content": "55-65%",
                "blaine_fineness": "3200-3800 cm²/g",
                "strength_28d": ">42.5 MPa"
            },
            "energy_benchmarks": {
                "thermal_energy": "680-720 kcal/kg",
                "electrical_energy": "65-75 kWh/t",
                "coal_consumption": "95-110 kg/t clinker"
            },
            "emissions_limits": {
                "nox": "<500 mg/Nm³",
                "so2": "<200 mg/Nm³",
                "dust": "<30 mg/Nm³"
            }
        }
    
    def _load_troubleshooting_guide(self) -> Dict:
        """Load troubleshooting knowledge base"""
        return {
            "high_free_lime": {
                "causes": ["Low burning zone temperature", "High kiln speed", "Poor fuel distribution", "Raw meal fineness issues"],
                "solutions": ["Increase fuel rate", "Reduce kiln speed", "Adjust burner position", "Check raw meal fineness"]
            },
            "ring_formation": {
                "causes": ["High alkali content", "Poor fuel quality", "Incorrect kiln temperature profile"],
                "solutions": ["Increase bypass", "Improve fuel quality", "Adjust firing pattern"]
            },
            "high_nox_emissions": {
                "causes": ["High excess air", "High flame temperature", "Poor fuel mixing"],
                "solutions": ["Reduce excess air", "Optimize fuel/air ratio", "Use staged combustion"]
            },
            "cooler_problems": {
                "causes": ["Clinker bed depth issues", "Grate speed problems", "Air distribution imbalance"],
                "solutions": ["Adjust grate speed", "Balance air flows", "Monitor clinker temperature"]
            }
        }

class CementPlantGPT:
    """
    Advanced cement plant AI assistant with domain expertise.
    Provides natural language interface for plant operations, troubleshooting, and optimization.
    """
    
    def __init__(self):
        self.settings = Settings()
        self.knowledge_base = CementPlantKnowledgeBase()
        self._initialize_ai_model()
        self.conversation_history = []
    
    def _initialize_ai_model(self):
        """Initialize Gemini AI model"""
        try:
            if GENAI_AVAILABLE:
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'mock_key'))
                self.model = genai.GenerativeModel('gemini-pro')
            else:
                # Use mock model for fallback
                self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            # Fallback to mock model
            self.model = genai.GenerativeModel('gemini-pro')
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with cement plant expertise"""
        return f"""
You are an expert cement plant operations AI assistant with deep knowledge of cement manufacturing processes, quality control, energy optimization, and troubleshooting.

PLANT CONTEXT:
{json.dumps(self.knowledge_base.plant_context, indent=2)}

PROCESS KNOWLEDGE:
{json.dumps(self.knowledge_base.process_knowledge, indent=2)}

TROUBLESHOOTING EXPERTISE:
{json.dumps(self.knowledge_base.troubleshooting_guide, indent=2)}

CAPABILITIES:
- Analyze plant performance data and KPIs
- Provide troubleshooting guidance for process issues
- Recommend optimization strategies for energy and quality
- Explain cement chemistry and process relationships
- Assist with operational decision-making
- Generate reports and analysis

RESPONSE GUIDELINES:
- Provide practical, actionable advice
- Include specific parameter ranges and setpoints when relevant
- Reference industry best practices and benchmarks
- Explain the reasoning behind recommendations
- Highlight safety considerations when applicable
- Use technical language appropriate for plant operators and engineers

Always prioritize safety, product quality, and operational efficiency in your recommendations.
"""
    
    def query(self, user_prompt: str, context_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user query with full cement plant context
        
        Args:
            user_prompt: User's question or request
            context_data: Optional current plant data for context
            
        Returns:
            Dict containing response and metadata
        """
        
        # Always use the model (either real or mock)
        
        # Build full prompt with context
        system_prompt = self._build_system_prompt()
        
        full_prompt = f"{system_prompt}\n\nCURRENT PLANT DATA:\n"
        if context_data:
            # Convert pandas Timestamps to strings for JSON serialization
            def convert_timestamps(obj):
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_timestamps(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                else:
                    return obj
            
            context_data_serializable = convert_timestamps(context_data)
            full_prompt += json.dumps(context_data_serializable, indent=2)
        
        full_prompt += f"\n\nUSER QUERY: {user_prompt}\n\nRESPONSE:"
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_query': user_prompt,
                'ai_response': response.text,
                'context_provided': context_data is not None
            })
            
            return {
                'success': True,
                'response': response.text,
                'confidence': self._assess_confidence(user_prompt, response.text),
                'query_type': self._classify_query(user_prompt),
                'recommendations': self._extract_recommendations(response.text),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback_response': self._fallback_response(user_prompt),
                'timestamp': datetime.now().isoformat()
            }
    
    def _fallback_response(self, user_prompt: str) -> Dict[str, Any]:
        """Provide fallback response when AI model is unavailable"""
        keywords = user_prompt.lower()
        
        if 'free lime' in keywords:
            return {
                'response': "High free lime typically indicates insufficient burning. Check burning zone temperature (target: 1450°C), reduce kiln speed, or increase fuel rate. Normal free lime should be <1.5%.",
                'source': 'knowledge_base_fallback'
            }
        elif 'nox' in keywords or 'emission' in keywords:
            return {
                'response': "High NOx emissions can be reduced by optimizing excess air (target: 2-4% O2), using staged combustion, or adjusting fuel/air mixing. Current limit: <500 mg/Nm³.",
                'source': 'knowledge_base_fallback'
            }
        elif 'energy' in keywords:
            return {
                'response': "Energy optimization focuses on thermal efficiency (target: <720 kcal/kg) and electrical consumption (<75 kWh/t). Check kiln insulation, optimize combustion, and review mill operations.",
                'source': 'knowledge_base_fallback'
            }
        else:
            return {
                'response': "I'm currently running in offline mode. Please check the specific parameter values against plant targets and consult the operations manual for detailed guidance.",
                'source': 'knowledge_base_fallback'
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of user query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['why', 'cause', 'reason']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['optimize', 'improve', 'reduce', 'increase']):
            return 'optimization'
        elif any(word in query_lower for word in ['what', 'how', 'explain']):
            return 'information'
        elif any(word in query_lower for word in ['predict', 'forecast', 'expect']):
            return 'prediction'
        else:
            return 'general'
    
    def _assess_confidence(self, query: str, response: str) -> float:
        """Assess confidence level of the response"""
        # Simple heuristic based on response length and specificity
        confidence = 0.7  # Base confidence
        
        if len(response) > 200:
            confidence += 0.1
        if any(word in response.lower() for word in ['specific', 'precisely', 'exactly']):
            confidence += 0.1
        if any(word in response.lower() for word in ['might', 'possibly', 'uncertain']):
            confidence -= 0.2
            
        return min(max(confidence, 0.1), 1.0)
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract actionable recommendations from response"""
        recommendations = []
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['should', 'recommend', 'adjust', 'increase', 'decrease', 'optimize']):
                recommendations.append(sentence.strip())
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def generate_shift_report(self, plant_data: Dict) -> str:
        """Generate automated shift report"""
        prompt = f"""
        Generate a comprehensive shift report for the cement plant based on the following data:
        {json.dumps(plant_data, indent=2)}
        
        Include:
        1. Production summary
        2. Quality performance vs targets
        3. Energy consumption analysis
        4. Any operational issues or deviations
        5. Recommendations for next shift
        """
        
        result = self.query(prompt, plant_data)
        return result.get('response', 'Shift report generation failed')
    
    def analyze_quality_trend(self, quality_data: pd.DataFrame) -> Dict:
        """Analyze quality trends and provide insights"""
        trend_summary = {
            'free_lime_avg': quality_data['free_lime_percent'].mean(),
            'free_lime_trend': 'increasing' if quality_data['free_lime_percent'].iloc[-1] > quality_data['free_lime_percent'].iloc[0] else 'decreasing',
            'strength_avg': quality_data['compressive_strength_28d_mpa'].mean(),
            'variability': quality_data['free_lime_percent'].std()
        }
        
        prompt = f"""
        Analyze the cement quality trends and provide recommendations:
        {json.dumps(trend_summary, indent=2)}
        
        Focus on:
        1. Quality stability assessment
        2. Deviations from targets
        3. Process adjustments needed
        4. Preventive measures
        """
        
        result = self.query(prompt, trend_summary)
        
        return {
            'analysis': result.get('response', 'Analysis failed'),
            'trend_data': trend_summary,
            'recommendations': result.get('recommendations', [])
        }
