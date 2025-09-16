"""
Cement Plant GPT - Natural Language Interface for Digital Twin
Implements JK Cement's requirement for intuitive plant interaction
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import json

logger = logging.getLogger(__name__)

class PlantKnowledgeBase:
    """Knowledge base containing plant configuration and operational data"""
    
    def __init__(self, config_path: str = "config/plant_config.yml"):
        self.config_path = config_path
        self.plant_config = self._load_plant_config()
        self.operational_history = []
        self.kpi_snapshots = []
        
    def _load_plant_config(self) -> Dict[str, Any]:
        """Load plant configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load plant config: {e}")
            return {}
    
    def get_plant_summary(self) -> str:
        """Get comprehensive plant summary for GPT context"""
        plant_info = self.plant_config.get('plant', {})
        
        summary = f"""
        Plant: {plant_info.get('name', 'Unknown')}
        Capacity: {plant_info.get('capacity_tpd', 'Unknown')} TPD
        Kiln Type: {plant_info.get('kiln_type', 'Unknown')}
        Commissioning Year: {plant_info.get('commissioning_year', 'Unknown')}
        
        Key Process Parameters:
        - Raw Materials: Limestone {plant_info.get('raw_materials', {}).get('limestone_pct', 'N/A')}%, Clay {plant_info.get('raw_materials', {}).get('clay_pct', 'N/A')}%
        - Fuel Mix: Coal {plant_info.get('fuel_mix', {}).get('coal_pct', 'N/A')}%, Petcoke {plant_info.get('fuel_mix', {}).get('petcoke_pct', 'N/A')}%
        - Energy Consumption: Electrical {plant_info.get('energy', {}).get('electrical_kwh_t', 'N/A')} kWh/t, Thermal {plant_info.get('energy', {}).get('thermal_kcal_kg', 'N/A')} kcal/kg
        
        Quality Targets:
        - C3S Content: {plant_info.get('quality_targets', {}).get('c3s_content_pct', 'N/A')}%
        - Free Lime: {plant_info.get('quality_targets', {}).get('free_lime_pct', 'N/A')}%
        - Compressive Strength: {plant_info.get('quality_targets', {}).get('compressive_strength_mpa', 'N/A')} MPa
        """
        
        return summary.strip()
    
    def get_recent_kpis(self, hours: int = 24) -> str:
        """Get recent KPI data for context"""
        if not self.kpi_snapshots:
            return "No recent KPI data available"
        
        recent_data = self.kpi_snapshots[-10:]  # Last 10 snapshots
        
        kpi_summary = "Recent KPI Performance:\n"
        for snapshot in recent_data:
            timestamp = snapshot.get('timestamp', 'Unknown')
            kpi_summary += f"- {timestamp}: Production {snapshot.get('production_tph', 'N/A')} tph, "
            kpi_summary += f"Free Lime {snapshot.get('free_lime_pct', 'N/A')}%, "
            kpi_summary += f"Power {snapshot.get('specific_power_kwh_t', 'N/A')} kWh/t\n"
        
        return kpi_summary
    
    def add_kpi_snapshot(self, kpi_data: Dict[str, Any]):
        """Add new KPI snapshot to history"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            **kpi_data
        }
        self.kpi_snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.kpi_snapshots) > 100:
            self.kpi_snapshots = self.kpi_snapshots[-100:]

class CementPlantGPT:
    """
    Natural language interface for the cement plant digital twin.
    Provides intelligent responses to operational queries and recommendations.
    """

    def __init__(self, config_path: str = "config/plant_config.yml"):
        self.knowledge_base = PlantKnowledgeBase(config_path)
        self.conversation_history = []
        
        # Initialize response templates
        self.response_templates = self._initialize_response_templates()
        
        logger.info("✅ Cement Plant GPT initialized")

    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize response templates for common queries"""
        return {
            "plant_status": """
            Based on current plant data:
            - Production Rate: {production_rate} tph
            - Free Lime: {free_lime}% (Target: <2.0%)
            - Specific Power: {specific_power} kWh/t
            - Kiln Temperature: {kiln_temp}°C
            - Overall Status: {status}
            
            Recommendations: {recommendations}
            """,
            
            "quality_analysis": """
            Quality Analysis Results:
            - C3S Content: {c3s_content}% (Target: 55-65%)
            - Free Lime: {free_lime}% (Target: <2.0%)
            - Compressive Strength: {strength} MPa (Target: >40 MPa)
            
            Quality Status: {quality_status}
            Issues Identified: {issues}
            Recommended Actions: {actions}
            """,
            
            "energy_optimization": """
            Energy Optimization Analysis:
            - Current Specific Power: {current_power} kWh/t
            - Target Specific Power: {target_power} kWh/t
            - Potential Savings: {savings}%
            
            Optimization Opportunities:
            {opportunities}
            
            Recommended Actions: {actions}
            """,
            
            "fuel_optimization": """
            Fuel Optimization Recommendations:
            - Current TSR: {current_tsr}%
            - Target TSR: {target_tsr}%
            - Alternative Fuel Potential: {alt_fuel_potential}
            
            Fuel Blend Recommendations:
            {fuel_recommendations}
            
            Expected Benefits: {benefits}
            """
        }

    def query(self, user_prompt: str, context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Process natural language query and provide intelligent response
        
        Args:
            user_prompt: User's question or request
            context_data: Additional context data (KPIs, sensor readings, etc.)
            
        Returns:
            Intelligent response with recommendations
        """
        logger.info(f"🤖 Processing GPT query: {user_prompt[:50]}...")
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_prompt,
            'context_data': context_data
        })
        
        # Analyze query intent
        intent = self._analyze_query_intent(user_prompt)
        
        # Generate response based on intent
        response = self._generate_response(intent, user_prompt, context_data)
        
        # Add response to history
        self.conversation_history[-1]['response'] = response
        
        return response

    def _analyze_query_intent(self, query: str) -> str:
        """Analyze user query to determine intent"""
        query_lower = query.lower()
        
        # Intent keywords
        intents = {
            "plant_status": ["status", "how is", "what's happening", "current state", "performance"],
            "quality_analysis": ["quality", "free lime", "strength", "c3s", "clinker quality"],
            "energy_optimization": ["energy", "power", "consumption", "efficiency", "optimize energy"],
            "fuel_optimization": ["fuel", "tsr", "alternative fuel", "rdf", "biomass", "fuel blend"],
            "maintenance": ["maintenance", "repair", "breakdown", "equipment", "mill", "kiln"],
            "production": ["production", "output", "throughput", "capacity", "rate"],
            "environmental": ["emissions", "co2", "nox", "so2", "dust", "environmental"],
            "troubleshooting": ["problem", "issue", "error", "alarm", "trouble", "fix", "solve"]
        }
        
        # Find matching intent
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "general_query"

    def _generate_response(self, intent: str, query: str, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate response based on intent and context"""
        
        if intent == "plant_status":
            return self._generate_plant_status_response(context_data)
        elif intent == "quality_analysis":
            return self._generate_quality_analysis_response(context_data)
        elif intent == "energy_optimization":
            return self._generate_energy_optimization_response(context_data)
        elif intent == "fuel_optimization":
            return self._generate_fuel_optimization_response(context_data)
        elif intent == "maintenance":
            return self._generate_maintenance_response(query, context_data)
        elif intent == "production":
            return self._generate_production_response(context_data)
        elif intent == "environmental":
            return self._generate_environmental_response(context_data)
        elif intent == "troubleshooting":
            return self._generate_troubleshooting_response(query, context_data)
        else:
            return self._generate_general_response(query, context_data)

    def _generate_plant_status_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate plant status response"""
        if not context_data:
            return "No current plant data available. Please provide sensor readings or KPI data."
        
        # Extract key metrics
        production_rate = context_data.get('production_tph', 'N/A')
        free_lime = context_data.get('free_lime_pct', 'N/A')
        specific_power = context_data.get('specific_power_kwh_t', 'N/A')
        kiln_temp = context_data.get('burning_zone_temp_c', 'N/A')
        
        # Determine overall status
        status = "Normal"
        recommendations = []
        
        if isinstance(free_lime, (int, float)) and free_lime > 2.0:
            status = "Attention Required"
            recommendations.append("Free lime is high - check kiln temperature and residence time")
        
        if isinstance(specific_power, (int, float)) and specific_power > 120:
            status = "Optimization Opportunity"
            recommendations.append("Specific power consumption is high - review grinding efficiency")
        
        if isinstance(kiln_temp, (int, float)) and kiln_temp > 1500:
            status = "High Temperature Alert"
            recommendations.append("Kiln temperature is high - monitor refractory condition")
        
        rec_text = "; ".join(recommendations) if recommendations else "Continue monitoring"
        
        return self.response_templates["plant_status"].format(
            production_rate=production_rate,
            free_lime=free_lime,
            specific_power=specific_power,
            kiln_temp=kiln_temp,
            status=status,
            recommendations=rec_text
        )

    def _generate_quality_analysis_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate quality analysis response"""
        if not context_data:
            return "No quality data available. Please provide clinker analysis results."
        
        c3s_content = context_data.get('c3s_content_pct', 'N/A')
        free_lime = context_data.get('free_lime_pct', 'N/A')
        strength = context_data.get('compressive_strength_28d_mpa', 'N/A')
        
        # Analyze quality status
        quality_status = "Good"
        issues = []
        actions = []
        
        if isinstance(free_lime, (int, float)) and free_lime > 2.0:
            quality_status = "Poor"
            issues.append("High free lime content")
            actions.append("Increase kiln temperature or residence time")
        
        if isinstance(c3s_content, (int, float)) and c3s_content < 55:
            quality_status = "Poor"
            issues.append("Low C3S content")
            actions.append("Optimize raw meal composition and burning conditions")
        
        if isinstance(strength, (int, float)) and strength < 40:
            quality_status = "Poor"
            issues.append("Low compressive strength")
            actions.append("Review clinker composition and grinding parameters")
        
        issues_text = "; ".join(issues) if issues else "No major issues detected"
        actions_text = "; ".join(actions) if actions else "Continue current operations"
        
        return self.response_templates["quality_analysis"].format(
            c3s_content=c3s_content,
            free_lime=free_lime,
            strength=strength,
            quality_status=quality_status,
            issues=issues_text,
            actions=actions_text
        )

    def _generate_energy_optimization_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate energy optimization response"""
        if not context_data:
            return "No energy data available. Please provide power consumption data."
        
        current_power = context_data.get('specific_power_kwh_t', 'N/A')
        target_power = 100  # Target from JK Cement requirements
        
        opportunities = []
        actions = []
        
        if isinstance(current_power, (int, float)):
            savings_potential = ((current_power - target_power) / current_power) * 100
            
            if savings_potential > 0:
                opportunities.append(f"Raw grinding optimization: Potential 3-5% savings")
                opportunities.append(f"Cement grinding optimization: Potential 2-3% savings")
                opportunities.append(f"Fan optimization: Potential 1-2% savings")
                
                actions.append("Review raw mill separator efficiency")
                actions.append("Optimize cement mill grinding media")
                actions.append("Check ID fan VFD settings")
        else:
            savings_potential = 0
        
        opportunities_text = "\n".join(f"- {opp}" for opp in opportunities)
        actions_text = "; ".join(actions) if actions else "Monitor current consumption"
        
        return self.response_templates["energy_optimization"].format(
            current_power=current_power,
            target_power=target_power,
            savings=savings_potential,
            opportunities=opportunities_text,
            actions=actions_text
        )

    def _generate_fuel_optimization_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate fuel optimization response"""
        if not context_data:
            return "No fuel data available. Please provide current fuel consumption data."
        
        current_tsr = context_data.get('tsr_pct', 0)
        target_tsr = 15  # JK Cement target
        
        fuel_recommendations = []
        benefits = []
        
        if current_tsr < target_tsr:
            fuel_recommendations.append("Increase RDF usage to 8-10%")
            fuel_recommendations.append("Add biomass fuel to 3-5%")
            fuel_recommendations.append("Optimize tire-derived fuel blend")
            
            benefits.append("Reduce fossil fuel costs")
            benefits.append("Improve environmental footprint")
            benefits.append("Achieve TSR target of 15%")
        
        fuel_rec_text = "\n".join(f"- {rec}" for rec in fuel_recommendations)
        benefits_text = "; ".join(benefits) if benefits else "Current fuel mix is optimal"
        
        return self.response_templates["fuel_optimization"].format(
            current_tsr=current_tsr,
            target_tsr=target_tsr,
            alt_fuel_potential="High",
            fuel_recommendations=fuel_rec_text,
            benefits=benefits_text
        )

    def _generate_maintenance_response(self, query: str, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate maintenance-related response"""
        query_lower = query.lower()
        
        if "mill" in query_lower:
            return """
            Mill Maintenance Recommendations:
            - Check grinding media wear and replace if needed
            - Inspect separator efficiency
            - Monitor vibration levels
            - Review lubrication system
            - Check for material buildup
            
            Next scheduled maintenance: Based on operating hours
            """
        elif "kiln" in query_lower:
            return """
            Kiln Maintenance Recommendations:
            - Inspect refractory lining condition
            - Check kiln shell temperature profile
            - Monitor kiln alignment
            - Review drive system condition
            - Check seal condition
            
            Critical: Monitor refractory thickness regularly
            """
        else:
            return "Please specify which equipment needs maintenance attention (mill, kiln, fan, etc.)"

    def _generate_production_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate production-related response"""
        if not context_data:
            return "No production data available."
        
        production_rate = context_data.get('production_tph', 'N/A')
        capacity = context_data.get('capacity_tpd', 'N/A')
        
        if isinstance(production_rate, (int, float)) and isinstance(capacity, (int, float)):
            utilization = (production_rate / capacity) * 100
            return f"""
            Production Status:
            - Current Rate: {production_rate} tph
            - Plant Capacity: {capacity} TPD
            - Utilization: {utilization:.1f}%
            
            Recommendations: {"Increase production" if utilization < 80 else "Maintain current rate"}
            """
        
        return f"Current production rate: {production_rate} tph"

    def _generate_environmental_response(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate environmental compliance response"""
        if not context_data:
            return "No environmental data available."
        
        co2 = context_data.get('co2_kg_t', 'N/A')
        nox = context_data.get('nox_mg_nm3', 'N/A')
        so2 = context_data.get('so2_mg_nm3', 'N/A')
        dust = context_data.get('dust_mg_nm3', 'N/A')
        
        return f"""
        Environmental Compliance Status:
        - CO2 Emissions: {co2} kg/t clinker
        - NOx Emissions: {nox} mg/Nm³
        - SO2 Emissions: {so2} mg/Nm³
        - Dust Emissions: {dust} mg/Nm³
        
        Status: {"Within limits" if isinstance(nox, (int, float)) and nox < 800 else "Monitor closely"}
        """

    def _generate_troubleshooting_response(self, query: str, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate troubleshooting response"""
        query_lower = query.lower()
        
        if "high" in query_lower and "temperature" in query_lower:
            return """
            High Temperature Troubleshooting:
            1. Check fuel flow rate and quality
            2. Verify raw meal feed rate
            3. Inspect kiln refractory condition
            4. Check kiln speed and residence time
            5. Review cooling air flow
            
            Immediate Action: Reduce fuel rate by 5-10%
            """
        elif "low" in query_lower and "production" in query_lower:
            return """
            Low Production Troubleshooting:
            1. Check raw material availability and quality
            2. Verify mill performance and separator efficiency
            3. Inspect kiln feed system
            4. Check for material buildup
            5. Review process parameters
            
            Immediate Action: Check raw mill throughput
            """
        else:
            return "Please describe the specific problem you're experiencing for detailed troubleshooting guidance."

    def _generate_general_response(self, query: str, context_data: Optional[Dict[str, Any]]) -> str:
        """Generate general response for unclear queries"""
        plant_summary = self.knowledge_base.get_plant_summary()
        
        return f"""
        I understand you're asking about: "{query}"
        
        {plant_summary}
        
        For more specific assistance, please ask about:
        - Plant status and performance
        - Quality analysis and optimization
        - Energy consumption and efficiency
        - Fuel optimization and TSR
        - Maintenance recommendations
        - Production optimization
        - Environmental compliance
        - Troubleshooting specific issues
        
        How can I help you optimize your cement plant operations?
        """

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")

    def export_knowledge(self) -> Dict[str, Any]:
        """Export plant knowledge for external use"""
        return {
            'plant_config': self.knowledge_base.plant_config,
            'recent_kpis': self.knowledge_base.kpi_snapshots[-10:],
            'conversation_count': len(self.conversation_history),
            'last_updated': datetime.now().isoformat()
        }
