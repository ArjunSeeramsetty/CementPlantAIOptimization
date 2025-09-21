# FILE: src/cement_ai_platform/copilot/enhanced_ai_assistant.py
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import json
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CementPlantCopilot:
    """Enhanced AI Copilot with cement domain expertise"""

    def __init__(self, plant_config, current_kpis: Dict):
        self.plant_config = plant_config
        self.current_kpis = current_kpis
        self.knowledge_base = self._initialize_knowledge_base()

        # Initialize Gemini
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.gemini_available = True
        else:
            self.gemini_available = False

    def _initialize_knowledge_base(self) -> Dict:
        """Initialize cement plant knowledge base"""

        return {
            'process_expertise': {
                'kiln_operations': {
                    'optimal_temperature_range': [1450, 1470],
                    'temperature_control_factors': [
                        'fuel rate adjustment',
                        'air/fuel ratio optimization',
                        'feed rate modification',
                        'kiln speed adjustment'
                    ],
                    'common_issues': {
                        'high_free_lime': [
                            'Increase kiln temperature by 5-10°C',
                            'Reduce kiln speed to increase residence time',
                            'Check raw meal fineness',
                            'Optimize fuel/air ratio'
                        ],
                        'ring_formation': [
                            'Adjust flame shape',
                            'Modify raw meal chemistry',
                            'Check coating formation',
                            'Review alternative fuel usage'
                        ]
                    }
                },
                'quality_control': {
                    'free_lime_factors': [
                        'kiln temperature',
                        'residence time',
                        'LSF level',
                        'raw meal fineness'
                    ],
                    'strength_optimization': [
                        'C3S content optimization',
                        'Cement fineness control',
                        'Gypsum content adjustment',
                        'Grinding optimization'
                    ]
                },
                'energy_optimization': {
                    'thermal_energy_reduction': [
                        'Optimize excess air',
                        'Improve heat recovery',
                        'Enhance preheater efficiency',
                        'Alternative fuel substitution'
                    ],
                    'electrical_energy_saving': [
                        'Grinding optimization',
                        'Fan efficiency improvement',
                        'Motor upgrades',
                        'Process optimization'
                    ]
                },
                'environmental_compliance': {
                    'nox_reduction': [
                        'Staged combustion',
                        'Low NOx burners',
                        'SNCR/SCR systems',
                        'Alternative fuel usage'
                    ],
                    'dust_control': [
                        'Baghouse optimization',
                        'ESP enhancement',
                        'Process modifications',
                        'Material handling improvements'
                    ]
                }
            },
            'fuel_optimization': {
                'alternative_fuels': {
                    'rdf': {
                        'benefits': ['Cost reduction', 'CO2 reduction', 'Waste utilization'],
                        'challenges': ['Quality variation', 'Handling complexity', 'Ash content'],
                        'optimal_usage': '15-25% TSR'
                    },
                    'biomass': {
                        'benefits': ['Carbon neutral', 'Renewable', 'Local availability'],
                        'challenges': ['Seasonal availability', 'Moisture content', 'Storage'],
                        'optimal_usage': '5-15% TSR'
                    }
                }
            }
        }

    def generate_contextual_response(self, query: str) -> Dict[str, Any]:
        """Generate context-aware response with cement expertise"""

        # Analyze query intent
        intent = self._analyze_query_intent(query)

        if self.gemini_available:
            # Build context-rich prompt
            context_prompt = self._build_expert_prompt(query, intent)

            try:
                # Generate response using Gemini
                response = self.model.generate_content(context_prompt)

                # Enhance response with plant-specific recommendations
                enhanced_response = self._enhance_with_plant_context(response.text, intent)

                return {
                    'answer': enhanced_response['answer'],
                    'confidence': enhanced_response['confidence'],
                    'plant_specific_actions': enhanced_response['actions'],
                    'related_kpis': enhanced_response['related_kpis'],
                    'recommendations': enhanced_response['recommendations']
                }

            except Exception as e:
                return self._generate_fallback_response(query, intent)
        else:
            return self._generate_fallback_response(query, intent)

    def _analyze_query_intent(self, query: str) -> str:
        """Analyze query intent for context-aware responses"""

        query_lower = query.lower()

        if any(word in query_lower for word in ['temperature', 'kiln', 'burning', 'flame']):
            return 'kiln_operations'
        elif any(word in query_lower for word in ['free lime', 'quality', 'strength', 'cement']):
            return 'quality_control'
        elif any(word in query_lower for word in ['energy', 'fuel', 'consumption', 'cost']):
            return 'energy_optimization'
        elif any(word in query_lower for word in ['nox', 'emissions', 'dust', 'environmental']):
            return 'environmental'
        elif any(word in query_lower for word in ['alternative fuel', 'tsr', 'rdf', 'biomass']):
            return 'fuel_optimization'
        elif any(word in query_lower for word in ['maintenance', 'breakdown', 'repair']):
            return 'maintenance'
        else:
            return 'general'

    def _build_expert_prompt(self, query: str, intent: str) -> str:
        """Build expert-level prompt with plant context"""

        system_prompt = f"""
You are an expert cement plant process engineer with 20+ years of experience in cement manufacturing optimization.
You have deep knowledge of:
- Kiln operations and pyrometrics
- Quality control and cement chemistry
- Energy optimization and alternative fuels
- Environmental compliance and emissions control
- Predictive maintenance and equipment reliability

Current Plant Context:
- Plant: {self.plant_config.plant_name}
- Location: {self.plant_config.location}
- Capacity: {self.plant_config.capacity_tpd:,.0f} TPD
- Technology: {self.plant_config.technology_level}
- Kiln Type: {self.plant_config.kiln_type}

Current Operating Conditions:
- Kiln Temperature: {self.current_kpis.get('kiln_temperature_c', 'N/A')}°C
- Free Lime: {self.current_kpis.get('free_lime_pct', 'N/A')}%
- Thermal Energy: {self.current_kpis.get('thermal_energy_kcal_kg', 'N/A')} kcal/kg
- NOx Emissions: {self.current_kpis.get('nox_emissions_mg_nm3', 'N/A')} mg/Nm³
- OEE: {self.current_kpis.get('oee_percentage', 'N/A')}%

Query Intent: {intent}

Provide expert technical advice that is:
1. Specific to this plant's configuration and current conditions
2. Actionable with clear implementation steps
3. Quantified with expected impacts where possible
4. Considers operational constraints and safety
5. References industry best practices

User Query: {query}
"""

        return system_prompt

    def _enhance_with_plant_context(self, base_response: str, intent: str) -> Dict[str, Any]:
        """Enhance response with plant-specific context and actions"""

        # Extract plant-specific actions based on intent
        actions = self._get_plant_specific_actions(intent)

        # Identify related KPIs
        related_kpis = self._get_related_kpis(intent)

        # Generate specific recommendations
        recommendations = self._generate_specific_recommendations(intent)

        # Assess response confidence based on available data
        confidence = self._assess_response_confidence(intent)

        return {
            'answer': base_response,
            'confidence': confidence,
            'actions': actions,
            'related_kpis': related_kpis,
            'recommendations': recommendations
        }

    def _get_plant_specific_actions(self, intent: str) -> List[str]:
        """Get plant-specific actionable recommendations"""

        actions = []

        if intent == 'kiln_operations':
            current_temp = self.current_kpis.get('kiln_temperature_c', 1450)
            optimal_temp = self.plant_config.process['kiln_temperature_c']

            if current_temp < optimal_temp - 10:
                actions.append(f"Increase kiln temperature by {optimal_temp - current_temp:.0f}°C to reach optimal {optimal_temp}°C")
            elif current_temp > optimal_temp + 10:
                actions.append(f"Reduce kiln temperature by {current_temp - optimal_temp:.0f}°C to reach optimal {optimal_temp}°C")

            actions.append("Monitor oxygen levels and adjust air/fuel ratio accordingly")
            actions.append("Check preheater tower temperatures for optimal heat transfer")

        elif intent == 'quality_control':
            current_free_lime = self.current_kpis.get('free_lime_pct', 1.2)
            target_free_lime = self.plant_config.quality['free_lime_pct']

            if current_free_lime > target_free_lime + 0.3:
                actions.append(f"Reduce free lime from {current_free_lime:.1f}% to target {target_free_lime:.1f}%")
                actions.append("Increase burning zone temperature or reduce kiln speed")
                actions.append("Check raw meal fineness and LSF levels")

        elif intent == 'energy_optimization':
            current_thermal = self.current_kpis.get('thermal_energy_kcal_kg', 3200)
            target_thermal = self.plant_config.energy['thermal']

            if current_thermal > target_thermal + 50:
                potential_savings = ((current_thermal - target_thermal) / current_thermal) * 100
                actions.append(f"Target {potential_savings:.1f}% thermal energy reduction")
                actions.append("Optimize excess air levels and heat recovery systems")
                actions.append("Consider increasing alternative fuel usage")

        elif intent == 'fuel_optimization':
            current_tsr = self._calculate_current_tsr()
            if current_tsr < 20:
                actions.append(f"Increase TSR from current {current_tsr:.1f}% to target 25%")
                actions.append("Evaluate RDF and biomass availability in your region")
                actions.append("Conduct fuel quality tests and heating value analysis")

        return actions

    def _get_related_kpis(self, intent: str) -> List[str]:
        """Get KPIs related to the query intent"""

        kpi_mapping = {
            'kiln_operations': ['kiln_temperature_c', 'free_lime_pct', 'o2_percentage'],
            'quality_control': ['free_lime_pct', 'cement_strength_28d', 'quality_consistency_score'],
            'energy_optimization': ['thermal_energy_kcal_kg', 'electrical_energy_kwh_t', 'energy_efficiency_pct'],
            'environmental': ['nox_emissions_mg_nm3', 'co2_emissions_kg_t', 'dust_emissions_mg_nm3'],
            'fuel_optimization': ['tsr_percentage', 'fuel_cost_per_ton', 'thermal_energy_kcal_kg']
        }

        return kpi_mapping.get(intent, ['oee_percentage', 'production_rate_tph'])

    def _generate_specific_recommendations(self, intent: str) -> List[Dict[str, str]]:
        """Generate specific recommendations with implementation details"""

        recommendations = []

        if intent == 'energy_optimization':
            recommendations.append({
                'action': 'Implement heat recovery optimization',
                'impact': '3-5% thermal energy reduction',
                'timeline': '2-3 months',
                'investment': '₹50-80 lakhs'
            })

        elif intent == 'fuel_optimization':
            recommendations.append({
                'action': 'Increase alternative fuel to 25% TSR',
                'impact': '₹15-25 lakhs monthly fuel cost savings',
                'timeline': '6-9 months',
                'investment': '₹2-3 crores for infrastructure'
            })

        elif intent == 'quality_control':
            recommendations.append({
                'action': 'Implement advanced process control',
                'impact': '±0.5% free lime consistency improvement',
                'timeline': '4-6 months',
                'investment': '₹1-2 crores'
            })

        return recommendations

    def _assess_response_confidence(self, intent: str) -> float:
        """Assess confidence level based on available data and intent"""

        # Base confidence on data availability and plant specificity
        data_availability = len([k for k in self.current_kpis.keys() if self.current_kpis[k] is not None])
        max_kpis = 15

        data_confidence = min(0.9, data_availability / max_kpis)

        # Intent-specific confidence adjustments
        intent_confidence = {
            'kiln_operations': 0.95,
            'quality_control': 0.92,
            'energy_optimization': 0.88,
            'fuel_optimization': 0.85,
            'environmental': 0.90,
            'maintenance': 0.75,
            'general': 0.70
        }.get(intent, 0.75)

        overall_confidence = (data_confidence + intent_confidence) / 2

        return round(overall_confidence, 2)

    def _calculate_current_tsr(self) -> float:
        """Calculate current TSR percentage"""
        alt_fuels = ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        alt_fuel_total = sum(self.plant_config.fuel_mix.get(fuel, 0) for fuel in alt_fuels)
        total_fuel = sum(self.plant_config.fuel_mix.values())

        return (alt_fuel_total / total_fuel) * 100 if total_fuel > 0 else 0

    def _generate_fallback_response(self, query: str, intent: str) -> Dict[str, Any]:
        """Generate fallback response if AI call fails"""

        fallback_responses = {
            'kiln_operations': {
                'answer': "For kiln temperature optimization, monitor your current burning zone temperature and adjust fuel rate accordingly. Typical optimal range is 1450-1470°C.",
                'actions': ["Check current temperature against target", "Adjust fuel/air ratio", "Monitor oxygen levels"],
                'confidence': 0.75
            },
            'quality_control': {
                'answer': "Quality optimization involves maintaining free lime within target range (0.8-1.8%) and ensuring consistent cement strength through proper burning zone control.",
                'actions': ["Monitor free lime levels", "Check kiln temperature", "Verify raw meal chemistry"],
                'confidence': 0.72
            },
            'energy_optimization': {
                'answer': "Energy optimization typically involves optimizing excess air, improving heat recovery, and considering alternative fuels. Target 3-5% reduction through systematic improvements.",
                'actions': ["Conduct energy audit", "Optimize air/fuel ratio", "Evaluate heat recovery"],
                'confidence': 0.70
            },
            'fuel_optimization': {
                'answer': "Alternative fuel optimization can reduce costs and emissions. Evaluate RDF, biomass, and other alternative fuels based on local availability and quality.",
                'actions': ["Assess alternative fuel availability", "Conduct fuel quality analysis", "Plan TSR increase strategy"],
                'confidence': 0.68
            },
            'environmental': {
                'answer': "Environmental compliance requires monitoring NOx, dust, and other emissions. Consider staged combustion and advanced control systems for emission reduction.",
                'actions': ["Monitor emission levels", "Check control systems", "Evaluate upgrade options"],
                'confidence': 0.73
            }
        }

        fallback = fallback_responses.get(intent, {
            'answer': "I understand your query. Please check current process parameters and consider consulting with your process engineer for specific recommendations.",
            'actions': ["Review current KPIs", "Consult process team"],
            'confidence': 0.60
        })

        return {
            'answer': fallback['answer'],
            'confidence': fallback['confidence'],
            'plant_specific_actions': fallback['actions'],
            'related_kpis': self._get_related_kpis(intent),
            'recommendations': []
        }

    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of available knowledge base for the UI"""
        
        return {
            'process_expertise_areas': list(self.knowledge_base['process_expertise'].keys()),
            'fuel_optimization_areas': list(self.knowledge_base['fuel_optimization'].keys()),
            'plant_context': {
                'plant_name': self.plant_config.plant_name,
                'technology_level': self.plant_config.technology_level,
                'capacity_tpd': self.plant_config.capacity_tpd
            },
            'current_kpis_available': len([k for k in self.current_kpis.keys() if self.current_kpis[k] is not None]),
            'gemini_available': self.gemini_available
        }
