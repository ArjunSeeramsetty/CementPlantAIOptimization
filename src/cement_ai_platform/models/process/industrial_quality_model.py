"""
Industrial Quality Model with Compressive Strength Prediction
Implements comprehensive cement quality modeling based on industrial correlations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class IndustrialQualityPredictor:
    """
    Industrial-grade cement quality predictor with comprehensive strength modeling.
    
    Features:
    - Multi-age compressive strength prediction (1, 3, 7, 28 days)
    - Clinker phase composition analysis
    - Gypsum optimization for setting time control
    - Particle size distribution effects
    - Quality optimization recommendations
    """
    
    def __init__(self):
        # Strength contribution factors (empirical, based on industrial data)
        self.phase_strength_factors = {
            'C3S': 1.2,    # Alite - primary strength contributor
            'C2S': 0.5,    # Belite - secondary strength contributor
            'C3A': 0.3,    # Tricalcium aluminate - early strength
            'C4AF': 0.2    # Ferrite - minimal strength contribution
        }
        
        # Maturity factors for different ages (based on Powers' model)
        self.maturity_factors = {
            1: 0.25,    # 1 day
            3: 0.45,    # 3 days
            7: 0.65,    # 7 days
            28: 1.0,    # 28 days (reference)
            56: 1.15,   # 56 days
            90: 1.25    # 90 days
        }
        
        # Fineness effect factors
        self.fineness_reference = 3500  # Blaine cm²/g reference
        self.fineness_exponent = 0.3    # Fineness effect exponent
        
        # Gypsum optimization parameters
        self.gypsum_optimization = {
            'base_so3': 0.7,           # Base SO3 content (%)
            'c3a_factor': 0.15,        # C3A effect factor
            'fineness_factor': 0.0001, # Fineness effect factor
            'optimal_range': (0.5, 1.5) # Optimal SO3 range (%)
        }
        
        print("🏗️ Industrial Quality Predictor initialized")
        print("📊 Multi-age strength prediction and gypsum optimization available")
    
    def predict_compressive_strength(self,
                                   clinker_composition: Dict[str, float],
                                   fineness_blaine: float,
                                   age_days: int,
                                   gypsum_content: Optional[float] = None,
                                   w_c_ratio: float = 0.4) -> Dict[str, Any]:
        """
        Predict compressive strength based on clinker phases and fineness.
        
        Uses simplified Powers' model approach with industrial correlations.
        
        Args:
            clinker_composition: Dict with clinker phase percentages
                {'C3S': 60, 'C2S': 20, 'C3A': 8, 'C4AF': 10}
            fineness_blaine: Blaine fineness (cm²/g)
            age_days: Concrete age in days
            gypsum_content: Gypsum content as SO3 (%)
            w_c_ratio: Water-cement ratio
            
        Returns:
            Dict with strength prediction and contributing factors
        """
        # Extract clinker phases
        c3s = clinker_composition.get('C3S', 60.0)
        c2s = clinker_composition.get('C2S', 20.0)
        c3a = clinker_composition.get('C3A', 8.0)
        c4af = clinker_composition.get('C4AF', 10.0)
        
        # Calculate cement factor based on strength-giving phases
        cement_factor = (self.phase_strength_factors['C3S'] * c3s +
                        self.phase_strength_factors['C2S'] * c2s +
                        self.phase_strength_factors['C3A'] * c3a +
                        self.phase_strength_factors['C4AF'] * c4af)
        
        # Fineness effect (finer cement = higher early strength)
        fineness_factor = (fineness_blaine / self.fineness_reference) ** self.fineness_exponent
        
        # Maturity factor based on age
        maturity_factor = self.maturity_factors.get(age_days, 1.0)
        if age_days not in self.maturity_factors:
            # Interpolate for intermediate ages
            maturity_factor = self._interpolate_maturity_factor(age_days)
        
        # Gypsum effect (optimal gypsum improves strength)
        gypsum_factor = 1.0
        if gypsum_content is not None:
            gypsum_factor = self._calculate_gypsum_effect(gypsum_content, c3a, fineness_blaine)
        
        # Water-cement ratio effect
        w_c_factor = self._calculate_wc_ratio_effect(w_c_ratio)
        
        # Calculate predicted strength
        base_strength = cement_factor * fineness_factor * maturity_factor * gypsum_factor * w_c_factor
        
        # Apply scaling factor (empirical calibration)
        strength_mpa = base_strength * 0.5
        
        # Ensure reasonable bounds
        strength_mpa = max(10, min(80, strength_mpa))
        
        return {
            'predicted_strength_mpa': strength_mpa,
            'cement_factor': cement_factor,
            'fineness_factor': fineness_factor,
            'maturity_factor': maturity_factor,
            'gypsum_factor': gypsum_factor,
            'w_c_factor': w_c_factor,
            'clinker_phases': {
                'C3S': c3s,
                'C2S': c2s,
                'C3A': c3a,
                'C4AF': c4af
            },
            'quality_grade': self._determine_quality_grade(strength_mpa, age_days)
        }
    
    def optimize_gypsum_content(self,
                               c3a_content: float,
                               fineness_blaine: float,
                               target_setting_time_min: float = 120,
                               clinker_composition: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate optimal gypsum content for setting time control.
        
        Based on simplified Lerch formula approach.
        
        Args:
            c3a_content: C3A content in clinker (%)
            fineness_blaine: Blaine fineness (cm²/g)
            target_setting_time_min: Target setting time (minutes)
            clinker_composition: Full clinker composition (optional)
            
        Returns:
            Dict with gypsum optimization results
        """
        # Base SO3 requirement
        base_so3 = self.gypsum_optimization['base_so3']
        
        # C3A effect (more C3A requires more gypsum)
        c3a_effect = self.gypsum_optimization['c3a_factor'] * c3a_content
        
        # Fineness effect (finer cement requires more gypsum)
        fineness_effect = self.gypsum_optimization['fineness_factor'] * (fineness_blaine - self.fineness_reference)
        
        # Calculate optimal SO3 content
        optimal_so3 = base_so3 + c3a_effect + fineness_effect
        
        # Adjust for target setting time
        setting_time_factor = self._calculate_setting_time_factor(target_setting_time_min)
        optimal_so3 *= setting_time_factor
        
        # Ensure within optimal range
        optimal_so3 = max(self.gypsum_optimization['optimal_range'][0],
                         min(self.gypsum_optimization['optimal_range'][1], optimal_so3))
        
        # Calculate gypsum content (assuming 80% SO3 in gypsum)
        gypsum_content = optimal_so3 / 0.8
        
        # Predict setting time with optimal gypsum
        predicted_setting_time = self._predict_setting_time(optimal_so3, c3a_content, fineness_blaine)
        
        return {
            'optimal_so3_percent': optimal_so3,
            'optimal_gypsum_percent': gypsum_content,
            'predicted_setting_time_min': predicted_setting_time,
            'target_setting_time_min': target_setting_time_min,
            'setting_time_deviation': predicted_setting_time - target_setting_time_min,
            'optimization_factors': {
                'base_so3': base_so3,
                'c3a_effect': c3a_effect,
                'fineness_effect': fineness_effect,
                'setting_time_factor': setting_time_factor
            }
        }
    
    def analyze_clinker_quality(self,
                              clinker_composition: Dict[str, float],
                              free_lime: float,
                              kiln_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive clinker quality analysis.
        
        Args:
            clinker_composition: Clinker phase composition
            free_lime: Free lime content (%)
            kiln_conditions: Kiln operating conditions
            
        Returns:
            Dict with comprehensive quality analysis
        """
        # Calculate quality indicators
        c3s = clinker_composition.get('C3S', 60.0)
        c2s = clinker_composition.get('C2S', 20.0)
        c3a = clinker_composition.get('C3A', 8.0)
        c4af = clinker_composition.get('C4AF', 10.0)
        
        # LSF (Lime Saturation Factor)
        cao = c3s * 0.737 + c2s * 0.651 + c3a * 0.623 + c4af * 0.461
        sio2 = c3s * 0.263 + c2s * 0.349
        lsf = cao / (2.8 * sio2) if sio2 > 0 else 0
        
        # SM (Silica Modulus)
        al2o3 = c3a * 0.377 + c4af * 0.210
        fe2o3 = c4af * 0.329
        sm = sio2 / (al2o3 + fe2o3) if (al2o3 + fe2o3) > 0 else 0
        
        # AM (Alumina Modulus)
        am = al2o3 / fe2o3 if fe2o3 > 0 else 0
        
        # Quality assessment
        quality_score = self._calculate_quality_score(c3s, c2s, free_lime, lsf, sm, am)
        
        # Burnability assessment
        burnability = self._assess_burnability(free_lime, kiln_conditions)
        
        # Strength potential
        strength_potential = self._calculate_strength_potential(c3s, c2s, c3a)
        
        return {
            'clinker_phases': clinker_composition,
            'quality_indicators': {
                'lsf': lsf,
                'sm': sm,
                'am': am,
                'free_lime': free_lime
            },
            'quality_score': quality_score,
            'burnability_assessment': burnability,
            'strength_potential': strength_potential,
            'quality_grade': self._determine_clinker_grade(quality_score),
            'recommendations': self._generate_quality_recommendations(quality_score, free_lime, lsf, sm)
        }
    
    def predict_cement_properties(self,
                                clinker_composition: Dict[str, float],
                                fineness_blaine: float,
                                gypsum_content: float,
                                age_days: int = 28) -> Dict[str, Any]:
        """
        Predict comprehensive cement properties.
        
        Args:
            clinker_composition: Clinker phase composition
            fineness_blaine: Blaine fineness (cm²/g)
            gypsum_content: Gypsum content (%)
            age_days: Concrete age (days)
            
        Returns:
            Dict with comprehensive cement properties
        """
        # Strength prediction
        strength_result = self.predict_compressive_strength(
            clinker_composition=clinker_composition,
            fineness_blaine=fineness_blaine,
            age_days=age_days,
            gypsum_content=gypsum_content
        )
        
        # Setting time prediction
        c3a = clinker_composition.get('C3A', 8.0)
        setting_time = self._predict_setting_time(gypsum_content * 0.8, c3a, fineness_blaine)
        
        # Workability prediction
        workability = self._predict_workability(fineness_blaine, gypsum_content)
        
        # Durability indicators
        durability = self._assess_durability(clinker_composition, fineness_blaine)
        
        return {
            'strength_properties': strength_result,
            'setting_time_min': setting_time,
            'workability_index': workability,
            'durability_assessment': durability,
            'cement_classification': self._classify_cement_type(clinker_composition),
            'quality_grade': strength_result['quality_grade']
        }
    
    # Helper methods
    def _interpolate_maturity_factor(self, age_days: int) -> float:
        """Interpolate maturity factor for intermediate ages."""
        ages = sorted(self.maturity_factors.keys())
        
        if age_days <= ages[0]:
            return self.maturity_factors[ages[0]]
        if age_days >= ages[-1]:
            return self.maturity_factors[ages[-1]]
        
        # Find surrounding ages
        for i in range(len(ages) - 1):
            if ages[i] <= age_days <= ages[i + 1]:
                # Linear interpolation
                t1, f1 = ages[i], self.maturity_factors[ages[i]]
                t2, f2 = ages[i + 1], self.maturity_factors[ages[i + 1]]
                return f1 + (f2 - f1) * (age_days - t1) / (t2 - t1)
        
        return 1.0
    
    def _calculate_gypsum_effect(self, gypsum_content: float, c3a: float, fineness: float) -> float:
        """Calculate gypsum effect on strength."""
        # Optimal gypsum range
        optimal_gypsum = 0.7 + 0.15 * c3a + (fineness - self.fineness_reference) / 10000
        
        # Gypsum effect (bell curve around optimal)
        deviation = abs(gypsum_content - optimal_gypsum)
        effect = 1.0 - (deviation / optimal_gypsum) * 0.1  # 10% reduction per 100% deviation
        
        return max(0.8, min(1.2, effect))
    
    def _calculate_wc_ratio_effect(self, w_c_ratio: float) -> float:
        """Calculate water-cement ratio effect on strength."""
        # Optimal w/c ratio is around 0.4
        optimal_wc = 0.4
        deviation = abs(w_c_ratio - optimal_wc)
        
        # Strength decreases with higher w/c ratio
        effect = 1.0 - deviation * 0.5  # 50% reduction per 0.1 increase in w/c
        
        return max(0.5, min(1.0, effect))
    
    def _calculate_setting_time_factor(self, target_setting_time: float) -> float:
        """Calculate factor for target setting time adjustment."""
        # Longer setting time requires more gypsum
        if target_setting_time > 120:
            return 1.0 + (target_setting_time - 120) / 1000  # 1% increase per 10 min
        else:
            return 1.0 - (120 - target_setting_time) / 1000  # 1% decrease per 10 min
    
    def _predict_setting_time(self, so3_content: float, c3a: float, fineness: float) -> float:
        """Predict setting time based on SO3, C3A, and fineness."""
        # Base setting time
        base_time = 120  # minutes
        
        # SO3 effect (more SO3 = longer setting time)
        so3_effect = (so3_content - 0.7) * 50  # 50 min per 0.1% SO3 deviation
        
        # C3A effect (more C3A = shorter setting time)
        c3a_effect = -(c3a - 8) * 5  # 5 min per 1% C3A deviation
        
        # Fineness effect (finer = shorter setting time)
        fineness_effect = -(fineness - self.fineness_reference) / 100  # 1 min per 100 Blaine
        
        setting_time = base_time + so3_effect + c3a_effect + fineness_effect
        
        return max(60, min(300, setting_time))  # Reasonable bounds
    
    def _determine_quality_grade(self, strength_mpa: float, age_days: int) -> str:
        """Determine cement quality grade based on strength."""
        if age_days == 28:
            if strength_mpa >= 52.5:
                return "CEM I 52.5N"
            elif strength_mpa >= 42.5:
                return "CEM I 42.5N"
            elif strength_mpa >= 32.5:
                return "CEM I 32.5N"
            else:
                return "Below Standard"
        else:
            # For other ages, use relative assessment
            if strength_mpa >= 40:
                return "High Quality"
            elif strength_mpa >= 25:
                return "Good Quality"
            else:
                return "Standard Quality"
    
    def _calculate_quality_score(self, c3s: float, c2s: float, free_lime: float, 
                                lsf: float, sm: float, am: float) -> float:
        """Calculate overall quality score."""
        score = 0.0
        
        # C3S contribution (higher is better)
        score += min(1.0, c3s / 70) * 0.3
        
        # C2S contribution
        score += min(1.0, c2s / 25) * 0.2
        
        # Free lime (lower is better)
        score += max(0, 1.0 - free_lime / 2.0) * 0.2
        
        # LSF (optimal around 0.95)
        lsf_score = 1.0 - abs(lsf - 0.95) / 0.1
        score += max(0, lsf_score) * 0.15
        
        # SM (optimal around 2.5)
        sm_score = 1.0 - abs(sm - 2.5) / 0.5
        score += max(0, sm_score) * 0.1
        
        # AM (optimal around 1.5)
        am_score = 1.0 - abs(am - 1.5) / 0.5
        score += max(0, am_score) * 0.05
        
        return min(1.0, score)
    
    def _assess_burnability(self, free_lime: float, kiln_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Assess clinker burnability."""
        temp = kiln_conditions.get('temperature', 1450)
        
        # Burnability assessment
        if free_lime < 1.0 and temp < 1450:
            burnability = "Excellent"
        elif free_lime < 1.5 and temp < 1500:
            burnability = "Good"
        elif free_lime < 2.0:
            burnability = "Acceptable"
        else:
            burnability = "Poor"
        
        return {
            'burnability': burnability,
            'free_lime': free_lime,
            'kiln_temperature': temp,
            'recommendations': self._generate_burnability_recommendations(burnability, free_lime)
        }
    
    def _calculate_strength_potential(self, c3s: float, c2s: float, c3a: float) -> Dict[str, float]:
        """Calculate strength potential at different ages."""
        phases = {'C3S': c3s, 'C2S': c2s, 'C3A': c3a}
        
        strength_potential = {}
        for age in [1, 3, 7, 28]:
            strength = self.predict_compressive_strength(phases, 3500, age)['predicted_strength_mpa']
            strength_potential[f'{age}_day'] = strength
        
        return strength_potential
    
    def _determine_clinker_grade(self, quality_score: float) -> str:
        """Determine clinker quality grade."""
        if quality_score >= 0.9:
            return "Premium"
        elif quality_score >= 0.8:
            return "High"
        elif quality_score >= 0.7:
            return "Good"
        elif quality_score >= 0.6:
            return "Standard"
        else:
            return "Below Standard"
    
    def _generate_quality_recommendations(self, quality_score: float, free_lime: float, 
                                        lsf: float, sm: float) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Overall quality needs improvement")
        
        if free_lime > 1.5:
            recommendations.append("Reduce free lime by increasing kiln temperature or residence time")
        
        if lsf < 0.9:
            recommendations.append("Increase lime saturation factor by adjusting raw mix")
        elif lsf > 1.0:
            recommendations.append("Decrease lime saturation factor by reducing limestone")
        
        if sm < 2.0:
            recommendations.append("Increase silica modulus by adding sand")
        elif sm > 3.0:
            recommendations.append("Decrease silica modulus by reducing sand")
        
        if not recommendations:
            recommendations.append("Quality parameters are within acceptable ranges")
        
        return recommendations
    
    def _generate_burnability_recommendations(self, burnability: str, free_lime: float) -> List[str]:
        """Generate burnability improvement recommendations."""
        recommendations = []
        
        if burnability == "Poor":
            recommendations.append("Increase kiln temperature significantly")
            recommendations.append("Increase residence time")
            recommendations.append("Improve raw meal fineness")
        elif burnability == "Acceptable":
            recommendations.append("Slight increase in kiln temperature recommended")
            recommendations.append("Monitor free lime trends")
        else:
            recommendations.append("Burnability is satisfactory")
        
        return recommendations
    
    def _predict_workability(self, fineness: float, gypsum_content: float) -> float:
        """Predict cement workability."""
        # Fineness effect (finer = better workability)
        fineness_factor = fineness / self.fineness_reference
        
        # Gypsum effect (optimal gypsum improves workability)
        optimal_gypsum = 0.7
        gypsum_factor = 1.0 - abs(gypsum_content - optimal_gypsum) / optimal_gypsum * 0.2
        
        workability = fineness_factor * gypsum_factor * 100  # Scale to 0-100
        
        return max(50, min(100, workability))
    
    def _assess_durability(self, clinker_composition: Dict[str, float], fineness: float) -> Dict[str, Any]:
        """Assess cement durability characteristics."""
        c3a = clinker_composition.get('C3A', 8.0)
        c4af = clinker_composition.get('C4AF', 10.0)
        
        # Sulfate resistance (lower C3A = better)
        sulfate_resistance = "Good" if c3a < 8 else "Moderate" if c3a < 12 else "Poor"
        
        # Heat of hydration (higher C3A = more heat)
        heat_of_hydration = "Low" if c3a < 6 else "Moderate" if c3a < 10 else "High"
        
        # Fineness effect on durability
        fineness_durability = "Good" if fineness > 4000 else "Moderate" if fineness > 3000 else "Poor"
        
        return {
            'sulfate_resistance': sulfate_resistance,
            'heat_of_hydration': heat_of_hydration,
            'fineness_durability': fineness_durability,
            'overall_durability': self._calculate_overall_durability(sulfate_resistance, heat_of_hydration, fineness_durability)
        }
    
    def _calculate_overall_durability(self, sulfate_resistance: str, heat_of_hydration: str, fineness_durability: str) -> str:
        """Calculate overall durability rating."""
        scores = {
            'Good': 3, 'Moderate': 2, 'Poor': 1
        }
        
        total_score = (scores[sulfate_resistance] + scores[heat_of_hydration] + scores[fineness_durability]) / 3
        
        if total_score >= 2.5:
            return "High"
        elif total_score >= 2.0:
            return "Moderate"
        else:
            return "Low"
    
    def _classify_cement_type(self, clinker_composition: Dict[str, float]) -> str:
        """Classify cement type based on composition."""
        c3s = clinker_composition.get('C3S', 60.0)
        c2s = clinker_composition.get('C2S', 20.0)
        c3a = clinker_composition.get('C3A', 8.0)
        
        if c3s > 65 and c3a < 8:
            return "CEM I (High Early Strength)"
        elif c3s > 55 and c3a < 10:
            return "CEM I (Standard)"
        elif c2s > 25:
            return "CEM I (Low Heat)"
        else:
            return "CEM I (General Purpose)"


def create_industrial_quality_predictor() -> IndustrialQualityPredictor:
    """Factory function to create an industrial quality predictor."""
    return IndustrialQualityPredictor()
