import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CementChemistry:
    """
    Core cement chemistry calculations including Bogue equations and LSF calculations
    """
    
    def __init__(self):
        # Molecular weights for common cement compounds
        self.molecular_weights = {
            'CaO': 56.08, 'SiO2': 60.08, 'Al2O3': 101.96, 'Fe2O3': 159.69,
            'MgO': 40.30, 'SO3': 80.06, 'K2O': 94.20, 'Na2O': 61.98,
            'TiO2': 79.87, 'P2O5': 141.94, 'Mn2O3': 157.87, 'Cr2O3': 151.99
        }
        
        # Bogue compound molecular weights
        self.bogue_mw = {
            'C3S': 228.32, 'C2S': 172.24, 'C3A': 270.20, 'C4AF': 485.96
        }
    
    def calculate_lsf(self, cao: float, sio2: float, al2o3: float, fe2o3: float) -> float:
        """
        Calculate Lime Saturation Factor (LSF)
        LSF = CaO / (2.8 * SiO2 + 1.2 * Al2O3 + 0.65 * Fe2O3)
        """
        denominator = 2.8 * sio2 + 1.2 * al2o3 + 0.65 * fe2o3
        if denominator == 0:
            return 0.0
        return cao / denominator
    
    def calculate_silica_modulus(self, sio2: float, al2o3: float, fe2o3: float) -> float:
        """
        Calculate Silica Modulus (SM)
        SM = SiO2 / (Al2O3 + Fe2O3)
        """
        denominator = al2o3 + fe2o3
        if denominator == 0:
            return 0.0
        return sio2 / denominator
    
    def calculate_alumina_modulus(self, al2o3: float, fe2o3: float) -> float:
        """
        Calculate Alumina Modulus (AM)
        AM = Al2O3 / Fe2O3
        """
        if fe2o3 == 0:
            return 0.0
        return al2o3 / fe2o3
    
    def calculate_bogue_compounds(self, cao: float, sio2: float, al2o3: float, fe2o3: float) -> Dict[str, float]:
        """
        Calculate Bogue compounds using standard equations
        """
        # Bogue equations (simplified version)
        c4af = 3.043 * fe2o3
        c3a = 2.650 * al2o3 - 1.692 * fe2o3
        c2s = 2.867 * sio2 - 0.7544 * cao
        c3s = 4.071 * cao - 7.600 * sio2 - 6.718 * al2o3 - 1.430 * fe2o3
        
        # Ensure non-negative values
        bogue = {
            'C3S': max(0, c3s),
            'C2S': max(0, c2s),
            'C3A': max(0, c3a),
            'C4AF': max(0, c4af)
        }
        
        return bogue
    
    def validate_chemistry(self, composition: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate cement chemistry for realistic constraints
        """
        validation = {}
        
        # Check major oxide ranges (typical Portland cement)
        validation['cao_range'] = 60.0 <= composition.get('CaO', 0) <= 67.0
        validation['sio2_range'] = 18.0 <= composition.get('SiO2', 0) <= 25.0
        validation['al2o3_range'] = 3.0 <= composition.get('Al2O3', 0) <= 8.0
        validation['fe2o3_range'] = 1.5 <= composition.get('Fe2O3', 0) <= 5.0
        
        # Calculate moduli
        cao, sio2, al2o3, fe2o3 = [composition.get(ox, 0) for ox in ['CaO', 'SiO2', 'Al2O3', 'Fe2O3']]
        
        lsf = self.calculate_lsf(cao, sio2, al2o3, fe2o3)
        sm = self.calculate_silica_modulus(sio2, al2o3, fe2o3)
        am = self.calculate_alumina_modulus(al2o3, fe2o3)
        
        validation['lsf_range'] = 0.85 <= lsf <= 1.02
        validation['sm_range'] = 2.0 <= sm <= 3.5
        validation['am_range'] = 1.0 <= am <= 4.0
        
        # Total should be close to 100%
        total_major = sum([composition.get(ox, 0) for ox in ['CaO', 'SiO2', 'Al2O3', 'Fe2O3']])
        validation['major_total'] = 85.0 <= total_major <= 95.0
        
        return validation

print("CementChemistry class created successfully!")
print("Features: LSF, moduli calculations, Bogue compounds, chemistry validation")