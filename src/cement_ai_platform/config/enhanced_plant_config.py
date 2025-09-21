# FILE: src/cement_ai_platform/config/enhanced_plant_config.py
import yaml
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

@dataclass
class PlantConfiguration:
    """Enhanced plant configuration with physics-based models"""
    plant_id: str
    plant_name: str
    location: str
    capacity_tpd: float
    kiln_type: str

    # Raw materials composition
    raw_materials: Dict[str, float]

    # Fuel mix and properties
    fuel_mix: Dict[str, float]
    fuel_properties: Dict[str, Dict[str, float]]

    # Energy baselines
    energy: Dict[str, float]

    # Process parameters
    process: Dict[str, Any]

    # Quality targets
    quality: Dict[str, float]

    # Environmental limits
    environmental: Dict[str, float]

    # DCS configuration
    dcs_tags: Dict[str, Any]

    # Plant metadata
    tenant_id: str
    region: str
    timezone: str
    commissioning_year: int
    technology_level: str  # "basic", "advanced", "state_of_art"

    def get_baseline_energy_consumption(self) -> Dict[str, float]:
        """Calculate baseline energy consumption"""
        return {
            'thermal_energy_kcal_kg': self.energy.get('thermal', 3200),
            'electrical_energy_kwh_t': self.energy.get('electrical', 95),
            'specific_power_kwh_t': self.energy.get('specific_power', 110)
        }

    def get_optimal_process_window(self) -> Dict[str, Dict[str, float]]:
        """Get optimal operating windows for key process variables"""
        base_temp = self.process.get('kiln_temperature_c', 1450)

        return {
            'kiln_temperature_c': {
                'optimal': base_temp,
                'min': base_temp - 20,
                'max': base_temp + 15,
                'control_tolerance': 5
            },
            'free_lime_pct': {
                'optimal': self.quality.get('free_lime_pct', 1.2),
                'min': 0.8,
                'max': 1.8,
                'control_tolerance': 0.2
            },
            'o2_percentage': {
                'optimal': 3.2,
                'min': 2.8,
                'max': 4.0,
                'control_tolerance': 0.3
            }
        }

class EnhancedPlantConfigManager:
    """Enhanced plant configuration manager with physics models"""

    def __init__(self):
        self.plants: Dict[str, PlantConfiguration] = {}
        self.load_all_plant_configurations()

    def load_all_plant_configurations(self):
        """Load all plant configurations with enhanced data"""

        # JK Cement plants based on real facilities
        plant_configs = [
            {
                'plant_id': 'jk_cement_nimbahera',
                'plant_name': 'JK Cement Nimbahera',
                'location': 'Rajasthan, India',
                'capacity_tpd': 12000,
                'kiln_type': 'Dry Process with 5-stage Preheater',
                'raw_materials': {
                    'limestone': 1250,
                    'clay': 180,
                    'iron_ore': 45,
                    'gypsum': 55,
                    'flyash': 20
                },
                'fuel_mix': {
                    'coal': 110,
                    'petcoke': 65,
                    'alternative_fuels': 35,  # RDF, biomass
                    'total_fuel_rate_tph': 15.8
                },
                'fuel_properties': {
                    'coal': {'cv_kcal_kg': 6200, 'cost_per_ton': 4800, 'carbon_factor': 94.6, 'ash_pct': 12},
                    'petcoke': {'cv_kcal_kg': 8100, 'cost_per_ton': 3900, 'carbon_factor': 102.1, 'ash_pct': 0.8},
                    'rdf': {'cv_kcal_kg': 4200, 'cost_per_ton': 1800, 'carbon_factor': 91.7, 'ash_pct': 18},
                    'biomass': {'cv_kcal_kg': 4500, 'cost_per_ton': 2400, 'carbon_factor': 0, 'ash_pct': 8}
                },
                'energy': {
                    'thermal': 3150,
                    'electrical': 92,
                    'specific_power': 108
                },
                'process': {
                    'kiln_temperature_c': 1455,
                    'kiln_speed_rpm': 3.6,
                    'preheater_stages': 5,
                    'cooler_type': 'Grate Cooler',
                    'burner_type': 'Multi-channel'
                },
                'quality': {
                    'free_lime_pct': 1.1,
                    'c3s_content_pct': 62,
                    'c2s_content_pct': 14,
                    'compressive_strength_28d_mpa': 48,
                    'blaine_cm2_g': 3450
                },
                'environmental': {
                    'nox_mg_nm3': 480,
                    'so2_mg_nm3': 180,
                    'dust_mg_nm3': 25,
                    'co2_kg_per_ton': 780
                },
                'dcs_tags': {
                    'update_frequencies': {
                        'critical_loops': 1,
                        'process_variables': 5,
                        'quality_lab': 3600
                    }
                },
                'tenant_id': 'jk_cement_group',
                'region': 'north_india',
                'timezone': 'Asia/Kolkata',
                'commissioning_year': 2008,
                'technology_level': 'advanced'
            },
            {
                'plant_id': 'jk_cement_mangrol',
                'plant_name': 'JK Cement Mangrol',
                'location': 'Rajasthan, India',
                'capacity_tpd': 8000,
                'kiln_type': 'Dry Process with 4-stage Preheater',
                'raw_materials': {
                    'limestone': 1280,
                    'clay': 190,
                    'iron_ore': 50,
                    'gypsum': 50,
                    'flyash': 15
                },
                'fuel_mix': {
                    'coal': 125,
                    'petcoke': 55,
                    'alternative_fuels': 25,
                    'total_fuel_rate_tph': 13.2
                },
                'fuel_properties': {
                    'coal': {'cv_kcal_kg': 5900, 'cost_per_ton': 5100, 'carbon_factor': 94.6, 'ash_pct': 14},
                    'petcoke': {'cv_kcal_kg': 7800, 'cost_per_ton': 4200, 'carbon_factor': 102.1, 'ash_pct': 1.2},
                    'rdf': {'cv_kcal_kg': 3900, 'cost_per_ton': 2000, 'carbon_factor': 91.7, 'ash_pct': 20},
                    'biomass': {'cv_kcal_kg': 4200, 'cost_per_ton': 2600, 'carbon_factor': 0, 'ash_pct': 10}
                },
                'energy': {
                    'thermal': 3280,
                    'electrical': 98,
                    'specific_power': 115
                },
                'process': {
                    'kiln_temperature_c': 1445,
                    'kiln_speed_rpm': 3.4,
                    'preheater_stages': 4,
                    'cooler_type': 'Grate Cooler',
                    'burner_type': 'Single-channel'
                },
                'quality': {
                    'free_lime_pct': 1.3,
                    'c2s_content_pct': 16,
                    'compressive_strength_28d_mpa': 46,
                    'blaine_cm2_g': 3380
                },
                'environmental': {
                    'nox_mg_nm3': 520,
                    'so2_mg_nm3': 200,
                    'dust_mg_nm3': 28,
                    'co2_kg_per_ton': 820
                },
                'dcs_tags': {
                    'update_frequencies': {
                        'critical_loops': 2,
                        'process_variables': 5,
                        'quality_lab': 3600
                    }
                },
                'tenant_id': 'jk_cement_group',
                'region': 'north_india',
                'timezone': 'Asia/Kolkata',
                'commissioning_year': 2003,
                'technology_level': 'basic'
            },
            {
                'plant_id': 'jk_cement_muddapur',
                'plant_name': 'JK Cement Muddapur',
                'location': 'Karnataka, India',
                'capacity_tpd': 6000,
                'kiln_type': 'Dry Process with 5-stage Preheater + Calciner',
                'raw_materials': {
                    'limestone': 1220,
                    'laterite': 200,
                    'iron_ore': 35,
                    'gypsum': 60,
                    'flyash': 25
                },
                'fuel_mix': {
                    'coal': 100,
                    'petcoke': 70,
                    'alternative_fuels': 45,  # Higher alt fuel usage
                    'total_fuel_rate_tph': 11.8
                },
                'fuel_properties': {
                    'coal': {'cv_kcal_kg': 6400, 'cost_per_ton': 4600, 'carbon_factor': 94.6, 'ash_pct': 10},
                    'petcoke': {'cv_kcal_kg': 8300, 'cost_per_ton': 3700, 'carbon_factor': 102.1, 'ash_pct': 0.6},
                    'rdf': {'cv_kcal_kg': 4400, 'cost_per_ton': 1600, 'carbon_factor': 91.7, 'ash_pct': 16},
                    'biomass': {'cv_kcal_kg': 4800, 'cost_per_ton': 2200, 'carbon_factor': 0, 'ash_pct': 6}
                },
                'energy': {
                    'thermal': 3080,  # Best performance due to modern tech
                    'electrical': 88,
                    'specific_power': 102
                },
                'process': {
                    'kiln_temperature_c': 1460,
                    'kiln_speed_rpm': 3.7,
                    'preheater_stages': 5,
                    'cooler_type': 'Advanced Grate Cooler',
                    'burner_type': 'Multi-channel with AFR'
                },
                'quality': {
                    'free_lime_pct': 0.9,  # Best quality control
                    'c3s_content_pct': 64,
                    'c2s_content_pct': 13,
                    'compressive_strength_28d_mpa': 52,
                    'blaine_cm2_g': 3520
                },
                'environmental': {
                    'nox_mg_nm3': 420,  # Best environmental performance
                    'so2_mg_nm3': 150,
                    'dust_mg_nm3': 20,
                    'co2_kg_per_ton': 720
                },
                'dcs_tags': {
                    'update_frequencies': {
                        'critical_loops': 1,
                        'process_variables': 3,
                        'quality_lab': 1800
                    }
                },
                'tenant_id': 'jk_cement_group',
                'region': 'south_india',
                'timezone': 'Asia/Kolkata',
                'commissioning_year': 2015,
                'technology_level': 'state_of_art'
            }
        ]

        # Load configurations
        for config_data in plant_configs:
            plant_config = PlantConfiguration(**config_data)
            self.plants[plant_config.plant_id] = plant_config

    def get_plant_config(self, plant_id: str) -> Optional[PlantConfiguration]:
        """Get specific plant configuration"""
        return self.plants.get(plant_id)

    def get_all_plants(self) -> List[PlantConfiguration]:
        """Get all plant configurations"""
        return list(self.plants.values())

    def get_plants_by_tenant(self, tenant_id: str) -> List[PlantConfiguration]:
        """Get plants for specific tenant"""
        return [plant for plant in self.plants.values() if plant.tenant_id == tenant_id]

    def export_configurations(self, format_type: str = 'json') -> str:
        """Export configurations in specified format"""
        if format_type == 'json':
            return json.dumps({
                plant_id: asdict(plant)
                for plant_id, plant in self.plants.items()
            }, indent=2)
        elif format_type == 'yaml':
            return yaml.dump({
                plant_id: asdict(plant)
                for plant_id, plant in self.plants.items()
            }, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def get_plant_comparison_matrix(self) -> pd.DataFrame:
        """Generate comparison matrix for all plants"""
        comparison_data = []

        for plant in self.plants.values():
            comparison_data.append({
                'plant_id': plant.plant_id,
                'capacity_tpd': plant.capacity_tpd,
                'technology_level': plant.technology_level,
                'thermal_energy': plant.energy['thermal'],
                'electrical_energy': plant.energy['electrical'],
                'commissioning_year': plant.commissioning_year,
                'nox_emissions': plant.environmental['nox_mg_nm3'],
                'free_lime_pct': plant.quality['free_lime_pct']
            })

        return pd.DataFrame(comparison_data)
