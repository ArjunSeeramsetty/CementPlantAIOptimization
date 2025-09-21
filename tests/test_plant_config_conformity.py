"""Test script to verify conformity between plant_config.yml and DCS simulator implementation."""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantConfigConformityTester:
    """
    Test class to verify conformity between plant_config.yml and DCS simulator implementation.
    """
    
    def __init__(self, config_path: str = 'config/plant_config.yml'):
        """Initialize the tester with plant configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.dcs_simulator = self._load_dcs_simulator()
        
    def _load_config(self) -> dict:
        """Load plant configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded plant configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_dcs_simulator(self):
        """Load DCS simulator to get tag definitions."""
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            
            from simulation.dcs_simulator import CementPlantDCSSimulator
            simulator = CementPlantDCSSimulator(self.config_path)
            return simulator
        except Exception as e:
            logger.error(f"Failed to load DCS simulator: {e}")
            return None
    
    def test_plant_metadata_conformity(self) -> Dict[str, bool]:
        """Test if plant metadata is properly configured."""
        logger.info("=== Testing Plant Metadata Conformity ===")
        
        results = {}
        plant_config = self.config.get('plant', {})
        
        # Test required plant metadata
        required_fields = ['name', 'location', 'capacity_tpd', 'kiln_type']
        for field in required_fields:
            results[f'plant_{field}'] = field in plant_config and plant_config[field] is not None
        
        # Test specific values
        results['capacity_10000_tpd'] = plant_config.get('capacity_tpd') == 10000
        results['kiln_dry_process'] = plant_config.get('kiln_type') == 'Dry Process'
        results['location_india'] = plant_config.get('location') == 'India'
        
        logger.info(f"Plant metadata conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_raw_materials_conformity(self) -> Dict[str, bool]:
        """Test if raw materials are properly configured and used."""
        logger.info("=== Testing Raw Materials Conformity ===")
        
        results = {}
        raw_materials = self.config.get('plant', {}).get('raw_materials', {})
        
        # Test required raw materials
        required_materials = ['limestone', 'clay', 'iron_ore', 'gypsum']
        for material in required_materials:
            results[f'raw_material_{material}'] = material in raw_materials
        
        # Test if raw materials are used in DCS tags
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        results['raw_mill_tags_present'] = any('raw_mill' in tag for tag in dcs_tags)
        
        logger.info(f"Raw materials conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_fuel_mix_conformity(self) -> Dict[str, bool]:
        """Test if fuel mix is properly configured and used."""
        logger.info("=== Testing Fuel Mix Conformity ===")
        
        results = {}
        fuel_mix = self.config.get('plant', {}).get('fuel_mix', {})
        fuel_flows = self.config.get('plant', {}).get('fuel_flows', {})
        
        # Test fuel mix configuration
        required_fuels = ['coal', 'petcoke', 'alternative_fuels']
        for fuel in required_fuels:
            results[f'fuel_mix_{fuel}'] = fuel in fuel_mix
        
        # Test fuel flows configuration
        required_flows = ['coal_flow_tph', 'petcoke_flow_tph', 'alternative_fuel_flow_tph']
        for flow in required_flows:
            results[f'fuel_flow_{flow}'] = flow in fuel_flows
        
        # Test if fuel flows are in DCS tags
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        for flow in required_flows:
            results[f'dcs_tag_{flow}'] = flow in dcs_tags
        
        # Test total fuel flow correlation
        results['total_fuel_flow_tag'] = 'total_fuel_flow_tph' in dcs_tags
        
        logger.info(f"Fuel mix conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_energy_consumption_conformity(self) -> Dict[str, bool]:
        """Test if energy consumption is properly configured."""
        logger.info("=== Testing Energy Consumption Conformity ===")
        
        results = {}
        energy_config = self.config.get('plant', {}).get('energy', {})
        
        # Test energy configuration
        results['electrical_energy'] = 'electrical' in energy_config
        results['thermal_energy'] = 'thermal' in energy_config
        
        # Test specific values
        results['electrical_95_kwh'] = energy_config.get('electrical') == 95
        results['thermal_3200_kcal'] = energy_config.get('thermal') == 3200
        
        logger.info(f"Energy consumption conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_process_parameters_conformity(self) -> Dict[str, bool]:
        """Test if process parameters are properly configured."""
        logger.info("=== Testing Process Parameters Conformity ===")
        
        results = {}
        process_config = self.config.get('plant', {}).get('process', {})
        
        # Test process parameters
        required_params = ['kiln_temperature_c', 'kiln_speed_rpm', 'preheater_stages', 'cooler_type']
        for param in required_params:
            results[f'process_{param}'] = param in process_config
        
        # Test specific values
        results['kiln_temp_1450'] = process_config.get('kiln_temperature_c') == 1450
        results['kiln_speed_3_5'] = process_config.get('kiln_speed_rpm') == 3.5
        results['preheater_5_stages'] = process_config.get('preheater_stages') == 5
        results['cooler_grate_type'] = process_config.get('cooler_type') == 'Grate Cooler'
        
        # Test if preheater stages are used in DCS tags
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        preheater_tags = [tag for tag in dcs_tags if 'preheater_stage' in tag]
        results['preheater_5_stages_tags'] = len(preheater_tags) == 5
        
        logger.info(f"Process parameters conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_quality_targets_conformity(self) -> Dict[str, bool]:
        """Test if quality targets are properly configured."""
        logger.info("=== Testing Quality Targets Conformity ===")
        
        results = {}
        quality_config = self.config.get('plant', {}).get('quality', {})
        
        # Test quality targets
        required_quality = ['free_lime_pct', 'c3s_content_pct', 'c2s_content_pct', 'compressive_strength_28d_mpa']
        for quality in required_quality:
            results[f'quality_{quality}'] = quality in quality_config
        
        # Test if quality targets are in DCS tags
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        for quality in required_quality:
            results[f'dcs_quality_{quality}'] = quality in dcs_tags
        
        logger.info(f"Quality targets conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_environmental_limits_conformity(self) -> Dict[str, bool]:
        """Test if environmental limits are properly configured."""
        logger.info("=== Testing Environmental Limits Conformity ===")
        
        results = {}
        env_config = self.config.get('plant', {}).get('environmental', {})
        
        # Test environmental limits
        required_env = ['nox_mg_nm3', 'so2_mg_nm3', 'dust_mg_nm3', 'co2_kg_per_ton']
        for env in required_env:
            results[f'environmental_{env}'] = env in env_config
        
        # Test if environmental limits are in DCS tags
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        for env in required_env:
            # Convert co2_kg_per_ton to co2_kg_t for DCS tag
            dcs_env = env.replace('co2_kg_per_ton', 'co2_kg_t')
            results[f'dcs_env_{env}'] = dcs_env in dcs_tags
        
        logger.info(f"Environmental limits conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_dcs_tags_conformity(self) -> Dict[str, bool]:
        """Test if DCS tags match configuration."""
        logger.info("=== Testing DCS Tags Conformity ===")
        
        results = {}
        dcs_config = self.config.get('dcs_tags', {})
        dcs_tags = list(self.dcs_simulator.tags.keys()) if self.dcs_simulator else []
        
        # Test update frequencies
        update_freq = dcs_config.get('update_frequencies', {})
        results['critical_loops_freq'] = 'critical_loops' in update_freq
        results['process_variables_freq'] = 'process_variables' in update_freq
        results['quality_lab_freq'] = 'quality_lab' in update_freq
        
        # Test tag categories
        tag_categories = ['raw_mill', 'preheater', 'kiln', 'fuel_system', 'cooler', 'cement_mill', 'environmental', 'quality_lab']
        for category in tag_categories:
            category_tags = dcs_config.get(category, [])
            results[f'category_{category}'] = len(category_tags) > 0
            
            # Test if category tags are in DCS simulator
            for tag in category_tags:
                results[f'dcs_tag_{tag}'] = tag in dcs_tags
        
        # Test total tag count
        results['total_tags_reasonable'] = len(dcs_tags) >= 40  # Should have 40+ tags
        
        logger.info(f"DCS tags conformity: {sum(results.values())}/{len(results)} passed")
        return results
    
    def test_update_frequencies_usage(self) -> Dict[str, bool]:
        """Test if update frequencies are actually used in the simulator."""
        logger.info("=== Testing Update Frequencies Usage ===")
        
        results = {}
        
        if self.dcs_simulator:
            # Test if simulator has update frequency logic
            results['update_freq_method_exists'] = hasattr(self.dcs_simulator, '_apply_update_frequencies')
            
            # Test if configuration is used
            update_freq = self.config.get('dcs_tags', {}).get('update_frequencies', {})
            results['config_has_frequencies'] = len(update_freq) > 0
            
            # Test if frequencies are reasonable
            critical_freq = update_freq.get('critical_loops', 1)
            process_freq = update_freq.get('process_variables', 5)
            quality_freq = update_freq.get('quality_lab', 3600)
            
            results['critical_freq_reasonable'] = 1 <= critical_freq <= 5
            results['process_freq_reasonable'] = 5 <= process_freq <= 60
            results['quality_freq_reasonable'] = 3600 <= quality_freq <= 86400
        
        logger.info(f"Update frequencies usage: {sum(results.values())}/{len(results)} passed")
        return results
    
    def run_complete_conformity_test(self) -> Dict[str, Dict[str, bool]]:
        """Run complete conformity test suite."""
        logger.info("🧪 Starting Complete Plant Config Conformity Test")
        logger.info("=" * 60)
        
        test_results = {}
        
        # Run all tests
        test_results['plant_metadata'] = self.test_plant_metadata_conformity()
        test_results['raw_materials'] = self.test_raw_materials_conformity()
        test_results['fuel_mix'] = self.test_fuel_mix_conformity()
        test_results['energy_consumption'] = self.test_energy_consumption_conformity()
        test_results['process_parameters'] = self.test_process_parameters_conformity()
        test_results['quality_targets'] = self.test_quality_targets_conformity()
        test_results['environmental_limits'] = self.test_environmental_limits_conformity()
        test_results['dcs_tags'] = self.test_dcs_tags_conformity()
        test_results['update_frequencies'] = self.test_update_frequencies_usage()
        
        # Calculate overall results
        total_tests = sum(len(category_results) for category_results in test_results.values())
        passed_tests = sum(sum(category_results.values()) for category_results in test_results.values())
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 CONFORMITY TEST RESULTS")
        logger.info("=" * 60)
        
        for category, results in test_results.items():
            passed = sum(results.values())
            total = len(results)
            percentage = (passed / total) * 100 if total > 0 else 0
            logger.info(f"{category.upper()}: {passed}/{total} ({percentage:.1f}%)")
        
        overall_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        logger.info(f"\nOVERALL CONFORMITY: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
        
        if overall_percentage >= 90:
            logger.info("🎉 EXCELLENT CONFORMITY - Ready for POC!")
        elif overall_percentage >= 75:
            logger.info("✅ GOOD CONFORMITY - Minor improvements needed")
        elif overall_percentage >= 50:
            logger.info("⚠️ MODERATE CONFORMITY - Significant improvements needed")
        else:
            logger.info("❌ POOR CONFORMITY - Major implementation gaps")
        
        return test_results


if __name__ == "__main__":
    # Run the complete conformity test
    tester = PlantConfigConformityTester()
    results = tester.run_complete_conformity_test()
    
    # Save results
    import json
    with open('artifacts/plant_config_conformity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Results saved to: artifacts/plant_config_conformity_results.json")
