import numpy as np
import pandas as pd
from scipy.optimize import fsolve

class ProcessUnitModels:
    """Collection of process unit models for cement manufacturing"""

    def __init__(self, thermo_framework):
        self.thermo = thermo_framework

    def cyclone_separator(self, inlet_conditions, efficiency_curve=None):
        """Model cyclone separator for dust collection"""

        if efficiency_curve is None:
            # Default Barth model for cyclone efficiency
            def efficiency_curve(dp):  # dp in micrometers
                dp_50 = 5.0  # Cut size in micrometers
                return 1 - np.exp(-(dp/dp_50)**2)

        # Particle size distribution (simplified)
        particle_sizes = np.logspace(0, 2, 20)  # 1 to 100 micrometers
        mass_fraction = np.exp(-particle_sizes/20) / np.sum(np.exp(-particle_sizes/20))

        # Calculate collection efficiency for each size
        collection_eff = efficiency_curve(particle_sizes)
        overall_efficiency = np.sum(mass_fraction * collection_eff)

        # Mass balance
        inlet_dust = inlet_conditions.get('dust_loading', 50.0)  # g/m³
        collected_dust = inlet_dust * overall_efficiency
        outlet_dust = inlet_dust - collected_dust

        # Pressure drop calculation (simplified)
        inlet_velocity = inlet_conditions.get('velocity', 15.0)  # m/s
        pressure_drop = 0.5 * 1.2 * inlet_velocity**2 * 8  # Pa

        return {
            'overall_efficiency': overall_efficiency,
            'collected_dust_g_m3': collected_dust,
            'outlet_dust_g_m3': outlet_dust,
            'pressure_drop_Pa': pressure_drop,
            'size_efficiency': dict(zip(particle_sizes, collection_eff))
        }

    def heat_exchanger(self, hot_stream, cold_stream, effectiveness=0.8):
        """Model heat exchanger (preheater/cooler)"""

        # Stream properties
        hot_T_in = hot_stream['temperature']  # K
        hot_m_cp = hot_stream['mass_flow'] * hot_stream['cp']  # W/K

        cold_T_in = cold_stream['temperature']  # K
        cold_m_cp = cold_stream['mass_flow'] * cold_stream['cp']  # W/K

        # Heat exchanger calculations
        C_min = min(hot_m_cp, cold_m_cp)
        C_max = max(hot_m_cp, cold_m_cp)
        C_ratio = C_min / C_max

        # Maximum possible heat transfer
        Q_max = C_min * (hot_T_in - cold_T_in)

        # Actual heat transfer
        Q_actual = effectiveness * Q_max

        # Outlet temperatures
        hot_T_out = hot_T_in - Q_actual / hot_m_cp
        cold_T_out = cold_T_in + Q_actual / cold_m_cp

        # Pressure drops (simplified)
        hot_dp = 500  # Pa
        cold_dp = 300  # Pa

        return {
            'heat_transfer_W': Q_actual,
            'hot_outlet_T': hot_T_out,
            'cold_outlet_T': cold_T_out,
            'hot_pressure_drop': hot_dp,
            'cold_pressure_drop': cold_dp,
            'effectiveness': Q_actual / Q_max if Q_max > 0 else 0
        }

    def raw_mill(self, feed_composition, operating_conditions):
        """Model raw material grinding mill"""

        feed_rate = operating_conditions.get('feed_rate', 100.0)  # t/h
        mill_power = operating_conditions.get('power', 5000.0)  # kW

        # Grinding kinetics (simplified Bond's law)
        work_index = 15.0  # kWh/t (typical for limestone)
        feed_size = operating_conditions.get('feed_F80', 50.0)  # mm
        product_size = operating_conditions.get('product_F80', 0.09)  # mm

        # Calculate specific energy
        specific_energy = work_index * (1/np.sqrt(product_size) - 1/np.sqrt(feed_size))
        required_power = feed_rate * specific_energy

        # Mill efficiency
        mill_efficiency = min(1.0, mill_power / required_power) if required_power > 0 else 1.0
        actual_product_size = feed_size / (1 + mill_efficiency * 10)

        # Heat generation from grinding
        heat_generation = mill_power * 0.95  # 95% of power becomes heat

        # Product temperature rise
        cp_solid = 1000  # J/kg·K
        temp_rise = heat_generation * 3600 / (feed_rate * 1000 * cp_solid)  # K

        return {
            'specific_energy_kWh_t': specific_energy,
            'mill_efficiency': mill_efficiency,
            'actual_F80_mm': actual_product_size,
            'heat_generation_kW': heat_generation,
            'temperature_rise_K': temp_rise,
            'throughput_t_h': feed_rate * mill_efficiency
        }

    def preheater_tower(self, raw_meal_feed, gas_conditions, n_stages=5):
        """Model multi-stage suspension preheater"""

        # Initialize arrays for each stage
        stage_results = []

        # Starting conditions
        meal_temp = raw_meal_feed.get('temperature', 300)  # K
        meal_flow = raw_meal_feed.get('mass_flow', 150)  # t/h

        gas_temp = gas_conditions.get('temperature', 1200)  # K
        gas_flow = gas_conditions.get('mass_flow', 200)  # t/h

        # Stage-by-stage calculation (counter-current flow)
        for stage in range(n_stages):
            # Heat transfer calculation
            heat_exchanger_result = self.heat_exchanger(
                hot_stream={
                    'temperature': gas_temp,
                    'mass_flow': gas_flow * 1000 / 3600,  # kg/s
                    'cp': 1200  # J/kg·K
                },
                cold_stream={
                    'temperature': meal_temp,
                    'mass_flow': meal_flow * 1000 / 3600,  # kg/s
                    'cp': 1000  # J/kg·K
                },
                effectiveness=0.7  # Reduced due to pneumatic transport
            )

            # Update temperatures
            meal_temp = heat_exchanger_result['cold_outlet_T']
            gas_temp = heat_exchanger_result['hot_outlet_T']

            # Calcination in this stage (if temperature is high enough)
            if meal_temp > 1073:  # 800°C
                calcination_fraction = min(0.2, (meal_temp - 1073) / 200)  # Max 20% per stage
            else:
                calcination_fraction = 0.0

            # Cyclone separation efficiency
            cyclone_result = self.cyclone_separator({
                'dust_loading': 50.0,
                'velocity': 12.0
            })

            stage_results.append({
                'stage': stage + 1,
                'meal_temp_K': meal_temp,
                'gas_temp_K': gas_temp,
                'calcination_fraction': calcination_fraction,
                'heat_transfer_MW': heat_exchanger_result['heat_transfer_W'] / 1e6,
                'cyclone_efficiency': cyclone_result['overall_efficiency']
            })

        # Overall performance
        total_calcination = sum(stage['calcination_fraction'] for stage in stage_results)
        final_meal_temp = stage_results[-1]['meal_temp_K']

        preheater_results_df = pd.DataFrame(stage_results)

        return {
            'stage_results': preheater_results_df,
            'total_calcination_fraction': total_calcination,
            'final_meal_temperature_K': final_meal_temp,
            'gas_exit_temperature_K': gas_temp,
            'preheating_efficiency': (final_meal_temp - 300) / (1200 - 300)
        }

# Initialize process unit models
process_units = ProcessUnitModels(thermo_framework)

print("=== Testing Process Unit Models ===")

# Test cyclone separator
cyclone_test = process_units.cyclone_separator({
    'dust_loading': 75.0,  # g/m³
    'velocity': 18.0       # m/s
})
print(f"Cyclone efficiency: {cyclone_test['overall_efficiency']:.3f}")
print(f"Pressure drop: {cyclone_test['pressure_drop_Pa']:.0f} Pa")

# Test heat exchanger
hx_test = process_units.heat_exchanger(
    hot_stream={'temperature': 1400, 'mass_flow': 50, 'cp': 1200},
    cold_stream={'temperature': 400, 'mass_flow': 100, 'cp': 1000}
)
print(f"Heat transfer: {hx_test['heat_transfer_W']/1e6:.2f} MW")

# Test preheater tower
raw_meal = {'temperature': 300, 'mass_flow': 150}
hot_gas = {'temperature': 1200, 'mass_flow': 200}

preheater_results = process_units.preheater_tower(raw_meal, hot_gas)
print(f"Preheater calcination: {preheater_results['total_calcination_fraction']:.3f}")
print(f"Final meal temperature: {preheater_results['final_meal_temperature_K']-273:.0f}°C")

# Store results for downstream use
process_unit_results = {
    'cyclone': cyclone_test,
    'heat_exchanger': hx_test,
    'preheater': preheater_results,
    'process_units_obj': process_units
}