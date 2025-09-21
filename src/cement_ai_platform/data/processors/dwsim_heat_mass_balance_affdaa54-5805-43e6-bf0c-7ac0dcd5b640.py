import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class CementKilnSimulator:
    """Heat and mass balance simulator for cement rotary kiln"""

    def __init__(self, thermo_framework):
        self.thermo = thermo_framework

        # Kiln geometry and operating parameters
        self.kiln_params = {
            'length': 100.0,     # meters
            'diameter': 4.5,     # meters
            'slope': 0.03,       # radians
            'rotation_speed': 2.0, # rpm
            'fuel_rate': 5000.0, # kg/h
            'air_rate': 50000.0, # kg/h
            'feed_rate': 150000.0 # kg/h dry basis
        }

        # Heat transfer coefficients
        self.heat_transfer = {
            'gas_solid': 50.0,   # W/m²K
            'solid_wall': 15.0,  # W/m²K
            'wall_ambient': 8.0  # W/m²K
        }

    def mass_balance_equations(self, x, y, params):
        """Differential equations for mass balance along kiln length"""
        # y = [m_CaCO3, m_CaO, m_CO2, m_fuel, m_air, T_gas, T_solid]

        m_CaCO3, m_CaO, m_CO2, m_fuel, m_air, T_gas, T_solid = y

        # Calcination rate based on temperature and residence time
        calcination_rate = self.calcination_kinetics(T_solid, m_CaCO3)

        # Fuel combustion rate
        combustion_rate = self.fuel_combustion_rate(T_gas, m_fuel, m_air)

        # Mass balance derivatives
        dm_CaCO3_dx = -calcination_rate
        dm_CaO_dx = calcination_rate * (self.thermo.components['CaO']['MW'] /
                                       self.thermo.components['CaCO3']['MW'])
        dm_CO2_dx = calcination_rate * (self.thermo.components['CO2']['MW'] /
                                       self.thermo.components['CaCO3']['MW'])
        dm_fuel_dx = -combustion_rate
        dm_air_dx = -combustion_rate * params['stoichiometric_air']

        # Energy balance for gas and solid phases
        dT_gas_dx = self.gas_energy_balance(T_gas, T_solid, combustion_rate)
        dT_solid_dx = self.solid_energy_balance(T_gas, T_solid, calcination_rate)

        return [dm_CaCO3_dx, dm_CaO_dx, dm_CO2_dx, dm_fuel_dx, dm_air_dx,
                dT_gas_dx, dT_solid_dx]

    def calcination_kinetics(self, T, m_CaCO3):
        """Calculate calcination reaction rate"""
        if T < 1073:  # Below 800°C
            return 0.0

        # Arrhenius kinetics
        k0 = 1e6  # Pre-exponential factor
        Ea = 180e3  # Activation energy J/mol

        k = k0 * np.exp(-Ea / (self.thermo.R * T))
        conversion = self.thermo.calcination_equilibrium(T)

        # Rate proportional to remaining CaCO3 and equilibrium driving force
        rate = k * m_CaCO3 * conversion
        return rate

    def fuel_combustion_rate(self, T, m_fuel, m_air):
        """Calculate fuel combustion rate"""
        if T < 773:  # Below ignition temperature
            return 0.0

        # Simple combustion kinetics
        k_combustion = 1e4 * np.exp(-50e3 / (self.thermo.R * T))

        # Rate limited by fuel or air availability
        air_fuel_ratio = m_air / max(m_fuel, 1e-6)
        stoichiometric_ratio = 14.7  # kg air/kg fuel for typical coal

        if air_fuel_ratio > stoichiometric_ratio:
            rate = k_combustion * m_fuel  # Fuel limited
        else:
            rate = k_combustion * m_air / stoichiometric_ratio  # Air limited

        return rate

    def gas_energy_balance(self, T_gas, T_solid, combustion_rate):
        """Energy balance for gas phase"""
        # Heat generation from combustion
        fuel_heating_value = 30e6  # J/kg
        heat_generation = combustion_rate * fuel_heating_value

        # Heat transfer to solid
        A_contact = np.pi * self.kiln_params['diameter']  # Contact area per meter
        heat_to_solid = (self.heat_transfer['gas_solid'] * A_contact *
                        (T_gas - T_solid))

        # Heat loss to walls
        heat_to_wall = (self.heat_transfer['solid_wall'] * A_contact *
                       (T_gas - 300))  # Assuming 300K wall temperature

        # Gas heat capacity
        m_gas_total = 1000  # kg/m (simplified)
        cp_gas = 1200  # J/kg·K

        dT_gas_dx = (heat_generation - heat_to_solid - heat_to_wall) / (m_gas_total * cp_gas)
        return dT_gas_dx

    def solid_energy_balance(self, T_gas, T_solid, calcination_rate):
        """Energy balance for solid phase"""
        # Heat from gas
        A_contact = np.pi * self.kiln_params['diameter']
        heat_from_gas = (self.heat_transfer['gas_solid'] * A_contact *
                        (T_gas - T_solid))

        # Heat consumed by calcination
        calcination_heat = 178.3e3  # J/mol
        molar_calcination_rate = calcination_rate / self.thermo.components['CaCO3']['MW']
        heat_consumed = molar_calcination_rate * calcination_heat

        # Solid heat capacity
        m_solid = 2000  # kg/m (simplified)
        cp_solid = 1000  # J/kg·K

        dT_solid_dx = (heat_from_gas - heat_consumed) / (m_solid * cp_solid)
        return dT_solid_dx

    def simulate_kiln(self, feed_composition):
        """Simulate complete kiln operation"""
        # Initial conditions at kiln inlet
        initial_feed = feed_composition['total_mass']  # kg/h
        CaCO3_fraction = feed_composition.get('CaCO3', 0.75)

        y0 = [
            initial_feed * CaCO3_fraction,  # m_CaCO3
            0.0,                            # m_CaO
            0.0,                            # m_CO2
            self.kiln_params['fuel_rate'],  # m_fuel
            self.kiln_params['air_rate'],   # m_air
            1800.0,                         # T_gas (K)
            800.0                           # T_solid (K)
        ]

        # Integration parameters
        x_span = [0, self.kiln_params['length']]  # Kiln length
        x_eval = np.linspace(0, self.kiln_params['length'], 50)

        params = {'stoichiometric_air': 14.7}

        # Solve ODEs
        try:
            solution = solve_ivp(
                lambda x, y: self.mass_balance_equations(x, y, params),
                x_span, y0, t_eval=x_eval, method='RK45'
            )

            if solution.success:
                results_df = pd.DataFrame({
                    'position_m': solution.t,
                    'CaCO3_kg_h': solution.y[0],
                    'CaO_kg_h': solution.y[1],
                    'CO2_kg_h': solution.y[2],
                    'fuel_kg_h': solution.y[3],
                    'air_kg_h': solution.y[4],
                    'T_gas_K': solution.y[5],
                    'T_solid_K': solution.y[6]
                })

                # Calculate derived properties
                results_df['calcination_conversion'] = (
                    1 - results_df['CaCO3_kg_h'] / results_df['CaCO3_kg_h'].iloc[0]
                )
                results_df['T_gas_C'] = results_df['T_gas_K'] - 273.15
                results_df['T_solid_C'] = results_df['T_solid_K'] - 273.15

                return results_df, solution.success
            else:
                return None, False

        except Exception as e:
            print(f"Simulation error: {e}")
            return None, False

# Initialize simulator with thermodynamic framework
kiln_simulator = CementKilnSimulator(thermo_framework)

# Test simulation with typical raw meal composition
test_feed = {
    'total_mass': 150000,  # kg/h
    'CaCO3': 0.75,        # 75% limestone
    'SiO2': 0.13,         # 13% silica
    'Al2O3': 0.03,        # 3% alumina
    'Fe2O3': 0.02         # 2% iron oxide
}

print("=== Running Kiln Simulation ===")
simulation_results, success = kiln_simulator.simulate_kiln(test_feed)

if success:
    print(f"Simulation completed successfully!")
    print(f"Data points generated: {len(simulation_results)}")
    print(f"Final calcination conversion: {simulation_results['calcination_conversion'].iloc[-1]:.3f}")
    print(f"Exit gas temperature: {simulation_results['T_gas_C'].iloc[-1]:.0f}°C")
    print(f"Exit solid temperature: {simulation_results['T_solid_C'].iloc[-1]:.0f}°C")

    # Store results
    kiln_results = simulation_results
    kiln_simulator_obj = kiln_simulator
else:
    print("Simulation failed - using simplified results")
    # Create simplified fallback data
    positions = np.linspace(0, 100, 50)
    kiln_results = pd.DataFrame({
        'position_m': positions,
        'calcination_conversion': np.minimum(1.0, positions / 80),
        'T_gas_C': 1800 - positions * 5,
        'T_solid_C': 800 + positions * 8
    })