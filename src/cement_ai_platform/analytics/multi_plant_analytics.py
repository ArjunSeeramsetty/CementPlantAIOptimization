# FILE: src/cement_ai_platform/analytics/multi_plant_analytics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MultiPlantPerformanceAnalyzer:
    """Advanced multi-plant performance comparison and benchmarking"""

    def __init__(self, plant_configs: List, physics_models: Dict):
        self.plant_configs = {plant.plant_id: plant for plant in plant_configs}
        self.physics_models = physics_models
        self.performance_data = self._generate_comparative_performance_data()

    def _generate_comparative_performance_data(self) -> pd.DataFrame:
        """Generate comprehensive performance data for all plants"""

        performance_records = []

        for plant_id, config in self.plant_configs.items():
            physics_model = self.physics_models[plant_id]

            # Generate 30 days of performance data
            for day in range(30):
                date = datetime.now() - timedelta(days=day)

                # Generate daily performance metrics
                daily_kpis = physics_model.generate_current_kpis()

                # Add plant-specific performance factors
                performance_factor = self._get_plant_performance_factor(config)

                record = {
                    'date': date,
                    'plant_id': plant_id,
                    'plant_name': config.plant_name,
                    'location': config.location,
                    'capacity_tpd': config.capacity_tpd,
                    'technology_level': config.technology_level,
                    'commissioning_year': config.commissioning_year,

                    # Energy Performance
                    'thermal_energy_kcal_kg': daily_kpis['thermal_energy_kcal_kg'] * performance_factor['energy'],
                    'electrical_energy_kwh_t': daily_kpis['electrical_energy_kwh_t'] * performance_factor['energy'],
                    'energy_cost_per_ton': self._calculate_energy_cost(daily_kpis, config),

                    # Quality Performance
                    'free_lime_pct': daily_kpis['free_lime_pct'] * performance_factor['quality'],
                    'cement_strength_28d': daily_kpis['cement_strength_28d'] * performance_factor['quality'],
                    'quality_consistency_score': self._calculate_quality_consistency(config),

                    # Environmental Performance
                    'nox_emissions_mg_nm3': daily_kpis['nox_emissions_mg_nm3'] * performance_factor['environmental'],
                    'co2_emissions_kg_t': self._calculate_co2_emissions(daily_kpis, config),
                    'dust_emissions_mg_nm3': config.environmental['dust_mg_nm3'] * (0.8 + np.random.uniform(0, 0.4)),

                    # Operational Performance
                    'oee_percentage': daily_kpis['oee_percentage'] * performance_factor['operational'],
                    'production_rate_tph': daily_kpis['production_rate_tph'],
                    'availability_pct': self._calculate_availability(config),
                    'utilization_pct': (daily_kpis['production_rate_tph'] / (config.capacity_tpd / 24)) * 100,

                    # Economic Performance
                    'total_cost_per_ton': self._calculate_total_cost_per_ton(daily_kpis, config),
                    'fuel_cost_per_ton': self._calculate_fuel_cost(daily_kpis, config),
                    'maintenance_cost_per_ton': self._calculate_maintenance_cost(config),

                    # TSR and Sustainability
                    'tsr_percentage': self._calculate_current_tsr(config),
                    'alt_fuel_savings_monthly': self._calculate_alt_fuel_savings(config),
                    'sustainability_score': self._calculate_sustainability_score(config, daily_kpis)
                }

                performance_records.append(record)

        return pd.DataFrame(performance_records)

    def _get_plant_performance_factor(self, config) -> Dict[str, float]:
        """Get plant-specific performance multipliers based on technology and age"""

        # Technology level impact
        tech_factors = {
            'state_of_art': {'energy': 0.92, 'quality': 1.05, 'environmental': 0.85, 'operational': 1.08},
            'advanced': {'energy': 0.98, 'quality': 1.02, 'environmental': 0.95, 'operational': 1.04},
            'basic': {'energy': 1.12, 'quality': 0.96, 'environmental': 1.15, 'operational': 0.94}
        }

        base_factors = tech_factors.get(config.technology_level, tech_factors['advanced'])

        # Age impact (plants degrade over time)
        plant_age = datetime.now().year - config.commissioning_year
        age_factor = max(0.85, 1.0 - (plant_age * 0.008))  # 0.8% degradation per year

        return {
            key: value * age_factor for key, value in base_factors.items()
        }

    def _calculate_energy_cost(self, kpis: Dict, config) -> float:
        """Calculate energy cost per ton of cement"""

        # Fuel cost
        fuel_cost = 0
        total_fuel_thermal = kpis['thermal_energy_kcal_kg']

        for fuel_type, properties in config.fuel_properties.items():
            fuel_fraction = config.fuel_mix.get(fuel_type, 0) / sum(config.fuel_mix.values())
            fuel_thermal = total_fuel_thermal * fuel_fraction
            fuel_consumption_kg = fuel_thermal / properties['cv_kcal_kg']
            fuel_cost += fuel_consumption_kg * properties['cost_per_ton'] / 1000

        # Electrical cost (₹6 per kWh)
        electrical_cost = kpis['electrical_energy_kwh_t'] * 6

        return fuel_cost + electrical_cost

    def _calculate_co2_emissions(self, kpis: Dict, config) -> float:
        """Calculate CO2 emissions per ton of cement"""

        co2_emissions = 0
        total_fuel_thermal = kpis['thermal_energy_kcal_kg']

        for fuel_type, properties in config.fuel_properties.items():
            fuel_fraction = config.fuel_mix.get(fuel_type, 0) / sum(config.fuel_mix.values())
            fuel_thermal = total_fuel_thermal * fuel_fraction
            fuel_consumption_kg = fuel_thermal / properties['cv_kcal_kg']
            co2_emissions += fuel_consumption_kg * properties['carbon_factor'] / 1000

        # Add process emissions (calcination)
        process_co2 = 525  # kg CO2/ton cement from limestone calcination

        return co2_emissions + process_co2

    def _calculate_quality_consistency(self, config) -> float:
        """Calculate quality consistency score (0-100)"""

        # Higher technology plants have better consistency
        base_consistency = {
            'state_of_art': 95,
            'advanced': 88,
            'basic': 82
        }.get(config.technology_level, 85)

        # Add some realistic variation
        return base_consistency + np.random.uniform(-3, 3)

    def _calculate_availability(self, config) -> float:
        """Calculate plant availability percentage"""

        base_availability = {
            'state_of_art': 96,
            'advanced': 92,
            'basic': 88
        }.get(config.technology_level, 90)

        # Age impact
        plant_age = datetime.now().year - config.commissioning_year
        age_impact = max(0, plant_age * 0.3)  # 0.3% loss per year

        return base_availability - age_impact + np.random.uniform(-2, 2)

    def _calculate_total_cost_per_ton(self, kpis: Dict, config) -> float:
        """Calculate total production cost per ton"""

        energy_cost = self._calculate_energy_cost(kpis, config)
        maintenance_cost = self._calculate_maintenance_cost(config)
        labor_cost = 450 + (config.capacity_tpd / 1000) * 50  # Scale with capacity
        other_costs = 280  # Raw materials, utilities, overhead

        return energy_cost + maintenance_cost + labor_cost + other_costs

    def _calculate_fuel_cost(self, kpis: Dict, config) -> float:
        """Calculate fuel cost per ton"""

        fuel_cost = 0
        total_fuel_thermal = kpis['thermal_energy_kcal_kg']

        for fuel_type, properties in config.fuel_properties.items():
            fuel_fraction = config.fuel_mix.get(fuel_type, 0) / sum(config.fuel_mix.values())
            fuel_thermal = total_fuel_thermal * fuel_fraction
            fuel_consumption_kg = fuel_thermal / properties['cv_kcal_kg']
            fuel_cost += fuel_consumption_kg * properties['cost_per_ton'] / 1000

        return fuel_cost

    def _calculate_maintenance_cost(self, config) -> float:
        """Calculate maintenance cost per ton"""

        base_maintenance = {
            'state_of_art': 180,
            'advanced': 220,
            'basic': 280
        }.get(config.technology_level, 220)

        # Age impact
        plant_age = datetime.now().year - config.commissioning_year
        age_factor = 1 + (plant_age * 0.02)  # 2% increase per year

        return base_maintenance * age_factor

    def _calculate_current_tsr(self, config) -> float:
        """Calculate current TSR percentage"""

        alt_fuels = ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        alt_fuel_total = sum(config.fuel_mix.get(fuel, 0) for fuel in alt_fuels)
        total_fuel = sum(config.fuel_mix.values())

        if total_fuel > 0:
            tsr = (alt_fuel_total / total_fuel) * 100
        else:
            tsr = 0

        # Add some realistic variation
        return tsr + np.random.uniform(-2, 2)

    def _calculate_alt_fuel_savings(self, config) -> float:
        """Calculate monthly savings from alternative fuels"""

        alt_fuels = ['rdf', 'biomass', 'tire_derived', 'plastic_waste']
        monthly_savings = 0

        for fuel in alt_fuels:
            if fuel in config.fuel_mix and fuel in config.fuel_properties:
                alt_fuel_rate = config.fuel_mix[fuel] * 24 * 30 * 1000  # kg/month
                coal_cost = alt_fuel_rate * config.fuel_properties['coal']['cost_per_ton'] / 1000
                alt_fuel_cost = alt_fuel_rate * config.fuel_properties[fuel]['cost_per_ton'] / 1000
                monthly_savings += coal_cost - alt_fuel_cost

        return max(0, monthly_savings)

    def _calculate_sustainability_score(self, config, kpis: Dict) -> float:
        """Calculate overall sustainability score (0-100)"""

        # TSR component (30% weightage)
        tsr_score = min(100, self._calculate_current_tsr(config) * 2.5)

        # CO2 reduction component (25% weightage)
        baseline_co2 = 850  # Industry baseline
        actual_co2 = self._calculate_co2_emissions(kpis, config)
        co2_score = max(0, (baseline_co2 - actual_co2) / baseline_co2 * 100)

        # Energy efficiency component (25% weightage)
        baseline_thermal = 3200
        actual_thermal = kpis['thermal_energy_kcal_kg']
        energy_score = max(0, (baseline_thermal - actual_thermal) / baseline_thermal * 100)

        # Environmental compliance component (20% weightage)
        nox_compliance = max(0, (600 - kpis['nox_emissions_mg_nm3']) / 600 * 100)

        sustainability_score = (
            tsr_score * 0.3 +
            co2_score * 0.25 +
            energy_score * 0.25 +
            nox_compliance * 0.2
        )

        return min(100, max(0, sustainability_score))

    def generate_performance_comparison(self) -> Dict:
        """Generate comprehensive performance comparison"""

        # Calculate average performance metrics for each plant
        plant_summary = self.performance_data.groupby(['plant_id', 'plant_name', 'technology_level']).agg({
            'thermal_energy_kcal_kg': 'mean',
            'electrical_energy_kwh_t': 'mean',
            'energy_cost_per_ton': 'mean',
            'free_lime_pct': 'mean',
            'cement_strength_28d': 'mean',
            'quality_consistency_score': 'mean',
            'nox_emissions_mg_nm3': 'mean',
            'co2_emissions_kg_t': 'mean',
            'oee_percentage': 'mean',
            'availability_pct': 'mean',
            'utilization_pct': 'mean',
            'total_cost_per_ton': 'mean',
            'tsr_percentage': 'mean',
            'sustainability_score': 'mean'
        }).round(2).reset_index()

        # Calculate rankings
        rankings = self._calculate_plant_rankings(plant_summary)

        # Generate insights
        insights = self._generate_performance_insights(plant_summary)

        # Calculate improvement opportunities
        improvement_opportunities = self._identify_improvement_opportunities(plant_summary)

        return {
            'plant_summary': plant_summary,
            'rankings': rankings,
            'insights': insights,
            'improvement_opportunities': improvement_opportunities,
            'benchmark_analysis': self._generate_benchmark_analysis(plant_summary)
        }

    def _calculate_plant_rankings(self, summary_df: pd.DataFrame) -> Dict:
        """Calculate plant rankings across different metrics"""

        rankings = {}

        # Energy efficiency ranking (lower is better)
        rankings['energy_efficiency'] = summary_df.nsmallest(3, 'thermal_energy_kcal_kg')[
            ['plant_name', 'thermal_energy_kcal_kg', 'technology_level']
        ].to_dict('records')

        # Quality ranking (higher strength, lower free lime variation)
        summary_df['quality_score'] = (
            summary_df['cement_strength_28d'] * 2 -
            abs(summary_df['free_lime_pct'] - 1.2) * 10 +
            summary_df['quality_consistency_score']
        )
        rankings['quality'] = summary_df.nlargest(3, 'quality_score')[
            ['plant_name', 'cement_strength_28d', 'free_lime_pct', 'quality_consistency_score']
        ].to_dict('records')

        # Environmental performance (lower emissions)
        summary_df['environmental_score'] = (
            100 - (summary_df['co2_emissions_kg_t'] / 10) -
            (summary_df['nox_emissions_mg_nm3'] / 10)
        )
        rankings['environmental'] = summary_df.nlargest(3, 'environmental_score')[
            ['plant_name', 'co2_emissions_kg_t', 'nox_emissions_mg_nm3', 'tsr_percentage']
        ].to_dict('records')

        # Overall operational excellence
        summary_df['operational_score'] = (
            summary_df['oee_percentage'] * 0.4 +
            summary_df['availability_pct'] * 0.3 +
            summary_df['utilization_pct'] * 0.3
        )
        rankings['operational'] = summary_df.nlargest(3, 'operational_score')[
            ['plant_name', 'oee_percentage', 'availability_pct', 'utilization_pct']
        ].to_dict('records')

        # Cost efficiency (lower cost is better)
        rankings['cost_efficiency'] = summary_df.nsmallest(3, 'total_cost_per_ton')[
            ['plant_name', 'total_cost_per_ton', 'energy_cost_per_ton']
        ].to_dict('records')

        # Sustainability ranking
        rankings['sustainability'] = summary_df.nlargest(3, 'sustainability_score')[
            ['plant_name', 'sustainability_score', 'tsr_percentage', 'co2_emissions_kg_t']
        ].to_dict('records')

        return rankings

    def _generate_performance_insights(self, summary_df: pd.DataFrame) -> List[str]:
        """Generate key performance insights"""

        insights = []

        # Best performing plant overall
        best_plant = summary_df.loc[summary_df['sustainability_score'].idxmax()]
        insights.append(
            f"🏆 {best_plant['plant_name']} leads in overall sustainability with a score of "
            f"{best_plant['sustainability_score']:.1f}/100, leveraging {best_plant['technology_level']} technology."
        )

        # Energy efficiency leader
        energy_leader = summary_df.loc[summary_df['thermal_energy_kcal_kg'].idxmin()]
        energy_savings = summary_df['thermal_energy_kcal_kg'].max() - energy_leader['thermal_energy_kcal_kg']
        insights.append(
            f"⚡ {energy_leader['plant_name']} demonstrates best energy efficiency at "
            f"{energy_leader['thermal_energy_kcal_kg']:.0f} kcal/kg, saving {energy_savings:.0f} kcal/kg vs. others."
        )

        # TSR opportunity
        tsr_gap = summary_df['tsr_percentage'].max() - summary_df['tsr_percentage'].min()
        if tsr_gap > 10:
            insights.append(
                f"🌱 Significant TSR improvement opportunity: {tsr_gap:.1f}% gap between "
                f"highest and lowest performing plants in alternative fuel usage."
            )

        # Cost optimization potential
        cost_gap = summary_df['total_cost_per_ton'].max() - summary_df['total_cost_per_ton'].min()
        insights.append(
            f"💰 Cost optimization potential: ₹{cost_gap:.0f}/ton gap represents "
            f"significant savings opportunity for underperforming plants."
        )

        # Technology correlation
        tech_correlation = summary_df.groupby('technology_level').agg({
            'thermal_energy_kcal_kg': 'mean',
            'sustainability_score': 'mean'
        })

        if len(tech_correlation) > 1:
            insights.append(
                f"🔬 Technology impact: State-of-art plants show {tech_correlation.loc['state_of_art', 'thermal_energy_kcal_kg'] - tech_correlation.loc['basic', 'thermal_energy_kcal_kg']:.0f} kcal/kg "
                f"better energy efficiency vs. basic technology plants."
            )

        return insights

    def _identify_improvement_opportunities(self, summary_df: pd.DataFrame) -> Dict:
        """Identify specific improvement opportunities for each plant"""

        opportunities = {}

        for _, plant in summary_df.iterrows():
            plant_opportunities = []

            # Energy efficiency opportunity
            best_thermal = summary_df['thermal_energy_kcal_kg'].min()
            if plant['thermal_energy_kcal_kg'] > best_thermal + 50:
                savings_potential = ((plant['thermal_energy_kcal_kg'] - best_thermal) /
                                   plant['thermal_energy_kcal_kg']) * 100
                plant_opportunities.append({
                    'category': 'Energy Efficiency',
                    'opportunity': f"Reduce thermal energy by {savings_potential:.1f}%",
                    'impact': f"₹{savings_potential * 30:.0f}/ton annual savings",
                    'benchmark': f"{best_thermal:.0f} kcal/kg (industry best)"
                })

            # TSR opportunity
            best_tsr = summary_df['tsr_percentage'].max()
            if plant['tsr_percentage'] < best_tsr - 5:
                tsr_potential = best_tsr - plant['tsr_percentage']
                plant_opportunities.append({
                    'category': 'Alternative Fuels',
                    'opportunity': f"Increase TSR by {tsr_potential:.1f}%",
                    'impact': f"₹{tsr_potential * 500:.0f}k monthly fuel cost savings",
                    'benchmark': f"{best_tsr:.1f}% TSR (company best)"
                })

            # Quality consistency opportunity
            best_consistency = summary_df['quality_consistency_score'].max()
            if plant['quality_consistency_score'] < best_consistency - 5:
                plant_opportunities.append({
                    'category': 'Quality Control',
                    'opportunity': f"Improve consistency by {best_consistency - plant['quality_consistency_score']:.1f} points",
                    'impact': "2-3% quality premium potential",
                    'benchmark': f"{best_consistency:.1f} consistency score"
                })

            # OEE opportunity
            best_oee = summary_df['oee_percentage'].max()
            if plant['oee_percentage'] < best_oee - 3:
                oee_potential = best_oee - plant['oee_percentage']
                plant_opportunities.append({
                    'category': 'Operational Excellence',
                    'opportunity': f"Increase OEE by {oee_potential:.1f}%",
                    'impact': f"{oee_potential * plant['capacity_tpd'] / 100:.0f} TPD additional capacity",
                    'benchmark': f"{best_oee:.1f}% OEE"
                })

            opportunities[plant['plant_name']] = plant_opportunities

        return opportunities

    def _generate_benchmark_analysis(self, summary_df: pd.DataFrame) -> Dict:
        """Generate industry benchmark analysis"""

        # Industry benchmarks (external data)
        industry_benchmarks = {
            'thermal_energy_kcal_kg': 3200,
            'electrical_energy_kwh_t': 95,
            'co2_emissions_kg_t': 850,
            'nox_emissions_mg_nm3': 500,
            'oee_percentage': 85,
            'tsr_percentage': 15
        }

        benchmark_analysis = {}

        for metric, benchmark in industry_benchmarks.items():
            plant_values = summary_df[metric]

            if metric in ['thermal_energy_kcal_kg', 'electrical_energy_kwh_t', 'co2_emissions_kg_t', 'nox_emissions_mg_nm3']:
                # Lower is better
                better_than_benchmark = (plant_values < benchmark).sum()
                performance_vs_benchmark = ((benchmark - plant_values.mean()) / benchmark * 100)
            else:
                # Higher is better
                better_than_benchmark = (plant_values > benchmark).sum()
                performance_vs_benchmark = ((plant_values.mean() - benchmark) / benchmark * 100)

            benchmark_analysis[metric] = {
                'industry_benchmark': benchmark,
                'jk_cement_average': plant_values.mean(),
                'plants_better_than_benchmark': better_than_benchmark,
                'total_plants': len(plant_values),
                'performance_vs_benchmark_pct': performance_vs_benchmark,
                'best_plant_value': plant_values.min() if metric in ['thermal_energy_kcal_kg', 'co2_emissions_kg_t'] else plant_values.max()
            }

        return benchmark_analysis


# Integration point for the enhanced dashboard
def create_enhanced_multi_plant_dashboard():
    """Create enhanced multi-plant dashboard with dynamic data"""

    # Load enhanced plant configurations
    from cement_ai_platform.config.enhanced_plant_config import EnhancedPlantConfigManager
    from cement_ai_platform.models.physics_models import DynamicProcessDataGenerator

    config_manager = EnhancedPlantConfigManager()
    plant_configs = config_manager.get_all_plants()

    # Create physics models for each plant
    physics_models = {}
    for plant_config in plant_configs:
        physics_models[plant_config.plant_id] = DynamicProcessDataGenerator(plant_config)

    # Create performance analyzer
    analyzer = MultiPlantPerformanceAnalyzer(plant_configs, physics_models)

    # Generate comparison data
    comparison_data = analyzer.generate_performance_comparison()

    return comparison_data, analyzer.performance_data
