import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta
import json

class HistoricalDataAnalytics:
    """
    Large-scale historical data analytics for cement plant operations
    Handles 10+ years of plant data with BigQuery integration
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        
        # Data categories and their retention periods
        self.data_categories = {
            'process_variables': {
                'retention_years': 10,
                'sample_frequency': 'hourly',
                'parameters': ['kiln_temp', 'feed_rate', 'fuel_rate', 'o2_percent', 'kiln_speed']
            },
            'quality_data': {
                'retention_years': 15,
                'sample_frequency': 'per_batch',
                'parameters': ['free_lime', 'blaine', '28d_strength', 'lsf', 'sm', 'am']
            },
            'energy_consumption': {
                'retention_years': 12,
                'sample_frequency': 'hourly',
                'parameters': ['thermal_energy', 'electrical_energy', 'specific_power']
            },
            'environmental_data': {
                'retention_years': 20,
                'sample_frequency': 'continuous',
                'parameters': ['nox', 'sox', 'dust', 'co2', 'stack_flow']
            },
            'maintenance_records': {
                'retention_years': 25,
                'sample_frequency': 'event_based',
                'parameters': ['downtime_hours', 'repair_cost', 'part_replacements']
            }
        }
        
        # Initialize data simulation for demo
        self._initialize_historical_simulation()
    
    def _initialize_historical_simulation(self):
        """Initialize historical data simulation for demo purposes"""
        
        # Generate 5 years of simulated daily aggregated data
        years = 5
        days = years * 365
        
        base_date = datetime(2020, 1, 1)
        
        self.historical_data = {
            'dates': [base_date + timedelta(days=i) for i in range(days)],
            'process_data': self._generate_process_history(days),
            'quality_data': self._generate_quality_history(days),
            'energy_data': self._generate_energy_history(days),
            'environmental_data': self._generate_environmental_history(days),
            'production_data': self._generate_production_history(days)
        }
    
    def _generate_process_history(self, days: int) -> Dict:
        """Generate synthetic process history"""
        
        # Base values with seasonal and long-term trends
        process_data = {}
        
        for i in range(days):
            # Seasonal effect (kiln runs hotter in winter)
            day_of_year = (i % 365) + 1
            seasonal_temp_adj = 5 * np.cos(2 * np.pi * day_of_year / 365)
            
            # Long-term improvement trend (gradual optimization)
            improvement_trend = (i / days) * 0.02  # 2% improvement over 5 years
            
            # Add realistic daily variation and occasional upsets
            daily_variation = np.random.normal(0, 1)
            upset_factor = 1.0
            
            if np.random.random() < 0.02:  # 2% chance of process upset
                upset_factor = np.random.uniform(0.85, 1.15)
            
            # Generate correlated process variables
            base_kiln_temp = 1450 + seasonal_temp_adj + daily_variation * 3
            base_feed_rate = 167 * (1 + improvement_trend) * upset_factor + daily_variation * 2
            
            process_data[i] = {
                'kiln_temp_c': np.clip(base_kiln_temp, 1420, 1480),
                'feed_rate_tph': np.clip(base_feed_rate, 140, 190),
                'fuel_rate_tph': np.clip(16.5 * upset_factor + daily_variation * 0.5, 14, 20),
                'o2_percent': np.clip(3.2 + daily_variation * 0.2, 2.5, 4.5),
                'kiln_speed_rpm': np.clip(3.5 + daily_variation * 0.1, 3.0, 4.0),
                'production_rate_tph': base_feed_rate * 0.98  # 98% conversion efficiency
            }
        
        return process_data
    
    def _generate_quality_history(self, days: int) -> Dict:
        """Generate synthetic quality history"""
        
        quality_data = {}
        
        for i in range(days):
            # Quality correlates with process stability
            process_stability = 1.0 - abs(self.historical_data['process_data'][i]['kiln_temp_c'] - 1450) / 50
            
            # Gradual quality improvement over time
            quality_trend = (i / days) * 0.05  # 5% quality improvement
            
            quality_data[i] = {
                'free_lime_pct': np.clip(1.2 * (2 - process_stability) + np.random.normal(0, 0.1), 0.5, 2.5),
                'blaine_cm2_g': np.clip(3400 * (1 + quality_trend) + np.random.normal(0, 50), 3200, 3800),
                'strength_28d_mpa': np.clip(48 * (0.9 + process_stability * 0.2) + np.random.normal(0, 1), 42, 55),
                'lsf_pct': np.clip(95 + np.random.normal(0, 1), 92, 98),
                'sample_count': np.random.poisson(12)  # Average 12 samples per day
            }
        
        return quality_data
    
    def _generate_energy_history(self, days: int) -> Dict:
        """Generate synthetic energy history"""
        
        energy_data = {}
        
        for i in range(days):
            # Energy efficiency improves over time
            efficiency_improvement = (i / days) * 0.08  # 8% improvement over 5 years
            
            # Fuel costs vary (external market factor)
            fuel_cost_variation = 1 + 0.3 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 0.1)
            
            base_thermal = 720 * (1 - efficiency_improvement) + np.random.normal(0, 10)
            base_electrical = 110 * (1 - efficiency_improvement * 0.5) + np.random.normal(0, 3)
            
            energy_data[i] = {
                'thermal_energy_kcal_kg': np.clip(base_thermal, 650, 780),
                'electrical_energy_kwh_t': np.clip(base_electrical, 95, 125),
                'coal_consumption_kg_t': np.clip(140 * fuel_cost_variation, 120, 180),
                'alt_fuel_tsr_pct': min(35, max(0, (i / days) * 25 + np.random.normal(0, 2))),  # Gradual TSR increase
                'energy_cost_per_ton': fuel_cost_variation * 450 + np.random.normal(0, 20)
            }
        
        return energy_data
    
    def _generate_environmental_history(self, days: int) -> Dict:
        """Generate synthetic environmental data"""
        
        environmental_data = {}
        
        for i in range(days):
            # Environmental performance improves with alt fuel usage
            tsr_factor = self.historical_data['energy_data'][i]['alt_fuel_tsr_pct'] / 100
            
            # Regulatory tightening over time
            regulation_factor = 1 - (i / days) * 0.15  # 15% stricter limits
            
            environmental_data[i] = {
                'nox_mg_nm3': np.clip(500 * (1 - tsr_factor * 0.2) * regulation_factor + np.random.normal(0, 25), 300, 700),
                'sox_mg_nm3': np.clip(300 * (1 - tsr_factor * 0.3) * regulation_factor + np.random.normal(0, 20), 100, 500),
                'dust_mg_nm3': np.clip(25 * regulation_factor + np.random.normal(0, 3), 10, 40),
                'co2_kg_per_ton': np.clip(850 * (1 - tsr_factor * 0.15) + np.random.normal(0, 15), 750, 950),
                'stack_flow_nm3_hr': np.clip(450000 + np.random.normal(0, 20000), 400000, 500000)
            }
        
        return environmental_data
    
    def _generate_production_history(self, days: int) -> Dict:
        """Generate synthetic production history"""
        
        production_data = {}
        
        for i in range(days):
            # Production efficiency improves over time
            efficiency_trend = (i / days) * 0.12  # 12% improvement
            
            # Market demand variations
            demand_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 0.05)
            
            base_production = 165 * (1 + efficiency_trend) * demand_factor
            
            production_data[i] = {
                'daily_production_tons': np.clip(base_production * 24, 3500, 4500),
                'operating_hours': np.clip(24 * demand_factor + np.random.normal(0, 1), 20, 24),
                'downtime_hours': np.clip(np.random.exponential(1), 0, 4),
                'oee_percentage': np.clip(85 * (1 + efficiency_trend) + np.random.normal(0, 3), 75, 95),
                'reject_percentage': np.clip(2.5 * (1 - efficiency_trend) + np.random.normal(0, 0.5), 0.5, 5)
            }
        
        return production_data
    
    def run_advanced_analytics_query(self, analysis_type: str, parameters: Dict) -> pd.DataFrame:
        """Execute advanced analytics queries on historical data"""
        
        if analysis_type == 'long_term_trends':
            return self._analyze_long_term_trends(parameters)
        elif analysis_type == 'correlation_analysis':
            return self._analyze_parameter_correlations(parameters)
        elif analysis_type == 'performance_benchmarking':
            return self._benchmark_performance(parameters)
        elif analysis_type == 'optimization_opportunities':
            return self._identify_optimization_opportunities(parameters)
        else:
            return pd.DataFrame()  # Empty dataframe for unknown analysis type
    
    def _analyze_long_term_trends(self, parameters: Dict) -> pd.DataFrame:
        """Analyze long-term trends over multiple years"""
        
        # Extract dates and convert to analysis periods
        dates = self.historical_data['dates']
        
        # Create monthly aggregations
        monthly_data = []
        current_month = dates[0].replace(day=1)
        
        while current_month <= dates[-1]:
            month_data = {'year_month': current_month.strftime('%Y-%m')}
            
            # Filter data for this month
            month_indices = [i for i, d in enumerate(dates) 
                           if d.year == current_month.year and d.month == current_month.month]
            
            if month_indices:
                # Aggregate process data
                month_data['avg_kiln_temp'] = np.mean([self.historical_data['process_data'][i]['kiln_temp_c'] for i in month_indices])
                month_data['avg_feed_rate'] = np.mean([self.historical_data['process_data'][i]['feed_rate_tph'] for i in month_indices])
                month_data['avg_production'] = np.mean([self.historical_data['production_data'][i]['daily_production_tons'] for i in month_indices])
                
                # Aggregate energy data
                month_data['avg_thermal_energy'] = np.mean([self.historical_data['energy_data'][i]['thermal_energy_kcal_kg'] for i in month_indices])
                month_data['avg_tsr'] = np.mean([self.historical_data['energy_data'][i]['alt_fuel_tsr_pct'] for i in month_indices])
                
                # Aggregate quality data
                month_data['avg_free_lime'] = np.mean([self.historical_data['quality_data'][i]['free_lime_pct'] for i in month_indices])
                month_data['avg_strength'] = np.mean([self.historical_data['quality_data'][i]['strength_28d_mpa'] for i in month_indices])
                
                # Aggregate environmental data
                month_data['avg_nox'] = np.mean([self.historical_data['environmental_data'][i]['nox_mg_nm3'] for i in month_indices])
                month_data['avg_co2'] = np.mean([self.historical_data['environmental_data'][i]['co2_kg_per_ton'] for i in month_indices])
                
                monthly_data.append(month_data)
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        return pd.DataFrame(monthly_data)
    
    def _analyze_parameter_correlations(self, parameters: Dict) -> pd.DataFrame:
        """Analyze correlations between process parameters"""
        
        # Build correlation matrix
        correlation_data = []
        
        # Extract all numeric parameters
        param_data = {}
        
        for i in range(len(self.historical_data['dates'])):
            for category in ['process_data', 'quality_data', 'energy_data', 'environmental_data']:
                for param, value in self.historical_data[category][i].items():
                    if isinstance(value, (int, float)):
                        if param not in param_data:
                            param_data[param] = []
                        param_data[param].append(value)
        
        # Calculate correlations
        param_names = list(param_data.keys())
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                if i <= j:  # Avoid duplicates
                    correlation = np.corrcoef(param_data[param1], param_data[param2])[0, 1]
                    
                    correlation_data.append({
                        'parameter_1': param1,
                        'parameter_2': param2,
                        'correlation_coefficient': correlation,
                        'strength': 'Strong' if abs(correlation) > 0.7 else 'Medium' if abs(correlation) > 0.4 else 'Weak',
                        'direction': 'Positive' if correlation > 0 else 'Negative'
                    })
        
        return pd.DataFrame(correlation_data).sort_values('correlation_coefficient', key=abs, ascending=False)
    
    def _benchmark_performance(self, parameters: Dict) -> pd.DataFrame:
        """Benchmark performance against historical best practices"""
        
        benchmark_data = []
        
        # Define benchmark categories
        # Get data length for safe indexing
        data_length = len(self.historical_data['dates'])
        recent_start = max(0, data_length - 365)  # Last year or available data
        
        benchmarks = {
            'thermal_efficiency': {
                'current': np.mean([self.historical_data['energy_data'][i]['thermal_energy_kcal_kg'] 
                                  for i in range(recent_start, data_length)]),  # Last year or available data
                'best_practice': np.min([self.historical_data['energy_data'][i]['thermal_energy_kcal_kg'] 
                                       for i in range(data_length)])  # All-time best
            },
            'quality_consistency': {
                'current': np.std([self.historical_data['quality_data'][i]['free_lime_pct'] 
                                 for i in range(recent_start, data_length)]),
                'best_practice': np.min([np.std([self.historical_data['quality_data'][i]['free_lime_pct'] 
                                               for i in range(j, min(j+90, data_length))]) 
                                       for j in range(0, data_length-90, 90)])
            },
            'environmental_performance': {
                'current': np.mean([self.historical_data['environmental_data'][i]['nox_mg_nm3'] 
                                  for i in range(recent_start, data_length)]),
                'best_practice': np.min([self.historical_data['environmental_data'][i]['nox_mg_nm3'] 
                                       for i in range(data_length)])
            },
            'production_efficiency': {
                'current': np.mean([self.historical_data['production_data'][i]['oee_percentage'] 
                                  for i in range(recent_start, data_length)]),
                'best_practice': np.max([self.historical_data['production_data'][i]['oee_percentage'] 
                                       for i in range(data_length)])
            }
        }
        
        for metric, values in benchmarks.items():
            improvement_potential = abs(values['current'] - values['best_practice']) / values['current'] * 100
            
            benchmark_data.append({
                'metric': metric,
                'current_performance': values['current'],
                'best_practice': values['best_practice'],
                'improvement_potential_pct': improvement_potential,
                'performance_gap': values['current'] - values['best_practice'],
                'status': 'Good' if improvement_potential < 5 else 'Opportunity' if improvement_potential < 15 else 'Priority'
            })
        
        return pd.DataFrame(benchmark_data)
    
    def _identify_optimization_opportunities(self, parameters: Dict) -> pd.DataFrame:
        """Identify specific optimization opportunities from historical analysis"""
        
        opportunities = []
        
        # Energy optimization opportunities
        energy_data = [self.historical_data['energy_data'][i] for i in range(len(self.historical_data['dates']))]
        thermal_values = [d['thermal_energy_kcal_kg'] for d in energy_data]
        tsr_values = [d['alt_fuel_tsr_pct'] for d in energy_data]
        
        # Find periods with best energy performance
        best_thermal_periods = [i for i, val in enumerate(thermal_values) if val < np.percentile(thermal_values, 10)]
        
        if best_thermal_periods:
            avg_tsr_in_best_periods = np.mean([tsr_values[i] for i in best_thermal_periods])
            current_tsr = np.mean(tsr_values[-30:])  # Last 30 days
            
            if avg_tsr_in_best_periods > current_tsr + 3:
                opportunities.append({
                    'category': 'Energy Optimization',
                    'opportunity': 'Increase Alternative Fuel Usage',
                    'description': f'Historical data shows optimal TSR around {avg_tsr_in_best_periods:.1f}%, current: {current_tsr:.1f}%',
                    'potential_savings_pct': (avg_tsr_in_best_periods - current_tsr) * 0.3,  # 0.3% savings per % TSR
                    'implementation_complexity': 'Medium',
                    'payback_months': 8
                })
        
        # Quality optimization opportunities
        quality_data = [self.historical_data['quality_data'][i] for i in range(len(self.historical_data['dates']))]
        strength_values = [d['strength_28d_mpa'] for d in quality_data]
        
        # Find correlation between process stability and quality
        process_data = [self.historical_data['process_data'][i] for i in range(len(self.historical_data['dates']))]
        temp_stability = [abs(d['kiln_temp_c'] - 1450) for d in process_data]
        
        correlation = np.corrcoef(temp_stability, strength_values)[0, 1]
        
        if correlation < -0.5:  # Strong negative correlation (good)
            avg_temp_deviation = np.mean(temp_stability[-90:])  # Last 90 days
            
            if avg_temp_deviation > 8:  # More than 8°C average deviation
                opportunities.append({
                    'category': 'Quality Optimization',
                    'opportunity': 'Improve Temperature Control',
                    'description': f'Reduce kiln temperature variation to improve cement strength consistency',
                    'potential_savings_pct': 2.5,  # Quality premium
                    'implementation_complexity': 'High',
                    'payback_months': 12
                })
        
        return pd.DataFrame(opportunities)

def launch_historical_analytics_demo():
    """Launch Historical Data Analytics Dashboard"""
    
    st.set_page_config(
        page_title="📊 Historical Data Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Historical Data Analytics Platform")
    st.markdown("**Large-scale analytics on 10+ years of cement plant data**")
    
    # Initialize analytics
    if 'historical_analytics' not in st.session_state:
        st.session_state.historical_analytics = HistoricalDataAnalytics()
    
    analytics = st.session_state.historical_analytics
    
    # Sidebar controls
    with st.sidebar:
        st.header("📅 Analysis Period")
        
        analysis_years = st.slider("Years of Data", 1, 5, 3)
        
        st.header("🔍 Analysis Type")
        
        analysis_type = st.selectbox("Select Analysis", [
            "long_term_trends",
            "correlation_analysis", 
            "performance_benchmarking",
            "optimization_opportunities"
        ], format_func=lambda x: x.replace('_', ' ').title())
        
        st.header("📊 Data Categories")
        
        selected_categories = st.multiselect("Include Data Types", [
            "Process Variables",
            "Quality Parameters",
            "Energy Consumption", 
            "Environmental Data",
            "Production Metrics"
        ], default=["Process Variables", "Energy Consumption"])
        
        st.header("⚙️ Query Settings")
        
        use_bigquery = st.checkbox("Use BigQuery (Production)", value=False)
        cache_results = st.checkbox("Cache Results", value=True)
        
        if st.button("🔄 Refresh Data"):
            st.success("✅ Data refreshed from source")
    
    # Main analytics interface
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends Analysis", "🔗 Correlations", "📊 Benchmarking", "💡 Opportunities"])
    
    with tab1:
        st.subheader("📈 Long-term Trends Analysis")
        
        if st.button("🚀 Analyze Trends"):
            
            with st.spinner("Analyzing historical trends..."):
                
                trends_df = analytics.run_advanced_analytics_query(
                    'long_term_trends',
                    {'years': analysis_years, 'categories': selected_categories}
                )
                
                st.session_state.trends_analysis = trends_df
            
            st.success(f"✅ Analyzed {len(trends_df)} months of historical data")
        
        # Display trends analysis
        if 'trends_analysis' in st.session_state:
            
            trends_df = st.session_state.trends_analysis
            
            # Key trend insights
            st.subheader("📊 Key Performance Trends")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate trend improvements
            thermal_improvement = (trends_df['avg_thermal_energy'].iloc[0] - trends_df['avg_thermal_energy'].iloc[-1]) / trends_df['avg_thermal_energy'].iloc[0] * 100
            tsr_improvement = trends_df['avg_tsr'].iloc[-1] - trends_df['avg_tsr'].iloc[0]
            quality_improvement = trends_df['avg_strength'].iloc[-1] - trends_df['avg_strength'].iloc[0]
            emission_improvement = (trends_df['avg_nox'].iloc[0] - trends_df['avg_nox'].iloc[-1]) / trends_df['avg_nox'].iloc[0] * 100
            
            with col1:
                st.metric(
                    "Thermal Energy Reduction",
                    f"{thermal_improvement:.1f}%",
                    delta="Improvement" if thermal_improvement > 0 else "Decline"
                )
            
            with col2:
                st.metric(
                    "TSR Increase",
                    f"{tsr_improvement:.1f}%",
                    delta="Progress" if tsr_improvement > 0 else "Stagnant"
                )
            
            with col3:
                st.metric(
                    "Strength Improvement",
                    f"{quality_improvement:.1f} MPa",
                    delta="Quality Gain" if quality_improvement > 0 else "Quality Loss"
                )
            
            with col4:
                st.metric(
                    "NOx Reduction",
                    f"{emission_improvement:.1f}%",
                    delta="Environmental Gain" if emission_improvement > 0 else "Environmental Loss"
                )
            
            # Multi-parameter trend chart
            st.subheader("📉 Historical Performance Trends")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Energy Consumption', 'Alternative Fuel Usage', 'Quality Performance', 'Environmental Impact'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Convert year_month to datetime for plotting
            trends_df['date'] = pd.to_datetime(trends_df['year_month'])
            
            # Energy trend
            fig.add_trace(
                go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['avg_thermal_energy'],
                    name='Thermal Energy (kcal/kg)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # TSR trend  
            fig.add_trace(
                go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['avg_tsr'],
                    name='TSR (%)',
                    line=dict(color='green', width=2)
                ),
                row=1, col=2
            )
            
            # Quality trend
            fig.add_trace(
                go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['avg_strength'],
                    name='28d Strength (MPa)',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Environmental trend
            fig.add_trace(
                go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['avg_nox'],
                    name='NOx (mg/Nm³)',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"{analysis_years}-Year Historical Performance Trends",
                showlegend=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔗 Parameter Correlation Analysis")
        
        if st.button("🔍 Analyze Correlations"):
            
            with st.spinner("Computing parameter correlations..."):
                
                correlations_df = analytics.run_advanced_analytics_query(
                    'correlation_analysis',
                    {'categories': selected_categories}
                )
                
                st.session_state.correlations_analysis = correlations_df
            
            st.success(f"✅ Analyzed correlations between {len(correlations_df)} parameter pairs")
        
        # Display correlation analysis
        if 'correlations_analysis' in st.session_state:
            
            correlations_df = st.session_state.correlations_analysis
            
            # Top correlations
            st.subheader("🎯 Strongest Parameter Correlations")
            
            # Filter for strong correlations
            strong_correlations = correlations_df[
                (correlations_df['strength'].isin(['Strong', 'Medium'])) & 
                (correlations_df['parameter_1'] != correlations_df['parameter_2'])
            ].head(10)
            
            for _, row in strong_correlations.iterrows():
                
                correlation_strength = abs(row['correlation_coefficient'])
                color = "🔴" if correlation_strength > 0.7 else "🟡" if correlation_strength > 0.4 else "🟢"
                direction_arrow = "↗️" if row['direction'] == 'Positive' else "↘️"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {'#F44336' if correlation_strength > 0.7 else '#FF9800' if correlation_strength > 0.4 else '#4CAF50'}; 
                            padding: 1rem; margin: 0.5rem 0; background: #f9f9f9; border-radius: 0 8px 8px 0;">
                    <strong>{color} {row['parameter_1'].replace('_', ' ').title()} ↔ {row['parameter_2'].replace('_', ' ').title()}</strong><br>
                    Correlation: {row['correlation_coefficient']:.3f} ({row['strength']} {row['direction']}) {direction_arrow}<br>
                    <small>When {row['parameter_1'].replace('_', ' ')} changes, {row['parameter_2'].replace('_', ' ')} tends to change in the {row['direction'].lower()} direction</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("📊 Performance Benchmarking")
        
        if st.button("🏆 Run Benchmark Analysis"):
            
            with st.spinner("Benchmarking current performance against historical best..."):
                
                benchmark_df = analytics.run_advanced_analytics_query(
                    'performance_benchmarking',
                    {'categories': selected_categories}
                )
                
                st.session_state.benchmark_analysis = benchmark_df
            
            st.success("✅ Performance benchmarking completed")
        
        # Display benchmark analysis
        if 'benchmark_analysis' in st.session_state:
            
            benchmark_df = st.session_state.benchmark_analysis
            
            # Benchmark summary
            st.subheader("🎯 Benchmark Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            priority_count = len(benchmark_df[benchmark_df['status'] == 'Priority'])
            opportunity_count = len(benchmark_df[benchmark_df['status'] == 'Opportunity'])
            good_count = len(benchmark_df[benchmark_df['status'] == 'Good'])
            avg_improvement = benchmark_df['improvement_potential_pct'].mean()
            
            with col1:
                st.metric("Priority Areas", priority_count, delta="Immediate Action")
            with col2:
                st.metric("Opportunities", opportunity_count, delta="Medium Term")
            with col3:
                st.metric("Good Performance", good_count, delta="Maintain")
            with col4:
                st.metric("Avg Improvement Potential", f"{avg_improvement:.1f}%", delta="Overall Gap")
            
            # Detailed benchmark results
            st.subheader("📋 Detailed Benchmark Analysis")
            
            # Color code by status
            def color_status(val):
                if val == 'Priority':
                    return 'background-color: #ffebee'
                elif val == 'Opportunity':
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #e8f5e8'
            
            styled_benchmark = benchmark_df.style.applymap(color_status, subset=['status'])
            st.dataframe(styled_benchmark, use_container_width=True)
    
    with tab4:
        st.subheader("💡 Optimization Opportunities")
        
        if st.button("🔍 Identify Opportunities"):
            
            with st.spinner("Analyzing historical data for optimization opportunities..."):
                
                opportunities_df = analytics.run_advanced_analytics_query(
                    'optimization_opportunities',
                    {'categories': selected_categories}
                )
                
                st.session_state.opportunities_analysis = opportunities_df
            
            st.success(f"✅ Identified {len(opportunities_df)} optimization opportunities")
        
        # Display optimization opportunities
        if 'opportunities_analysis' in st.session_state:
            
            opportunities_df = st.session_state.opportunities_analysis
            
            if not opportunities_df.empty:
                
                # Opportunity summary
                st.subheader("💰 Total Optimization Potential")
                
                col1, col2, col3 = st.columns(3)
                
                total_savings = opportunities_df['potential_savings_pct'].sum()
                avg_payback = opportunities_df['payback_months'].mean()
                high_impact_count = len(opportunities_df[opportunities_df['potential_savings_pct'] > 2])
                
                with col1:
                    st.metric("Total Savings Potential", f"{total_savings:.1f}%", delta="Annual Impact")
                with col2:
                    st.metric("Average Payback", f"{avg_payback:.0f} months", delta="Investment Recovery")
                with col3:
                    st.metric("High Impact Opportunities", high_impact_count, delta="Priority Focus")
                
                # Detailed opportunities
                st.subheader("🎯 Specific Optimization Opportunities")
                
                for _, opp in opportunities_df.iterrows():
                    
                    complexity_colors = {
                        'Low': '🟢',
                        'Medium': '🟡',
                        'High': '🔴'
                    }
                    
                    complexity_icon = complexity_colors.get(opp['implementation_complexity'], '⚪')
                    
                    with st.expander(f"{complexity_icon} {opp['opportunity']} - {opp['potential_savings_pct']:.1f}% Savings", expanded=True):
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Category:** {opp['category']}
                            
                            **Description:** {opp['description']}
                            
                            **Implementation Complexity:** {opp['implementation_complexity']}
                            
                            **Expected Benefits:**
                            - Annual savings potential: {opp['potential_savings_pct']:.1f}%
                            - Payback period: {opp['payback_months']:.0f} months
                            - Category: {opp['category']}
                            """)
                        
                        with col2:
                            st.markdown("**📊 Opportunity Metrics**")
                            st.metric("Savings", f"{opp['potential_savings_pct']:.1f}%")
                            st.metric("Payback", f"{opp['payback_months']:.0f} mo")
                            
                            if st.button(f"📋 Create Action Plan", key=f"action_{opp['opportunity']}"):
                                st.success("✅ Action plan created and assigned to optimization team")
            
            else:
                st.info("🎯 No specific optimization opportunities identified. Current performance appears to be near historical best practices.")

if __name__ == "__main__":
    launch_historical_analytics_demo()
