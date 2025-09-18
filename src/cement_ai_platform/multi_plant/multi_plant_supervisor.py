# NEW FILE: src/cement_ai_platform/multi_plant/multi_plant_supervisor.py
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from .plant_manager import MultiPlantManager, PlantConfiguration
from ..streaming.pubsub_simulator import CementPlantPubSubSimulator, RealTimeDataProcessor

logger = logging.getLogger(__name__)

class MultiPlantSupervisor:
    """
    Supervising Agent that orchestrates across multiple plant-level agents
    Provides centralized coordination, aggregation, and cross-plant decision making
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        
        # Initialize plant manager
        self.manager = MultiPlantManager(project_id)
        
        # Plant-level agents and processors
        self.plant_processors = {}
        self.plant_simulators = {}
        
        # Cross-plant state
        self.tenant_plants = {}
        self.aggregated_metrics = {}
        self.cross_plant_alerts = []
        self.orchestration_state = {
            'initialized': False,
            'active_plants': 0,
            'last_aggregation': None,
            'tenant_health_score': 0.0
        }
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.orchestration_thread = None
        
        # Initialize supervisor
        self._initialize_supervisor()
    
    def _initialize_supervisor(self):
        """Initialize the supervising agent"""
        
        logger.info("🚀 Initializing MultiPlantSupervisor...")
        
        # Discover active plants from registry
        self._discover_active_plants()
        
        # Initialize plant-level agents
        self._initialize_plant_agents()
        
        # Setup cross-plant orchestration
        self._setup_orchestration()
        
        self.orchestration_state['initialized'] = True
        logger.info("✅ MultiPlantSupervisor initialized successfully")
    
    def _discover_active_plants(self):
        """Discover active plants from Firestore registry"""
        
        logger.info("🔍 Discovering active plants...")
        
        # Get all tenants and their plants
        all_plants = list(self.manager.plant_registry.values())
        
        # Group by tenant
        for plant in all_plants:
            tenant_id = plant.tenant_id
            if tenant_id not in self.tenant_plants:
                self.tenant_plants[tenant_id] = []
            self.tenant_plants[tenant_id].append(plant)
        
        logger.info(f"📊 Discovered {len(all_plants)} plants across {len(self.tenant_plants)} tenants")
        
        for tenant_id, plants in self.tenant_plants.items():
            logger.info(f"   {tenant_id}: {len(plants)} plants")
    
    def _initialize_plant_agents(self):
        """Initialize plant-level agents for each discovered plant"""
        
        logger.info("🏭 Initializing plant-level agents...")
        
        for tenant_id, plants in self.tenant_plants.items():
            for plant in plants:
                try:
                    # Create real-time processor
                    processor = RealTimeDataProcessor(plant.plant_id)
                    
                    # Create Pub/Sub simulator
                    simulator = CementPlantPubSubSimulator(self.project_id)
                    
                    # Store references
                    self.plant_processors[plant.plant_id] = processor
                    self.plant_simulators[plant.plant_id] = simulator
                    
                    logger.info(f"✅ Initialized processor for {plant.plant_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize processor for {plant.plant_id}: {e}")
        
        self.orchestration_state['active_plants'] = len(self.plant_processors)
        logger.info(f"🏭 Initialized {self.orchestration_state['active_plants']} plant processors")
    
    def _setup_orchestration(self):
        """Setup cross-plant orchestration and monitoring"""
        
        logger.info("🎼 Setting up cross-plant orchestration...")
        
        # Initialize aggregated metrics structure
        self.aggregated_metrics = {
            'tenant_metrics': {},
            'plant_health_scores': {},
            'cross_plant_benchmarks': {},
            'alert_summary': {},
            'performance_trends': {}
        }
        
        logger.info("✅ Orchestration setup completed")
    
    def start_orchestration(self):
        """Start the cross-plant orchestration"""
        
        if self.running:
            logger.warning("⚠️ Orchestration already running")
            return
        
        logger.info("🚀 Starting cross-plant orchestration...")
        
        self.running = True
        
        # Start orchestration thread
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop)
        self.orchestration_thread.daemon = True
        self.orchestration_thread.start()
        
        # Start plant-level streaming
        self._start_plant_streaming()
        
        logger.info("✅ Cross-plant orchestration started")
    
    def stop_orchestration(self):
        """Stop the cross-plant orchestration"""
        
        logger.info("⏹️ Stopping cross-plant orchestration...")
        
        self.running = False
        
        # Stop plant-level streaming
        self._stop_plant_streaming()
        
        # Wait for orchestration thread to finish
        if self.orchestration_thread and self.orchestration_thread.is_alive():
            self.orchestration_thread.join(timeout=5)
        
        logger.info("✅ Cross-plant orchestration stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop"""
        
        logger.info("🔄 Starting orchestration loop...")
        
        while self.running:
            try:
                # Aggregate metrics from all plants
                self._aggregate_plant_metrics()
                
                # Perform cross-plant analysis
                self._perform_cross_plant_analysis()
                
                # Execute cross-plant decisions
                self._execute_cross_plant_decisions()
                
                # Update tenant health score
                self._update_tenant_health_score()
                
                # Sleep before next iteration
                time.sleep(30)  # 30-second orchestration cycle
                
            except Exception as e:
                logger.error(f"❌ Error in orchestration loop: {e}")
                time.sleep(10)  # Shorter sleep on error
    
    def _start_plant_streaming(self):
        """Start real-time streaming for all plants"""
        
        logger.info("📡 Starting plant-level streaming...")
        
        for plant_id, simulator in self.plant_simulators.items():
            try:
                # Subscribe to plant-specific streams
                simulator.subscribe_to_stream(
                    "process-variables",
                    lambda data, pid=plant_id: self._handle_plant_data(pid, data)
                )
                simulator.subscribe_to_stream(
                    "equipment-health",
                    lambda data, pid=plant_id: self._handle_equipment_data(pid, data)
                )
                
                logger.info(f"📡 Started streaming for {plant_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to start streaming for {plant_id}: {e}")
    
    def _stop_plant_streaming(self):
        """Stop real-time streaming for all plants"""
        
        logger.info("📡 Stopping plant-level streaming...")
        
        for plant_id, simulator in self.plant_simulators.items():
            try:
                simulator.stop_streaming()
                logger.info(f"📡 Stopped streaming for {plant_id}")
            except Exception as e:
                logger.error(f"❌ Error stopping streaming for {plant_id}: {e}")
    
    def _handle_plant_data(self, plant_id: str, data: Dict):
        """Handle incoming plant process data"""
        
        try:
            # Process data through plant-specific processor
            processor = self.plant_processors.get(plant_id)
            if processor:
                processor.process_process_variables(data)
            
            # Update aggregated metrics
            self._update_plant_metrics(plant_id, data)
            
        except Exception as e:
            logger.error(f"❌ Error handling plant data for {plant_id}: {e}")
    
    def _handle_equipment_data(self, plant_id: str, data: Dict):
        """Handle incoming equipment health data"""
        
        try:
            # Process equipment data
            processor = self.plant_processors.get(plant_id)
            if processor:
                processor.process_equipment_health(data)
            
            # Check for cross-plant alerts
            self._check_cross_plant_alerts(plant_id, data)
            
        except Exception as e:
            logger.error(f"❌ Error handling equipment data for {plant_id}: {e}")
    
    def _aggregate_plant_metrics(self):
        """Aggregate metrics from all plants"""
        
        logger.debug("📊 Aggregating plant metrics...")
        
        tenant_metrics = {}
        
        for tenant_id, plants in self.tenant_plants.items():
            tenant_metrics[tenant_id] = {
                'total_plants': len(plants),
                'active_plants': 0,
                'total_capacity': 0,
                'avg_health_score': 0.0,
                'critical_alerts': 0,
                'performance_score': 0.0
            }
            
            health_scores = []
            total_capacity = 0
            
            for plant in plants:
                plant_id = plant.plant_id
                
                # Get plant metrics
                plant_metrics = self._get_plant_metrics(plant_id)
                
                if plant_metrics:
                    tenant_metrics[tenant_id]['active_plants'] += 1
                    total_capacity += plant.capacity_tpd
                    
                    health_score = plant_metrics.get('health_score', 0.8)
                    health_scores.append(health_score)
                    
                    # Count critical alerts
                    alerts = plant_metrics.get('critical_alerts', 0)
                    tenant_metrics[tenant_id]['critical_alerts'] += alerts
            
            # Calculate averages
            if health_scores:
                tenant_metrics[tenant_id]['avg_health_score'] = sum(health_scores) / len(health_scores)
            tenant_metrics[tenant_id]['total_capacity'] = total_capacity
            
            # Calculate performance score
            tenant_metrics[tenant_id]['performance_score'] = (
                tenant_metrics[tenant_id]['avg_health_score'] * 0.6 +
                (1.0 - min(tenant_metrics[tenant_id]['critical_alerts'] / 10, 1.0)) * 0.4
            )
        
        self.aggregated_metrics['tenant_metrics'] = tenant_metrics
        self.orchestration_state['last_aggregation'] = datetime.now().isoformat()
    
    def _get_plant_metrics(self, plant_id: str) -> Dict:
        """Get current metrics for a specific plant"""
        
        try:
            # Simulate plant metrics (in real system, this would query actual data)
            import random
            
            metrics = {
                'health_score': random.uniform(0.8, 0.95),
                'critical_alerts': random.randint(0, 2),
                'performance_score': random.uniform(0.75, 0.90),
                'last_update': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error getting metrics for {plant_id}: {e}")
            return None
    
    def _perform_cross_plant_analysis(self):
        """Perform cross-plant analysis and benchmarking"""
        
        logger.debug("🔍 Performing cross-plant analysis...")
        
        for tenant_id, plants in self.tenant_plants.items():
            # Get cross-plant benchmarks
            benchmarks = self.manager.get_cross_plant_benchmarks(tenant_id)
            
            if benchmarks:
                self.aggregated_metrics['cross_plant_benchmarks'][tenant_id] = benchmarks
                
                # Identify best and worst performers
                self._identify_performance_patterns(tenant_id, benchmarks)
    
    def _identify_performance_patterns(self, tenant_id: str, benchmarks: Dict):
        """Identify performance patterns across plants"""
        
        try:
            # Analyze energy efficiency patterns
            energy_data = benchmarks.get('energy_efficiency', {})
            if energy_data and 'plants' in energy_data:
                best_energy = energy_data.get('best')
                worst_energy = energy_data.get('worst')
                
                if best_energy and worst_energy:
                    efficiency_gap = best_energy[1]['value'] - worst_energy[1]['value']
                    
                    if efficiency_gap > 0.1:  # Significant gap
                        alert = {
                            'type': 'Performance Gap',
                            'severity': 'Medium',
                            'message': f"Energy efficiency gap of {efficiency_gap:.1%} between {best_energy[1]['plant_name']} and {worst_energy[1]['plant_name']}",
                            'recommendation': 'Consider sharing best practices from high-performing plant',
                            'timestamp': datetime.now().isoformat()
                        }
                        self.cross_plant_alerts.append(alert)
            
        except Exception as e:
            logger.error(f"❌ Error identifying performance patterns: {e}")
    
    def _execute_cross_plant_decisions(self):
        """Execute cross-plant decisions and actions"""
        
        logger.debug("🎯 Executing cross-plant decisions...")
        
        # Check for tenant-wide model retraining triggers
        self._check_model_retraining_triggers()
        
        # Check for resource reallocation needs
        self._check_resource_reallocation()
        
        # Execute tenant-wide deployments
        self._execute_tenant_deployments()
    
    def _check_model_retraining_triggers(self):
        """Check if tenant-wide model retraining is needed"""
        
        for tenant_id, metrics in self.aggregated_metrics.get('tenant_metrics', {}).items():
            performance_score = metrics.get('performance_score', 0.8)
            
            # Trigger retraining if performance drops below threshold
            if performance_score < 0.7:
                logger.warning(f"⚠️ Low performance detected for {tenant_id}: {performance_score:.2f}")
                
                # Trigger tenant-wide model retraining
                self._trigger_tenant_model_retraining(tenant_id)
    
    def _check_resource_reallocation(self):
        """Check if resource reallocation is needed across plants"""
        
        for tenant_id, plants in self.tenant_plants.items():
            # Check if any plant has critical issues
            critical_plants = []
            
            for plant in plants:
                plant_metrics = self._get_plant_metrics(plant.plant_id)
                if plant_metrics and plant_metrics.get('critical_alerts', 0) > 3:
                    critical_plants.append(plant)
            
            # If multiple plants have issues, consider resource reallocation
            if len(critical_plants) > 1:
                logger.warning(f"⚠️ Multiple plants with critical issues in {tenant_id}")
                self._trigger_resource_reallocation(tenant_id, critical_plants)
    
    def _execute_tenant_deployments(self):
        """Execute tenant-wide model deployments"""
        
        # This would integrate with the model deployment system
        # For now, simulate periodic deployments
        pass
    
    def _trigger_tenant_model_retraining(self, tenant_id: str):
        """Trigger model retraining for all plants in a tenant"""
        
        logger.info(f"🔄 Triggering tenant-wide model retraining for {tenant_id}")
        
        plants = self.tenant_plants.get(tenant_id, [])
        
        for plant in plants:
            try:
                # Simulate model retraining trigger (in real system, this would trigger actual retraining)
                logger.info(f"✅ Triggered retraining for {plant.plant_name}: Success")
                    
            except Exception as e:
                logger.error(f"❌ Error triggering retraining for {plant.plant_id}: {e}")
    
    def _trigger_resource_reallocation(self, tenant_id: str, critical_plants: List[PlantConfiguration]):
        """Trigger resource reallocation for critical plants"""
        
        logger.info(f"🔄 Triggering resource reallocation for {tenant_id}")
        
        # This would integrate with resource management systems
        # For now, log the action
        for plant in critical_plants:
            logger.info(f"📋 Resource reallocation needed for {plant.plant_name}")
    
    def _update_tenant_health_score(self):
        """Update overall tenant health score"""
        
        tenant_metrics = self.aggregated_metrics.get('tenant_metrics', {})
        
        if tenant_metrics:
            # Calculate weighted average health score
            total_score = 0
            total_weight = 0
            
            for tenant_id, metrics in tenant_metrics.items():
                weight = metrics.get('total_capacity', 1000)  # Weight by capacity
                score = metrics.get('performance_score', 0.8)
                
                total_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                self.orchestration_state['tenant_health_score'] = total_score / total_weight
    
    def _check_cross_plant_alerts(self, plant_id: str, data: Dict):
        """Check for cross-plant alert conditions"""
        
        try:
            # Check for critical equipment issues
            if data.get('equipment_status') == 'Critical':
                alert = {
                    'type': 'Equipment Critical',
                    'severity': 'High',
                    'plant_id': plant_id,
                    'message': f"Critical equipment issue detected in {plant_id}",
                    'timestamp': datetime.now().isoformat()
                }
                self.cross_plant_alerts.append(alert)
            
            # Check for process anomalies
            if data.get('anomaly_score', 0) > 0.8:
                alert = {
                    'type': 'Process Anomaly',
                    'severity': 'Medium',
                    'plant_id': plant_id,
                    'message': f"Process anomaly detected in {plant_id}",
                    'timestamp': datetime.now().isoformat()
                }
                self.cross_plant_alerts.append(alert)
                
        except Exception as e:
            logger.error(f"❌ Error checking cross-plant alerts: {e}")
    
    def get_supervisor_status(self) -> Dict:
        """Get comprehensive supervisor status"""
        
        return {
            'supervisor_status': {
                'initialized': self.orchestration_state['initialized'],
                'running': self.running,
                'active_plants': self.orchestration_state['active_plants'],
                'tenant_health_score': self.orchestration_state['tenant_health_score'],
                'last_aggregation': self.orchestration_state['last_aggregation']
            },
            'tenant_summary': self.aggregated_metrics.get('tenant_metrics', {}),
            'cross_plant_benchmarks': self.aggregated_metrics.get('cross_plant_benchmarks', {}),
            'active_alerts': len(self.cross_plant_alerts),
            'recent_alerts': self.cross_plant_alerts[-5:] if self.cross_plant_alerts else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def deploy_model_to_tenant(self, tenant_id: str, model_config: Dict) -> Dict:
        """Deploy AI model to all plants in a tenant"""
        
        logger.info(f"🚀 Deploying model to tenant {tenant_id}")
        
        plants = self.tenant_plants.get(tenant_id, [])
        deployment_results = {}
        
        for plant in plants:
            try:
                # Simulate model deployment (in real system, this would deploy to actual agents)
                result = {
                    'plant_id': plant.plant_id,
                    'plant_name': plant.plant_name,
                    'deployment_status': 'Success',
                    'timestamp': datetime.now().isoformat()
                }
                deployment_results[plant.plant_id] = result
                
                logger.info(f"✅ Model deployed to {plant.plant_name}")
                
            except Exception as e:
                logger.error(f"❌ Error deploying model to {plant.plant_id}: {e}")
                deployment_results[plant.plant_id] = {
                    'plant_id': plant.plant_id,
                    'plant_name': plant.plant_name,
                    'deployment_status': f'Failed - {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'tenant_id': tenant_id,
            'deployment_results': deployment_results,
            'total_plants': len(plants),
            'successful_deployments': len([r for r in deployment_results.values() if r['deployment_status'] == 'Success']),
            'timestamp': datetime.now().isoformat()
        }
