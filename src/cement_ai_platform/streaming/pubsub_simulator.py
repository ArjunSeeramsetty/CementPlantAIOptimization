import json
import time
import threading
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1 import SubscriberClient
from typing import Dict, Callable, Any
import pandas as pd
import numpy as np
import random
from datetime import datetime

# Import centralized logging and retry mechanisms
from ..config.logging_config import get_logger
from ..utils.retry_decorator import retry_gcp_operation
from ..config.otel_tracer import trace_gcp_operation

logger = get_logger(__name__)

class CementPlantPubSubSimulator:
    """
    Simulates real-time sensor data streaming using Google Cloud Pub/Sub.
    
    This class provides a comprehensive simulation of cement plant sensor data
    streaming through Google Cloud Pub/Sub, including process variables, quality data,
    energy consumption, emissions, and equipment health metrics.
    
    Attributes:
        project_id (str): GCP project ID for Pub/Sub operations
        publisher (pubsub_v1.PublisherClient): Pub/Sub publisher client
        subscriber (pubsub_v1.SubscriberClient): Pub/Sub subscriber client
        topics (dict): Mapping of topic names to topic paths
        streaming (bool): Flag indicating if streaming is active
        
    Methods:
        start_streaming_simulation(interval_seconds: int) -> threading.Thread:
            Starts background thread publishing synthetic sensor data
        subscribe_to_stream(topic_name: str, callback: Callable) -> Future:
            Subscribes to a Pub/Sub topic and dispatches messages to callback
        stop_streaming() -> None:
            Stops the streaming simulation
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = SubscriberClient()
        self.streaming = False
        
        # Define sensor topics
        self.topics = {
            "process-variables": f"projects/{project_id}/topics/cement-process-variables",
            "quality-data": f"projects/{project_id}/topics/cement-quality-data",
            "energy-consumption": f"projects/{project_id}/topics/cement-energy-consumption",
            "emissions-data": f"projects/{project_id}/topics/cement-emissions-data",
            "equipment-health": f"projects/{project_id}/topics/cement-equipment-health"
        }
        
        # Create topics if they don't exist
        self._create_topics()
    
    @retry_gcp_operation()
    def _create_topics(self):
        """Create Pub/Sub topics for cement plant data streams"""
        for topic_name, topic_path in self.topics.items():
            try:
                self.publisher.create_topic(request={"name": topic_path})
                logger.info(f"SUCCESS: Created topic: {topic_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"INFO: Topic already exists: {topic_name}")
                else:
                    logger.error(f"ERROR: Error creating topic {topic_name}: {e}")
    
    def start_streaming_simulation(self, interval_seconds: int = 2):
        """Start simulating real-time sensor data streaming"""
        self.streaming = True
        
        def stream_data():
            while self.streaming:
                try:
                    # Generate realistic sensor data
                    sensor_data = self._generate_sensor_snapshot()
                    
                    # Publish to different topics
                    self._publish_process_data(sensor_data['process'])
                    self._publish_quality_data(sensor_data['quality'])
                    self._publish_energy_data(sensor_data['energy'])
                    self._publish_emissions_data(sensor_data['emissions'])
                    self._publish_equipment_health(sensor_data['equipment'])
                    
                    print(f"📡 Streaming data at {time.strftime('%H:%M:%S')}: "
                          f"Free Lime: {sensor_data['process']['free_lime_percent']:.2f}%, "
                          f"Kiln Temp: {sensor_data['process']['burning_zone_temp_c']:.0f}°C")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"❌ Streaming error: {e}")
                    time.sleep(5)
        
        # Start streaming in background thread
        streaming_thread = threading.Thread(target=stream_data, daemon=True)
        streaming_thread.start()
        
        print(f"🚀 Started real-time streaming simulation (interval: {interval_seconds}s)")
        return streaming_thread
    
    def _generate_sensor_snapshot(self) -> Dict[str, Dict]:
        """Generate realistic sensor data snapshot"""
        
        # Base process variables
        process_data = {
            'feed_rate_tph': random.uniform(150, 185),
            'fuel_rate_tph': random.uniform(14.5, 18.2),
            'kiln_speed_rpm': random.uniform(2.9, 4.0),
            'burning_zone_temp_c': random.uniform(1430, 1470),
            'free_lime_percent': random.uniform(0.8, 2.3),
            'preheater_stage1_temp_c': random.uniform(880, 950),
            'preheater_stage2_temp_c': random.uniform(820, 890),
            'cooler_outlet_temp_c': random.uniform(85, 125),
            'o2_percent': random.uniform(2.5, 4.2),
            'co_mg_nm3': random.uniform(50, 180),
            'kiln_torque_percent': random.uniform(65, 85)
        }
        
        # Quality parameters
        quality_data = {
            'compressive_strength_28d_mpa': random.uniform(40, 52),
            'blaine_fineness_cm2_g': random.uniform(3100, 3800),
            'c3s_content_percent': random.uniform(55, 68),
            'residue_45_micron_percent': random.uniform(8, 15)
        }
        
        # Energy consumption
        energy_data = {
            'thermal_energy_kcal_kg': random.uniform(680, 750),
            'electrical_energy_kwh_t': random.uniform(68, 82),
            'coal_consumption_kg_t': random.uniform(95, 115),
            'specific_power_consumption_kwh_t': random.uniform(28, 35)
        }
        
        # Emissions data
        emissions_data = {
            'nox_mg_nm3': random.uniform(400, 650),
            'so2_mg_nm3': random.uniform(120, 250),
            'dust_mg_nm3': random.uniform(15, 45),
            'co2_kg_per_ton': random.uniform(780, 850)
        }
        
        # Equipment health
        equipment_data = {
            'raw_mill_vibration_mm_s': random.uniform(2.5, 7.8),
            'kiln_vibration_mm_s': random.uniform(3.2, 8.5),
            'cement_mill_power_kw': random.uniform(3200, 4100),
            'id_fan_current_a': random.uniform(180, 220),
            'cooler_grate_speed_rpm': random.uniform(8, 14)
        }
        
        # Add realistic variations and anomalies (5% chance)
        if random.random() < 0.05:
            anomaly_type = random.choice(['high_free_lime', 'low_temp', 'high_vibration'])
            
            if anomaly_type == 'high_free_lime':
                process_data['free_lime_percent'] = random.uniform(2.1, 3.2)
                process_data['fuel_rate_tph'] *= 0.92  # Lower fuel causing high free lime
            elif anomaly_type == 'low_temp':
                process_data['burning_zone_temp_c'] = random.uniform(1380, 1420)
            elif anomaly_type == 'high_vibration':
                equipment_data['kiln_vibration_mm_s'] = random.uniform(8.5, 12.0)
        
        return {
            'process': {**process_data, 'timestamp': time.time()},
            'quality': {**quality_data, 'timestamp': time.time()},
            'energy': {**energy_data, 'timestamp': time.time()},
            'emissions': {**emissions_data, 'timestamp': time.time()},
            'equipment': {**equipment_data, 'timestamp': time.time()}
        }
    
    def _publish_process_data(self, data: Dict):
        """Publish process variables to Pub/Sub"""
        message_data = json.dumps({
            **data,
            'data_type': 'process_variables',
            'plant_id': 'jk_cement_demo_plant'
        }).encode('utf-8')
        
        try:
            future = self.publisher.publish(self.topics['process-variables'], message_data)
            return future.result()
        except Exception as e:
            print(f"❌ Error publishing process data: {e}")
    
    def _publish_quality_data(self, data: Dict):
        """Publish quality data to Pub/Sub"""
        message_data = json.dumps({
            **data,
            'data_type': 'quality_parameters',
            'plant_id': 'jk_cement_demo_plant'
        }).encode('utf-8')
        
        try:
            future = self.publisher.publish(self.topics['quality-data'], message_data)
            return future.result()
        except Exception as e:
            print(f"❌ Error publishing quality data: {e}")
    
    def _publish_energy_data(self, data: Dict):
        """Publish energy consumption data to Pub/Sub"""
        message_data = json.dumps({
            **data,
            'data_type': 'energy_consumption',
            'plant_id': 'jk_cement_demo_plant'
        }).encode('utf-8')
        
        try:
            future = self.publisher.publish(self.topics['energy-consumption'], message_data)
            return future.result()
        except Exception as e:
            print(f"❌ Error publishing energy data: {e}")
    
    def _publish_emissions_data(self, data: Dict):
        """Publish emissions data to Pub/Sub"""
        message_data = json.dumps({
            **data,
            'data_type': 'emissions_data',
            'plant_id': 'jk_cement_demo_plant'
        }).encode('utf-8')
        
        try:
            future = self.publisher.publish(self.topics['emissions-data'], message_data)
            return future.result()
        except Exception as e:
            print(f"❌ Error publishing emissions data: {e}")
    
    def _publish_equipment_health(self, data: Dict):
        """Publish equipment health data to Pub/Sub"""
        message_data = json.dumps({
            **data,
            'data_type': 'equipment_health',
            'plant_id': 'jk_cement_demo_plant'
        }).encode('utf-8')
        
        try:
            future = self.publisher.publish(self.topics['equipment-health'], message_data)
            return future.result()
        except Exception as e:
            print(f"❌ Error publishing equipment health data: {e}")
    
    def subscribe_to_stream(self, topic_name: str, callback: Callable[[Dict], None]):
        """Subscribe to a data stream and process messages with callback"""
        if topic_name not in self.topics:
            raise ValueError(f"Unknown topic: {topic_name}")
        
        subscription_path = f"projects/{self.project_id}/subscriptions/{topic_name}-subscription"
        
        # Create subscription if it doesn't exist
        try:
            self.subscriber.create_subscription(
                request={
                    "name": subscription_path,
                    "topic": self.topics[topic_name]
                }
            )
            print(f"✅ Created subscription: {topic_name}-subscription")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Subscription already exists: {topic_name}-subscription")
            else:
                print(f"❌ Error creating subscription: {e}")
        
        def message_handler(message):
            try:
                data = json.loads(message.data.decode('utf-8'))
                callback(data)
                message.ack()
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                message.nack()
        
        # Start listening
        streaming_pull_future = self.subscriber.subscribe(subscription_path, callback=message_handler)
        print(f"📥 Listening for messages on {topic_name}...")
        
        return streaming_pull_future
    
    def stop_streaming(self):
        """Stop the streaming simulation"""
        self.streaming = False
        print("⏹️ Stopped streaming simulation")

# Real-time data processor
class RealTimeDataProcessor:
    """Process real-time streaming data and trigger AI agent responses"""
    
    def __init__(self, plant_id: str = "default"):
        self.plant_id = plant_id
        self.agents = self._initialize_agents()
        self.alert_thresholds = {
            'free_lime_high': 2.0,
            'temperature_low': 1420,
            'temperature_high': 1480,
            'vibration_high': 8.0
        }
    
    def _initialize_agents(self):
        """Initialize AI agents for real-time response"""
        try:
            from cement_ai_platform.agents.unified_kiln_cooler_controller import UnifiedKilnCoolerController
            from cement_ai_platform.agents.plant_anomaly_detector import PlantAnomalyDetector
            
            return {
                'controller': UnifiedKilnCoolerController(),
                'anomaly_detector': PlantAnomalyDetector()
            }
        except ImportError as e:
            print(f"⚠️ Agent import warning: {e}")
            return {'controller': None, 'anomaly_detector': None}
    
    def process_process_variables(self, data: Dict):
        """Process real-time process variables and trigger control actions"""
        
        # Add plant context to data
        data['plant_id'] = self.plant_id
        
        # Check for critical conditions
        free_lime = data.get('free_lime_percent', 1.0)
        burning_zone_temp = data.get('burning_zone_temp_c', 1450)
        
        if free_lime > self.alert_thresholds['free_lime_high']:
            print(f"🚨 CRITICAL ALERT: Free lime high ({free_lime:.2f}%) - Triggering controller")
            
            if self.agents['controller']:
                # Get control recommendations
                control_response = self.agents['controller'].compute_unified_setpoints(data)
                
                print(f"🤖 AI Controller Response:")
                print(f"   Recommended fuel rate: {control_response['setpoints']['fuel_rate_tph']:.2f} t/h")
                print(f"   Recommended kiln speed: {control_response['setpoints']['kiln_speed_rpm']:.2f} rpm")
                print(f"   Predicted correction time: {control_response['performance_prediction']['time_to_stabilize']} minutes")
        
        if burning_zone_temp < self.alert_thresholds['temperature_low']:
            print(f"⚠️ WARNING: Burning zone temperature low ({burning_zone_temp:.0f}°C)")
        
        # Always run anomaly detection if available
        if self.agents['anomaly_detector']:
            anomaly_results = self.agents['anomaly_detector'].detect_anomalies(data)
            if anomaly_results['overall_anomaly_score'] > 0.3:
                print(f"🔍 Anomalies detected (score: {anomaly_results['overall_anomaly_score']:.2f})")
    
    def process_equipment_health(self, data: Dict):
        """Process equipment health data and trigger maintenance alerts"""
        
        # Add plant context to data
        data['plant_id'] = self.plant_id
        
        for key, value in data.items():
            if 'vibration' in key and value > self.alert_thresholds['vibration_high']:
                equipment_name = key.replace('_vibration_mm_s', '')
                print(f"🔧 MAINTENANCE ALERT: {equipment_name} high vibration ({value:.1f} mm/s)")
