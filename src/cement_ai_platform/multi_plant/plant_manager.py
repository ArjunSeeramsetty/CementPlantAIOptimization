# NEW FILE: src/cement_ai_platform/multi_plant/plant_manager.py
import yaml
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from google.cloud import firestore
from google.cloud import storage
from google.cloud import bigquery

@dataclass
class PlantConfiguration:
    """Configuration for a single cement plant"""
    plant_id: str
    plant_name: str
    location: str
    capacity_tpd: float
    kiln_type: str
    raw_materials: Dict[str, float]
    fuel_mix: Dict[str, float]
    energy: Dict[str, float]
    process: Dict[str, any]
    quality: Dict[str, float]
    environmental: Dict[str, float]
    dcs_tags: Dict[str, any]
    tenant_id: str
    region: str
    timezone: str
    
class MultiPlantManager:
    """
    Multi-plant support system with dynamic configuration loading
    and tenant isolation for scalable cement plant management
    """
    
    def __init__(self, project_id: str = "cement-ai-opt-38517"):
        self.project_id = project_id
        
        try:
            self.firestore_client = firestore.Client(project=project_id)
            self.storage_client = storage.Client(project=project_id)
            self.bq_client = bigquery.Client(project=project_id)
            self.cloud_available = True
        except Exception as e:
            print(f"⚠️ Google Cloud services not available: {e}")
            self.firestore_client = None
            self.storage_client = None
            self.bq_client = None
            self.cloud_available = False
        
        self.bucket_name = f"{project_id}-plant-configs"
        
        # Plant registry cache
        self.plant_registry = {}
        self.tenant_isolation = {}
        
        # Initialize multi-plant infrastructure
        self._initialize_multi_plant_system()
    
    def _initialize_multi_plant_system(self):
        """Initialize multi-plant support infrastructure"""
        
        if self.cloud_available:
            # Create Cloud Storage bucket for plant configurations
            try:
                bucket = self.storage_client.create_bucket(self.bucket_name)
                print(f"✅ Created plant config bucket: {self.bucket_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"ℹ️ Plant config bucket already exists: {self.bucket_name}")
                else:
                    print(f"❌ Error creating bucket: {e}")
            
            # Initialize Firestore collections for multi-plant data
            self._setup_firestore_collections()
        
        # Load default plant configurations
        self._load_default_plant_configs()
    
    def _setup_firestore_collections(self):
        """Setup Firestore collections for multi-plant data"""
        
        collections = [
            'plant_registry',
            'tenant_configurations', 
            'plant_data_streams',
            'multi_plant_analytics',
            'cross_plant_benchmarks'
        ]
        
        for collection in collections:
            # Create a dummy document to initialize collection
            doc_ref = self.firestore_client.collection(collection).document('_init')
            doc_ref.set({'initialized': True, 'timestamp': firestore.SERVER_TIMESTAMP})
            
            print(f"✅ Initialized Firestore collection: {collection}")
    
    def _load_default_plant_configs(self):
        """Load default plant configurations for demo"""
        
        default_plants = [
            {
                'plant_id': 'jk_cement_rajasthan_1',
                'plant_name': 'JK Cement Rajasthan Plant 1',
                'location': 'Rajasthan, India',
                'capacity_tpd': 4000,
                'tenant_id': 'jk_cement',
                'region': 'asia-south1',
                'timezone': 'Asia/Kolkata'
            },
            {
                'plant_id': 'jk_cement_mp_1', 
                'plant_name': 'JK Cement Madhya Pradesh Plant 1',
                'location': 'Madhya Pradesh, India',
                'capacity_tpd': 6000,
                'tenant_id': 'jk_cement',
                'region': 'asia-south1',
                'timezone': 'Asia/Kolkata'
            },
            {
                'plant_id': 'ultratech_gujarat_1',
                'plant_name': 'UltraTech Gujarat Plant 1', 
                'location': 'Gujarat, India',
                'capacity_tpd': 10000,
                'tenant_id': 'ultratech',
                'region': 'asia-south1',
                'timezone': 'Asia/Kolkata'
            },
            {
                'plant_id': 'acc_karnataka_1',
                'plant_name': 'ACC Karnataka Plant 1',
                'location': 'Karnataka, India', 
                'capacity_tpd': 8000,
                'tenant_id': 'acc_limited',
                'region': 'asia-south1',
                'timezone': 'Asia/Kolkata'
            },
            {
                'plant_id': 'jk_cement_uae_1',
                'plant_name': 'JK Cement UAE Plant 1',
                'location': 'UAE',
                'capacity_tpd': 3500,
                'tenant_id': 'jk_cement',
                'region': 'europe-west1',
                'timezone': 'Asia/Dubai'
            }
        ]
        
        for plant_data in default_plants:
            config = self._create_plant_config_from_template(plant_data)
            self.register_plant(config)
    
    def _create_plant_config_from_template(self, plant_data: Dict) -> PlantConfiguration:
        """Create plant configuration from template"""
        
        # Load base template
        base_config_path = Path("config/plant_config.yml")
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
        else:
            base_config = self._get_default_config_template()
        
        # Customize based on plant specifics
        capacity_factor = plant_data['capacity_tpd'] / 4000  # Base capacity
        
        # Scale parameters based on capacity
        raw_materials = base_config.get('raw_materials', {})
        fuel_mix = base_config.get('fuel_mix', {})
        
        # Regional adjustments
        if 'UAE' in plant_data['location']:
            # UAE plants typically have different fuel mix
            fuel_mix['natural_gas'] = fuel_mix.get('coal', 120) * 0.3
            fuel_mix['coal'] = fuel_mix.get('coal', 120) * 0.7
        
        return PlantConfiguration(
            plant_id=plant_data['plant_id'],
            plant_name=plant_data['plant_name'],
            location=plant_data['location'],
            capacity_tpd=plant_data['capacity_tpd'],
            kiln_type=base_config.get('plant', {}).get('kiln_type', 'Dry Process'),
            raw_materials=raw_materials,
            fuel_mix=fuel_mix,
            energy=base_config.get('energy', {}),
            process=base_config.get('process', {}),
            quality=base_config.get('quality', {}),
            environmental=base_config.get('environmental', {}),
            dcs_tags=base_config.get('dcs_tags', {}),
            tenant_id=plant_data['tenant_id'],
            region=plant_data['region'],
            timezone=plant_data['timezone']
        )
    
    def _get_default_config_template(self) -> Dict:
        """Get default configuration template"""
        
        return {
            'plant': {'kiln_type': 'Dry Process'},
            'raw_materials': {'limestone': 1200, 'clay': 200, 'iron_ore': 50, 'gypsum': 50},
            'fuel_mix': {'coal': 120, 'petcoke': 80, 'alternative_fuels': 20},
            'energy': {'electrical': 95, 'thermal': 3200},
            'process': {
                'kiln_temperature_c': 1450,
                'kiln_speed_rpm': 3.5,
                'preheater_stages': 5,
                'cooler_type': 'Grate Cooler'
            },
            'quality': {
                'free_lime_pct': 1.5,
                'c3s_content_pct': 60,
                'c2s_content_pct': 15,
                'compressive_strength_28d_mpa': 45
            },
            'environmental': {
                'nox_mg_nm3': 500,
                'so2_mg_nm3': 200,
                'dust_mg_nm3': 30,
                'co2_kg_per_ton': 800
            },
            'dcs_tags': {
                'update_frequencies': {
                    'critical_loops': 1,
                    'process_variables': 5,
                    'quality_lab': 3600
                }
            }
        }
    
    def register_plant(self, config: PlantConfiguration) -> bool:
        """Register a new plant in the multi-plant system"""
        
        try:
            if self.cloud_available:
                # Store in Firestore
                doc_ref = self.firestore_client.collection('plant_registry').document(config.plant_id)
                doc_ref.set(asdict(config))
                
                # Store configuration in Cloud Storage
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"{config.plant_id}/plant_config.json")
                blob.upload_from_string(json.dumps(asdict(config), indent=2))
            
            # Update local cache
            self.plant_registry[config.plant_id] = config
            
            # Setup tenant isolation
            if config.tenant_id not in self.tenant_isolation:
                self.tenant_isolation[config.tenant_id] = []
            self.tenant_isolation[config.tenant_id].append(config.plant_id)
            
            print(f"✅ Registered plant: {config.plant_name} ({config.plant_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error registering plant {config.plant_id}: {e}")
            return False
    
    def get_plant_config(self, plant_id: str) -> Optional[PlantConfiguration]:
        """Get plant configuration by ID"""
        
        # Check cache first
        if plant_id in self.plant_registry:
            return self.plant_registry[plant_id]
        
        # Load from Firestore if available
        if self.cloud_available:
            try:
                doc_ref = self.firestore_client.collection('plant_registry').document(plant_id)
                doc = doc_ref.get()
                
                if doc.exists:
                    config_data = doc.to_dict()
                    config = PlantConfiguration(**config_data)
                    self.plant_registry[plant_id] = config
                    return config
                else:
                    print(f"❌ Plant not found: {plant_id}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error loading plant config {plant_id}: {e}")
                return None
        
        return None
    
    def get_tenant_plants(self, tenant_id: str) -> List[PlantConfiguration]:
        """Get all plants for a specific tenant"""
        
        if self.cloud_available:
            try:
                # Query Firestore for tenant plants
                plants_ref = self.firestore_client.collection('plant_registry')
                query = plants_ref.where('tenant_id', '==', tenant_id)
                
                tenant_plants = []
                for doc in query.stream():
                    config_data = doc.to_dict()
                    config = PlantConfiguration(**config_data)
                    tenant_plants.append(config)
                
                return tenant_plants
                
            except Exception as e:
                print(f"❌ Error loading tenant plants for {tenant_id}: {e}")
                return []
        else:
            # Return cached plants for demo
            return [config for config in self.plant_registry.values() if config.tenant_id == tenant_id]
    
    def create_tenant_isolated_dataset(self, tenant_id: str) -> str:
        """Create tenant-isolated BigQuery dataset"""
        
        if not self.cloud_available:
            return f"cement_analytics_{tenant_id.replace('-', '_')}"
        
        dataset_id = f"cement_analytics_{tenant_id.replace('-', '_')}"
        
        # Create dataset with tenant isolation
        dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
        dataset.location = "US"
        dataset.description = f"Tenant-isolated dataset for {tenant_id}"
        
        try:
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            print(f"✅ Created tenant dataset: {dataset_id}")
            return dataset_id
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️ Tenant dataset already exists: {dataset_id}")
                return dataset_id
            else:
                print(f"❌ Error creating tenant dataset: {e}")
                return None
    
    def get_cross_plant_benchmarks(self, tenant_id: str) -> Dict:
        """Get cross-plant performance benchmarks for a tenant"""
        
        tenant_plants = self.get_tenant_plants(tenant_id)
        
        if not tenant_plants:
            return {}
        
        # Calculate benchmarks across plants
        benchmarks = {
            'energy_efficiency': {
                'best': None,
                'worst': None,
                'average': 0,
                'plants': {}
            },
            'quality_consistency': {
                'best': None,
                'worst': None,
                'average': 0,
                'plants': {}
            },
            'capacity_utilization': {
                'best': None,
                'worst': None,
                'average': 0,
                'plants': {}
            }
        }
        
        # Simulate benchmark data (in real system, this would query actual performance data)
        import random
        
        for plant in tenant_plants:
            # Simulate performance metrics
            energy_eff = random.uniform(0.85, 0.95)
            quality_cons = random.uniform(0.90, 0.98)
            capacity_util = random.uniform(0.85, 0.95)
            
            benchmarks['energy_efficiency']['plants'][plant.plant_id] = {
                'value': energy_eff,
                'plant_name': plant.plant_name
            }
            benchmarks['quality_consistency']['plants'][plant.plant_id] = {
                'value': quality_cons,
                'plant_name': plant.plant_name
            }
            benchmarks['capacity_utilization']['plants'][plant.plant_id] = {
                'value': capacity_util,
                'plant_name': plant.plant_name
            }
        
        # Calculate best, worst, average
        for metric in benchmarks:
            values = [p['value'] for p in benchmarks[metric]['plants'].values()]
            benchmarks[metric]['average'] = sum(values) / len(values)
            
            best_plant = max(benchmarks[metric]['plants'].items(), key=lambda x: x[1]['value'])
            worst_plant = min(benchmarks[metric]['plants'].items(), key=lambda x: x[1]['value'])
            
            benchmarks[metric]['best'] = best_plant
            benchmarks[metric]['worst'] = worst_plant
        
        return benchmarks
