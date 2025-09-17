# JK Cement Digital Twin Platform - Agent Modules

This directory contains the five critical agent modules that implement all JK Cement requirements for the Digital Twin platform.

## 🚀 **Agent Modules Overview**

### 1. **Alternative Fuel Optimizer** (`alternative_fuel_optimizer.py`)
- **Purpose**: Optimizes alternative fuel blend to maximize TSR (Thermal Substitution Rate)
- **Key Features**:
  - Multi-fuel optimization (coal, petcoke, RDF, biomass, tire chips)
  - Quality constraint handling (chlorine, sulfur, alkali limits)
  - Cost-benefit analysis
  - RDF scenario generation
- **Target**: Achieve 15%+ TSR while maintaining clinker quality

### 2. **Cement Plant GPT Interface** (`cement_plant_gpt.py`)
- **Purpose**: Natural language interface for plant operations and troubleshooting
- **Key Features**:
  - Domain-specific cement plant knowledge base
  - Troubleshooting guidance
  - Shift report generation
  - Quality trend analysis
  - Fallback mode when Google AI unavailable
- **Target**: Provide expert-level plant guidance through natural language

### 3. **Unified Kiln-Cooler Controller** (`unified_kiln_cooler_controller.py`)
- **Purpose**: Coordinated control of kiln and cooler systems
- **Key Features**:
  - PID controllers with anti-windup and rate limiting
  - Thermal coupling between kiln and cooler
  - Safety limits and operational constraints
  - Performance prediction
  - Auto-tuning capabilities
- **Target**: Optimize process control for quality and efficiency

### 4. **Utility Optimizer** (`utility_optimizer.py`)
- **Purpose**: Optimize utility systems (compressed air, water, material handling)
- **Key Features**:
  - Compressed air system efficiency analysis
  - Water consumption optimization
  - Material handling system optimization
  - Cost savings calculation
  - Maintenance recommendations
- **Target**: Reduce utility costs by 15-25% annually

### 5. **Plant Anomaly Detector** (`plant_anomaly_detector.py`)
- **Purpose**: Real-time anomaly detection and equipment health monitoring
- **Key Features**:
  - Multi-parameter equipment health assessment
  - Statistical and ML-based anomaly detection
  - Failure prediction and maintenance scheduling
  - Alert generation and prioritization
  - Historical trend analysis
- **Target**: Prevent equipment failures and optimize maintenance

## 🔧 **Unified Platform Integration** (`jk_cement_platform.py`)

The `JKCementDigitalTwinPlatform` class orchestrates all five agents:

```python
from cement_ai_platform.agents.jk_cement_platform import create_unified_platform

# Create platform instance
platform = create_unified_platform()

# Process plant data through all agents
results = platform.process_plant_data(plant_data)

# Access individual agent results
fuel_optimization = results['fuel_optimization']
control_setpoints = results['control_setpoints']
utility_optimization = results['utility_optimization']
anomaly_detection = results['anomaly_detection']
gpt_analysis = results['gpt_analysis']
```

## 📊 **Key Performance Indicators**

The platform tracks these KPIs:
- **TSR Achievement**: Target 15%+ alternative fuel substitution
- **Plant Health Score**: Equipment and process health (0-1 scale)
- **Control Health Score**: Process control performance (0-1 scale)
- **Utility Savings**: Annual cost savings potential
- **Overall Performance Score**: Combined platform performance

## 🛠 **Installation and Dependencies**

### Core Dependencies (Required)
```bash
pip install numpy pandas scipy scikit-learn pyyaml matplotlib seaborn
```

### Optional Dependencies (Enhanced Functionality)
```bash
# For GPT interface
pip install google-generativeai

# For advanced ML models
pip install tensorflow torch

# For cloud integration
pip install google-cloud-bigquery google-cloud-storage
```

## 🧪 **Testing**

Run the comprehensive test suite:
```bash
python scripts/test_jk_cement_platform.py
```

This tests:
- Individual agent functionality
- Unified platform processing
- Performance tracking
- Export functionality
- Anomaly model training

## 📈 **Expected Benefits**

Based on JK Cement requirements:
- **Alternative Fuel Optimization**: 15%+ TSR achievement
- **Process Control**: 5-10% improvement in process stability
- **Utility Optimization**: $100K-500K annual savings
- **Anomaly Detection**: 20-30% reduction in unplanned downtime
- **GPT Interface**: Improved operator decision-making

## 🔄 **Data Flow**

```
Plant Data → Unified Platform → Agent Processing → Results
     ↓              ↓                    ↓           ↓
Sensor Data → Data Validation → Individual → Comprehensive
KPIs        → Context Building → Analysis  → Recommendations
```

## 🚨 **Fallback Modes**

The platform includes robust fallback mechanisms:
- **GPT Interface**: Mock responses when Google AI unavailable
- **Process Models**: Simplified models when advanced models unavailable
- **Anomaly Detection**: Rule-based detection when ML models unavailable

## 📝 **Configuration**

Configure the platform using `config/plant_config.yml`:
- Plant parameters and limits
- DCS tag definitions
- Quality targets
- Environmental limits
- Equipment specifications

## 🔍 **Monitoring and Logging**

The platform provides comprehensive logging:
- Agent execution status
- Performance metrics
- Error handling
- Optimization history
- Alert generation

## 🎯 **JK Cement Requirements Coverage**

✅ **Alternative Fuel Optimization**: Implemented with TSR optimization
✅ **Cement Plant GPT Interface**: Implemented with domain expertise
✅ **Unified Kiln-Cooler Controller**: Implemented with PID control
✅ **Utility Optimization**: Implemented for all utility systems
✅ **Plant Anomaly Detection**: Implemented with ML-based monitoring

## 📞 **Support**

For issues or questions:
1. Check the test suite results
2. Review the configuration files
3. Examine the logging output
4. Consult the individual agent documentation

The platform is designed to be production-ready and scalable for JK Cement's 6-month POC implementation.
