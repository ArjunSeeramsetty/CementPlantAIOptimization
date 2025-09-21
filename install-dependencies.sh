#!/bin/bash
# Bash Script for Clean Dependency Installation
# JK Cement Digital Twin Platform - Zero Conflict Setup

echo "🚀 Starting Clean Dependency Installation..."

# Step 1: Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Step 2: Install core packages (force reinstall to avoid conflicts)
echo "🔧 Installing core packages..."
pip install --force-reinstall streamlit==1.28.2
pip install --force-reinstall pandas==1.5.3  
pip install --force-reinstall numpy==1.24.4
pip install --force-reinstall plotly==5.17.0

# Step 3: Install utilities
echo "⚙️ Installing utilities..."
pip install pyyaml==6.0.1 python-dotenv==1.0.0 requests==2.31.0

# Step 4: Install Google Cloud (minimal set)
echo "☁️ Installing Google Cloud packages..."
pip install google-cloud-bigquery==3.11.4
pip install google-cloud-storage==2.10.0
pip install google-generativeai==0.3.2

# Step 5: Install ML packages
echo "🤖 Installing ML packages..."
pip install scikit-learn==1.3.2
pip install scipy==1.10.1

# Step 6: Verify installation
echo "✅ Verifying installation..."
python -c "
try:
    import streamlit; print('✅ Streamlit:', streamlit.__version__)
    import pandas; print('✅ Pandas:', pandas.__version__)
    import numpy; print('✅ NumPy:', numpy.__version__)
    import plotly; print('✅ Plotly:', plotly.__version__)
    import google.cloud.bigquery; print('✅ Google Cloud BigQuery: OK')
    import google.cloud.storage; print('✅ Google Cloud Storage: OK')
    import google.generativeai; print('✅ Google Generative AI: OK')
    import sklearn; print('✅ Scikit-learn:', sklearn.__version__)
    import scipy; print('✅ SciPy:', scipy.__version__)
    print('🚀 All dependencies installed successfully!')
except Exception as e:
    print('❌ Error:', str(e))
    exit(1)
"

echo "🎉 Installation completed!"
