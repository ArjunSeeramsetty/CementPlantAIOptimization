#!/bin/bash
# setup.sh - Local development setup for JK Cement Digital Twin Platform

echo "🚀 Setting up JK Cement Digital Twin Platform..."

# Remove existing environment if it exists
if [ -d ".venv" ]; then
    echo "🗑️ Removing existing virtual environment..."
    rm -rf .venv
fi

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
python -m venv .venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip and tools
echo "⬆️ Upgrading pip and tools..."
python -m pip install --upgrade pip==23.3.2 setuptools==69.0.3 wheel==0.42.0

# Install dependencies in order to avoid conflicts
echo "📚 Installing dependencies..."

echo "  📦 Core packages..."
pip install typing-extensions>=4.3.0 toml>=0.10.2 packaging==23.2 tenacity==8.5.0

echo "  🔢 Data packages..."
pip install numpy==1.24.4 pandas==2.0.3

echo "  🎨 UI packages..."
pip install streamlit==1.28.2 plotly==5.17.0

echo "  🔧 Utility packages..."
pip install pyyaml==6.0.1 python-dotenv==1.0.0 requests==2.30.0

echo "  ☁️ Google Cloud packages..."
pip install google-cloud-bigquery==3.14.1 google-cloud-storage==2.14.0 google-cloud-firestore==2.13.1

echo "  🤖 ML packages..."
pip install scikit-learn==1.3.2 scipy==1.11.4

echo "  🛠️ Dev packages..."
pip install pytest==7.4.3 black==23.12.1 flake8==6.1.0 mypy==1.8.0

echo "  🔒 Security packages..."
pip install bandit==1.7.5 safety==2.3.5

echo "  📊 Additional packages..."
pip install openpyxl==3.1.2 xlsxwriter==3.1.9 Pillow==10.1.0

# Verify installation
echo "✅ Verifying installation..."
python -c "
import streamlit, pandas, numpy, plotly
import google.cloud.bigquery
print('🎉 All packages installed successfully!')
print(f'Streamlit: {streamlit.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
"

echo "🎯 Setup completed successfully!"
echo "Run: streamlit run main.py"
