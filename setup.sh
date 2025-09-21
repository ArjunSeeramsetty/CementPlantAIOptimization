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

# Install all dependencies from consolidated requirements
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

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
