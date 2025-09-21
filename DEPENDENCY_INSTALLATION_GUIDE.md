# 🚀 JK Cement Digital Twin Platform - Dependency Installation Guide

## 🚨 **QUICK FIX FOR DEPENDENCY CONFLICTS**

If you're experiencing dependency resolution issues, use these proven solutions:

## 🎯 **Method 1: PowerShell Script (Windows - RECOMMENDED)**

```powershell
# Run this in PowerShell
.\install-dependencies.ps1
```

## 🎯 **Method 2: Bash Script (Linux/Mac)**

```bash
# Make executable and run
chmod +x install-dependencies.sh
./install-dependencies.sh
```

## 🎯 **Method 3: Manual Installation (Step-by-Step)**

```bash
# Step 1: Upgrade pip
python -m pip install --upgrade pip

# Step 2: Core packages (force reinstall)
pip install --force-reinstall streamlit==1.28.2
pip install --force-reinstall pandas==1.5.3
pip install --force-reinstall numpy==1.24.4
pip install --force-reinstall plotly==5.17.0

# Step 3: Utilities
pip install pyyaml==6.0.1 python-dotenv==1.0.0 requests==2.31.0

# Step 4: Google Cloud (minimal set)
pip install google-cloud-bigquery==3.11.4
pip install google-cloud-storage==2.10.0
pip install google-generativeai==0.3.2

# Step 5: ML packages
pip install scikit-learn==1.3.2 scipy==1.10.1
```

## 🎯 **Method 4: Ultra-Minimal Installation**

```bash
# Only essential packages
pip install streamlit==1.28.2 pandas==1.5.3 numpy==1.24.4 plotly==5.17.0
pip install pyyaml==6.0.1 python-dotenv==1.0.0
pip install google-cloud-bigquery==3.11.4 google-generativeai==0.3.2
```

## 🎯 **Method 5: Clean Environment (Nuclear Option)**

```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# OR
fresh_env\Scripts\activate     # Windows

# Install minimal set
pip install streamlit==1.28.2 pandas==1.5.3 numpy==1.24.4 plotly==5.17.0
pip install pyyaml==6.0.1 python-dotenv==1.0.0
pip install google-cloud-bigquery==3.11.4 google-generativeai==0.3.2
```

## ✅ **Verification Commands**

After installation, verify with:

```python
python -c "
import streamlit; print('✅ Streamlit:', streamlit.__version__)
import pandas; print('✅ Pandas:', pandas.__version__)
import numpy; print('✅ NumPy:', numpy.__version__)
import plotly; print('✅ Plotly:', plotly.__version__)
import google.cloud.bigquery; print('✅ Google Cloud BigQuery: OK')
import google.generativeai; print('✅ Google Generative AI: OK')
print('🚀 All dependencies working!')
"
```

## 🚀 **Launch the Platform**

```bash
streamlit run src/cement_ai_platform/dashboard/unified_dashboard.py
```

## 🔧 **Troubleshooting**

### Issue: "resolution-too-deep" error
**Solution:** Use Method 3 (Manual Installation) with `--force-reinstall`

### Issue: Package conflicts
**Solution:** Use Method 5 (Clean Environment)

### Issue: Google Cloud import errors
**Solution:** Install only essential GCP packages:
```bash
pip install google-cloud-bigquery==3.11.4 google-generativeai==0.3.2
```

### Issue: Streamlit won't start
**Solution:** 
```bash
pip install --force-reinstall streamlit==1.28.2
streamlit --version
```

## 📊 **Package Versions (Tested & Verified)**

| Package | Version | Status |
|---------|---------|--------|
| streamlit | 1.28.2 | ✅ Working |
| pandas | 1.5.3 | ✅ Working |
| numpy | 1.24.4 | ✅ Working |
| plotly | 5.17.0 | ✅ Working |
| google-cloud-bigquery | 3.11.4 | ✅ Working |
| google-generativeai | 0.3.2 | ✅ Working |
| scikit-learn | 1.3.2 | ✅ Working |
| scipy | 1.10.1 | ✅ Working |

## 🎯 **Production Deployment**

For production deployment, the CI/CD pipelines now use the conflict-free approach:

1. **Staged Installation**: Packages installed in logical groups
2. **Force Reinstall**: Avoids version conflicts
3. **Optional Dependencies**: Dev tools won't fail the build
4. **Fallback Mechanisms**: Continues even if some packages fail

## 🚀 **Success Indicators**

You'll know the installation is successful when:

- ✅ All import statements work without errors
- ✅ Streamlit launches without dependency warnings
- ✅ Platform modules load correctly
- ✅ Google Cloud integrations function
- ✅ AI features work properly

## 📞 **Need Help?**

If you still encounter issues:

1. Try Method 5 (Clean Environment)
2. Check Python version compatibility (3.9+ recommended)
3. Ensure virtual environment is activated
4. Use `--force-reinstall` for problematic packages

The platform is designed to work with minimal dependencies while maintaining full functionality! 🎉
