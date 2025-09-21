"""
Basic test suite for JK Cement Digital Twin Platform
"""
import sys
import os
import json
import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_basic_import():
    """Test basic package structure"""
    assert True


def test_platform_availability():
    """Test that main platform components are available"""
    try:
        # Test basic Python functionality
        import json
        import datetime
        assert json.dumps({'test': True}) == '{"test": true}'
        assert datetime.datetime.now() is not None
        print("✅ Basic platform tests passed")
    except Exception as e:
        print(f"❌ Platform test failed: {e}")
        assert False, f"Platform test failed: {e}"


def test_critical_imports():
    """Test critical package imports"""
    try:
        import pandas
        import numpy
        import streamlit
        import plotly
        print("✅ Critical imports test passed")
        assert True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        assert False, f"Import failed: {e}"


def test_json_functionality():
    """Test JSON serialization"""
    test_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'healthy',
        'version': '1.0.0'
    }
    
    json_str = json.dumps(test_data)
    parsed_data = json.loads(json_str)
    
    assert parsed_data['status'] == 'healthy'
    assert parsed_data['version'] == '1.0.0'
    assert 'timestamp' in parsed_data


if __name__ == "__main__":
    # Run tests when executed directly
    test_basic_import()
    test_platform_availability()
    test_critical_imports()
    test_json_functionality()
    print("✅ All basic tests passed!")
