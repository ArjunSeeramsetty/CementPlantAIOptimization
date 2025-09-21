"""Simple test script for DCS simulator."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulation.dcs_simulator import generate_dcs_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dcs_simulator():
    """Test the DCS simulator independently."""
    logger.info("Testing DCS Simulator...")
    
    try:
        # Generate 1 hour of data
        dcs_data = generate_dcs_data(
            duration_hours=1,
            sample_rate_seconds=1,
            output_path='data/processed/test_dcs_data.csv'
        )
        
        logger.info(f"✅ DCS Simulator Test Successful!")
        logger.info(f"  Records: {len(dcs_data):,}")
        logger.info(f"  Tags: {len(dcs_data.columns)}")
        logger.info(f"  Duration: {dcs_data.index[-1] - dcs_data.index[0]}")
        
        # Show sample data
        logger.info("\nSample DCS Data:")
        logger.info(dcs_data.head())
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DCS Simulator Test Failed: {e}")
        return False

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    success = test_dcs_simulator()
    sys.exit(0 if success else 1)
