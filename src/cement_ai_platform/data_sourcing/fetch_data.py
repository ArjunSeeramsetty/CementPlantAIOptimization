"""Data sourcing module for downloading foundational datasets."""

import os
import requests
import zipfile
import io
import pandas as pd
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_mendeley_lci_data(raw_path: str = 'data/raw', processed_path: str = 'data/processed') -> bool:
    """
    Downloads and extracts the Primary Life Cycle Inventory Data for Indian Cement Plants
    from Mendeley Data (DOI: 10.17632/hk9yfsdhh9.2).
    
    This dataset provides the foundational mass and energy balances for our digital twin.
    
    Args:
        raw_path: Directory to store raw downloaded files
        processed_path: Directory to store processed CSV files
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Try multiple Mendeley URLs as the direct download link may not work
    urls = [
        "https://data.mendeley.com/public-files/datasets/hk9yfsdhh9/files/10545f4a-0738-4635-9231-300f88155c21/file_downloaded",
        "https://data.mendeley.com/datasets/hk9yfsdhh9/2/files/10545f4a-0738-4635-9231-300f88155c21/file_downloaded",
        "https://data.mendeley.com/api/datasets/hk9yfsdhh9/files/10545f4a-0738-4635-9231-300f88155c21/download"
    ]
    
    zip_filepath = os.path.join(raw_path, 'mendeley_lci_data.zip')
    
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    logger.info("Downloading Mendeley LCI dataset for Indian cement plants...")
    
    for i, url in enumerate(urls):
        try:
            logger.info(f"Trying URL {i+1}/{len(urls)}: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check if response is actually a zip file
            content_type = response.headers.get('content-type', '')
            content_length = int(response.headers.get('content-length', 0))
            
            if content_length < 10000:  # Too small for a real dataset
                logger.warning(f"Response too small ({content_length} bytes), trying next URL...")
                continue
            
            # Save the zip file
            with open(zip_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded zip file to {zip_filepath} ({content_length} bytes)")
            
            # Try to extract and process the Excel file
            try:
                with zipfile.ZipFile(zip_filepath, 'r') as z:
                    excel_files = [f for f in z.namelist() if f.endswith('.xlsx')]
                    
                    if not excel_files:
                        logger.warning("No Excel files found in the downloaded archive, trying next URL...")
                        continue
                        
                    excel_filename = excel_files[0]
                    z.extract(excel_filename, path=raw_path)
                    logger.info(f"Extracted '{excel_filename}' to '{raw_path}'")

                    # Read the Excel file and save each sheet as a CSV
                    excel_filepath = os.path.join(raw_path, excel_filename)
                    xls = pd.ExcelFile(excel_filepath)
                    
                    processed_files = []
                    for sheet_name in xls.sheet_names:
                        try:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            # Clean up sheet names for use as filenames
                            clean_sheet_name = sheet_name.lower().replace(' ', '_').replace('-', '_')
                            csv_path = os.path.join(processed_path, f"mendeley_lci_{clean_sheet_name}.csv")
                            df.to_csv(csv_path, index=False)
                            processed_files.append(csv_path)
                            logger.info(f"  - Saved sheet '{sheet_name}' ({len(df)} rows) to '{csv_path}'")
                        except Exception as e:
                            logger.warning(f"  - Failed to process sheet '{sheet_name}': {e}")
                    
                    logger.info(f"Mendeley LCI data successfully processed. {len(processed_files)} files created.")
                    return True
                    
            except zipfile.BadZipFile:
                logger.warning("Downloaded file is not a valid zip file, trying next URL...")
                continue
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading from URL {i+1}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error processing URL {i+1}: {e}")
            continue
    
    logger.error("All Mendeley URLs failed. Please download the dataset manually from:")
    logger.error("https://data.mendeley.com/datasets/hk9yfsdhh9/2")
    logger.error("And place the Excel file in data/raw/ as 'mendeley_lci_data.xlsx'")
    return False


def download_kaggle_quality_data(raw_path: str = 'data/raw', processed_path: str = 'data/processed') -> bool:
    """
    Downloads the Kaggle Concrete Compressive Strength dataset.
    
    This dataset is used for training the quality prediction models.
    
    Args:
        raw_path: Directory to store raw downloaded files
        processed_path: Directory to store processed CSV files
        
    Returns:
        bool: True if successful, False otherwise
    """
    dataset_slug = 'vinayakshanawad/cement-manufacturing-concrete-dataset'
    kaggle_raw_path = os.path.join(raw_path, 'kaggle_strength')
    
    os.makedirs(kaggle_raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    logger.info(f"Downloading Kaggle dataset: {dataset_slug}...")
    
    try:
        # Try to import kaggle API
        import kaggle
        
        # Ensure Kaggle API credentials are configured
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_slug, path=kaggle_raw_path, unzip=True)
        
        # The file is named 'concrete_data.csv' in the archive
        original_file = os.path.join(kaggle_raw_path, 'concrete_data.csv')
        final_file = os.path.join(processed_path, 'concrete_compressive_strength.csv')
        
        if os.path.exists(original_file):
            df = pd.read_csv(original_file)
            df.to_csv(final_file, index=False)
            logger.info(f"Successfully downloaded and processed Kaggle data: {len(df)} records")
            logger.info(f"Data saved to {final_file}")
            return True
        else:
            logger.error("Error: 'concrete_data.csv' not found in the downloaded files.")
            return False

    except ImportError:
        logger.error("Kaggle API not available. Please install kaggle package:")
        logger.error("pip install kaggle")
        logger.error("And configure your API key in ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {e}")
        logger.error("Please ensure your kaggle.json API key is correctly configured")
        return False




def download_global_cement_database(raw_path: str = 'data/raw', processed_path: str = 'data/processed') -> bool:
    """
    Downloads the Global Cement Production Assets Database from Dryad.
    
    This provides context and benchmarking data for our digital twin.
    
    Args:
        raw_path: Directory to store raw downloaded files
        processed_path: Directory to store processed CSV files
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.error("Global Cement Database download not implemented.")
    logger.error("Please download the dataset manually from:")
    logger.error("https://datadryad.org/stash/dataset/doi:10.5061/dryad.XXXXX")
    logger.error("And place the CSV file in data/raw/ as 'global_cement_database.csv'")
    return False


def download_all_datasets() -> dict:
    """
    Downloads all required datasets for the digital twin POC.
    
    Returns:
        dict: Status of each dataset download
    """
    results = {}
    
    logger.info("=== Starting Data Sourcing for Digital Twin POC ===")
    
    # Download Mendeley LCI data (primary source)
    results['mendeley_lci'] = download_mendeley_lci_data()
    
    # Download Kaggle quality data
    results['kaggle_quality'] = download_kaggle_quality_data()
    
    # Download/create global cement database
    results['global_database'] = download_global_cement_database()
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    
    logger.info(f"=== Data Sourcing Complete: {successful}/{total} datasets successful ===")
    
    for dataset, status in results.items():
        status_str = "✅ SUCCESS" if status else "❌ FAILED"
        logger.info(f"  {dataset}: {status_str}")
    
    return results


if __name__ == "__main__":
    # Run data sourcing when executed directly
    download_all_datasets()
