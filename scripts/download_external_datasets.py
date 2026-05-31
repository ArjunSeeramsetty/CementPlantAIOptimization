"""Automation script to download and organize external datasets.

Downloads public datasets from Kaggle, UCI ML Repository, and Zenodo records
dynamically using standard Python libraries, requests, and API lookups.
"""

import os
import sys
import shutil
import zipfile
from typing import Dict, List, Any

# Ensure requirements are satisfied dynamically
def ensure_package(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# Ensure requests is available
ensure_package("requests")
import requests

class DatasetDownloader:
    def __init__(self) -> None:
        self.base_dir = os.path.join(os.getcwd(), "data", "external")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Standard browser headers to bypass API request blocks
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }

    def download_ppe_kit(self) -> None:
        """Download the Kaggle PPE Kit Detection dataset using kagglehub."""
        print("\n--- 1. Downloading PPE Kit Detection Dataset ---")
        try:
            ensure_package("kagglehub")
            import kagglehub
        except Exception as e:
            print(f"[Warning] Could not import/install kagglehub library: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers")
            print("And extract contents directly to: data/external/ppe_construction/")
            return
        
        target_dir = os.path.join(self.base_dir, "ppe_construction")
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 1:
            print("[Info] PPE Kit dataset already exists. Skipping.")
            return

        try:
            print("Downloading from Kaggle (ketakichalke/ppe-kit-detection-construction-site-workers)...")
            path = kagglehub.dataset_download("ketakichalke/ppe-kit-detection-construction-site-workers")
            print(f"Downloaded to temporary cache: {path}")
            
            # Copy to target directory
            os.makedirs(target_dir, exist_ok=True)
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"[Success] PPE Kit dataset successfully saved to: {target_dir}")
        except Exception as e:
            print(f"[Error] Failed to download PPE Kit dataset: {e}")

    def download_arxiv_clinker(self) -> None:
        """Download the Arxiv paper PDF detailing the clinker microstructure dataset."""
        print("\n--- 2. Downloading Cement Clinker Microstructure Documentation ---")
        target_dir = os.path.join(self.base_dir, "clinker_microstructure")
        os.makedirs(target_dir, exist_ok=True)
        
        pdf_path = os.path.join(target_dir, "microstructure_paper.pdf")
        if os.path.exists(pdf_path):
            print("[Info] Microstructure paper documentation already exists. Skipping.")
            return

        try:
            url = "https://arxiv.org/pdf/2211.03223.pdf"
            print(f"Downloading Arxiv paper PDF from {url}...")
            
            r = requests.get(url, headers=self.headers, stream=True)
            r.raise_for_status()
            with open(pdf_path, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            print(f"[Success] Document saved to: {pdf_path}")
        except Exception as e:
            print(f"[Error] Failed to download Arxiv document: {e}")

    def download_uci_steel(self) -> None:
        """Download the UCI Steel Industry Energy Consumption dataset."""
        print("\n--- 3. Downloading UCI Steel Industry Energy Dataset ---")
        target_dir = os.path.join(self.base_dir, "steel_energy")
        os.makedirs(target_dir, exist_ok=True)

        # Check if dataset CSV is already present
        csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
        if csv_files:
            print(f"[Info] Steel energy dataset already exists ({csv_files[0]}). Skipping.")
            return

        zip_path = os.path.join(target_dir, "steel_energy.zip")
        try:
            url = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
            print(f"Downloading dataset ZIP from {url}...")
            
            r = requests.get(url, headers=self.headers, stream=True)
            r.raise_for_status()
            with open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            
            # Extract zip
            print("Extracting ZIP archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Cleanup zip file
            os.remove(zip_path)
            print(f"[Success] UCI Steel Energy dataset extracted to: {target_dir}")
        except Exception as e:
            print(f"[Error] Failed to download/extract UCI Steel dataset: {e}")

    def _download_url(self, url: str, dest_path: str = None) -> str:
        """Download URL content using curl.exe if on Windows, falling back to requests."""
        import platform
        import subprocess
        
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            try:
                cmd = ["curl.exe", "-s", "-L", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"]
                if dest_path:
                    cmd.extend(["-o", dest_path])
                cmd.append(url)
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode == 0:
                    if dest_path:
                        return ""
                    return result.stdout
            except Exception as e:
                print(f"[Debug] curl.exe failed, falling back to requests: {e}")
        
        # Fallback to requests
        if dest_path:
            r = requests.get(url, headers=self.headers, stream=True)
            r.raise_for_status()
            with open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            return ""
        else:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            return r.text

    def download_zenodo_record(self, record_id: int, folder_name: str) -> None:
        """Query Zenodo REST API and download all files for the specified record."""
        print(f"\n--- Downloading Zenodo Record ID: {record_id} ---")
        target_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        # Check if folder has files already
        if len(os.listdir(target_dir)) > 0:
            print(f"[Info] Zenodo files for {folder_name} already present. Skipping.")
            return

        try:
            import json
            api_url = f"https://zenodo.org/api/records/{record_id}"
            print(f"Querying Zenodo API: {api_url}...")
            
            response_text = self._download_url(api_url)
            record_data = json.loads(response_text)
            
            # Find files inside metadata
            files = record_data.get("files", [])
            # In some newer Zenodo API configurations, it resides under entries
            if not files and "entries" in record_data:
                files = record_data["entries"]
            
            if not files:
                print(f"[Warning] No files found in Zenodo metadata for record {record_id}.")
                return

            print(f"Found {len(files)} file(s) associated with this record.")
            for file_info in files:
                # Resolve download link
                filename = file_info.get("key") or file_info.get("filename") or file_info.get("id")
                # Links structure
                links = file_info.get("links", {})
                download_url = links.get("self") or links.get("download") or f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
                
                # Double check filename
                if not filename:
                    filename = download_url.split("/")[-1].split("?")[0]
                
                dest_path = os.path.join(target_dir, filename)
                print(f"Downloading {filename} from {download_url}...")
                
                self._download_url(download_url, dest_path)
                print(f"[Success] Saved file: {filename}")
                
        except Exception as e:
            print(f"[Error] Failed to download Zenodo record {record_id}: {e}")
            print(f"Please download files manually from: https://zenodo.org/records/{record_id}")
            print(f"And place them in: data/external/{folder_name}/")


    def download_roboflow_clinker(self, api_key: str = None) -> None:
        """Download the Roboflow Clinker sizing dataset using API key."""
        print("\n--- Downloading Roboflow Clinker Sizing Dataset ---")
        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        
        if not api_key:
            print("[Warning] Skipping Roboflow: ROBOFLOW_API_KEY not provided.")
            print("To download the Clinker Sizing dataset, run this script with:")
            print("  python scripts/download_external_datasets.py --roboflow-key YOUR_API_KEY")
            return

        try:
            ensure_package("roboflow")
            from roboflow import Roboflow
        except Exception as e:
            print(f"[Warning] Could not import/install roboflow library: {e}")
            return
        
        target_dir = os.path.join(self.base_dir, "clinker_sizing")
        os.makedirs(target_dir, exist_ok=True)

        try:
            print("Initializing Roboflow client...")
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("ziix").project("clinker-6c23i")
            version = project.version(1)
            
            print("Downloading dataset zip in YOLOv8 format...")
            dataset = version.download("yolov8", location=target_dir)
            print(f"[Success] Clinker sizing dataset saved to: {dataset.location}")
        except Exception as e:
            print(f"[Error] Failed to download Roboflow dataset: {e}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Download external datasets.")
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow Private API Key")
    args = parser.parse_args()

    downloader = DatasetDownloader()
    
    # 1. PPE Kit Detection (Kaggle)
    try:
        downloader.download_ppe_kit()
    except Exception as e:
        print(f"Error during PPE Kit download: {e}")
        
    # 2. Clinker Microstructure Paper (Arxiv)
    try:
        downloader.download_arxiv_clinker()
    except Exception as e:
        print(f"Error during Arxiv download: {e}")

    # 3. Steel Industry Energy Consumption (UCI)
    try:
        downloader.download_uci_steel()
    except Exception as e:
        print(f"Error during UCI Steel download: {e}")

    # 4. Global CO2 Emissions (Zenodo)
    try:
        downloader.download_zenodo_record(20397304, "cement_co2_emissions")
    except Exception as e:
        print(f"Error during Zenodo CO2 download: {e}")

    # 5. CEMCAP Capture Techno-Economic Analysis (Zenodo)
    try:
        downloader.download_zenodo_record(2597091, "cemcap_co2_capture")
    except Exception as e:
        print(f"Error during Zenodo CEMCAP download: {e}")
        
    # 6. Clinker Sizing (Roboflow)
    try:
        downloader.download_roboflow_clinker(args.roboflow_key)
    except Exception as e:
        print(f"Error during Roboflow download: {e}")


if __name__ == "__main__":
    main()
