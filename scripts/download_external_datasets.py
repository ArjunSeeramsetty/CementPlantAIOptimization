"""Automation script to download and organize external datasets.

Downloads public Kaggle datasets anonymously using kagglehub, fetches Arxiv
documents, and provides a utility to download Roboflow datasets via API keys.
"""

import os
import shutil
import urllib.request
import sys

# Ensure requirements are satisfied dynamically
def ensure_package(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


class DatasetDownloader:
    def __init__(self) -> None:
        self.base_dir = os.path.join(os.getcwd(), "data", "external")
        os.makedirs(self.base_dir, exist_ok=True)

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
            # Set user agent to avoid bot blocks
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"[Success] Document saved to: {pdf_path}")
        except Exception as e:
            print(f"[Error] Failed to download Arxiv document: {e}")

    def download_roboflow_clinker(self, api_key: str = None) -> None:
        """Download the Roboflow Clinker sizing dataset using API key."""
        print("\n--- 3. Downloading Roboflow Clinker Sizing Dataset ---")
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
    
    try:
        downloader.download_ppe_kit()
    except Exception as e:
        print(f"Error during PPE Kit download: {e}")
        
    try:
        downloader.download_arxiv_clinker()
    except Exception as e:
        print(f"Error during Arxiv download: {e}")
        
    try:
        downloader.download_roboflow_clinker(args.roboflow_key)
    except Exception as e:
        print(f"Error during Roboflow download: {e}")


if __name__ == "__main__":
    main()
