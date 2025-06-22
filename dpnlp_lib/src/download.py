import os
import requests
import zipfile
from pathlib import Path

home_dir = Path(__file__).parent
DATASET_PATH_DEMO = home_dir / "dataset/mimic-iii-clinical-database-demo-1.4"
#DATASET_PATH = home_dir / "dataset/physionet.org/files/mimiciii/1.4"
DATASET_PATH = Path("/home/tanalp/physionet.org/files/mimiciii/1.4")
dataset_url = "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip"  # use demo dataset for now, only includes 100 patients and clinical notes removed, 26 tables

def download_file(url, save_path):
    print(f"Downloading MIMIC-III Demo from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Dataset saved to {save_path}.")
    else:
        raise Exception(f"Failed. Status code: {response.status_code}")


def extract_zip(zip_path, extraction_path):
    print(f"Extracting {zip_path} to {extraction_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_path)
    print(f"Completed!")


if __name__ == "__main__":
    os.makedirs(DATASET_PATH_DEMO, exist_ok=True)
    ZIP_PATH = os.path.join(DATASET_PATH_DEMO, "dataset.zip")
    download_file(dataset_url, ZIP_PATH)
    extract_zip(ZIP_PATH, DATASET_PATH_DEMO)
