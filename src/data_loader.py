"""
src/data_loader.py
Modular data loader and parser for the CUAD (Contract Understanding Atticus Dataset).
Includes automated downloading and ZIP extraction for evaluator reproducibility.
"""

import json
import logging
import urllib.request
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load external YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CUADDataLoader:
    def __init__(self, config: Optional[dict] = None, config_path: str = "config.yaml"):
        self.config = config or load_config(config_path)
        self.raw_dir = Path(self.config["paths"]["raw_data_dir"])
        self.raw_path = Path(self.config["paths"]["cuad_json_path"])
        # Updated Zenodo URL (using /records/ and pointing to the official .zip file)
        self.dataset_url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
        self.zip_path = self.raw_dir / "CUAD_v1.zip"

    def download_dataset(self):
        """Downloads and extracts the dataset automatically if it is missing."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset not found locally. Downloading ZIP from {self.dataset_url}...")
        logger.info("This is a ~105MB file and may take a few minutes. Please wait...")
        
        # 1. Download the ZIP file
        urllib.request.urlretrieve(self.dataset_url, self.zip_path)
        logger.info("Download complete! Extracting files...")
        
        # 2. Extract the ZIP file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
            
        # 3. Find the JSON file inside the extracted folders and move it to our configured path
        for file in self.raw_dir.rglob("*.json"):
            if "CUAD" in file.name:
                file.replace(self.raw_path)
                break
                
        logger.info("Extraction complete and ready for processing!")

    def load_raw_json(self) -> dict:
        """Loads the raw CUAD JSON dataset file, downloading it first if necessary."""
        if not self.raw_path.exists():
            self.download_dataset()
            
        logger.info(f"Loading raw CUAD data from {self.raw_path}...")
        with open(self.raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def extract_contracts(self) -> pd.DataFrame:
        """
        Parses SQuAD-formatted CUAD JSON into a structured DataFrame.
        Extracts contract title, full text context, and clause annotations.
        """
        raw_data = self.load_raw_json()
        contracts = []

        data_entries = raw_data.get("data", [])
        logger.info(f"Extracting contracts from {len(data_entries)} document groups...")

        for entry in data_entries:
            title = entry.get("title", "Untitled Contract")
            for paragraph in entry.get("paragraphs", []):
                context = paragraph.get("context", "")
                qas = paragraph.get("qas", [])

                clause_records = []
                for qa in qas:
                    clause_type = qa.get("question", "")
                    answers = qa.get("answers", [])
                    for ans in answers:
                        clause_records.append({
                            "clause_type": clause_type,
                            "text": ans.get("text", ""),
                            "start_pos": ans.get("answer_start", -1)
                        })

                contracts.append({
                    "contract_title": title,
                    "full_text": context,
                    "num_chars": len(context),
                    "clauses": clause_records,
                    "num_clauses": len(clause_records)
                })

        df = pd.DataFrame(contracts)
        logger.info(f"Successfully processed {len(df)} contract records.")
        return df


if __name__ == "__main__":
    loader = CUADDataLoader()
    df_contracts = loader.extract_contracts()