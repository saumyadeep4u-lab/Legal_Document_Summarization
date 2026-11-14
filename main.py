"""
main.py
Master execution script for the Legal Document Summarization pipeline.
"""
import logging
import yaml
import os
from src.generate_pdf import main as generate_executive_brief

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    logger.info("Initializing Legal Document Summarization Pipeline...")
    
    # Load config to verify local model checkpoint path exists
    config = load_config()
    model_path = config.get("preprocessing", {}).get("tokenizer_name", "facebook/bart-large-cnn")
    logger.info(f"Target Model Source: {model_path}")

    if "./models" in model_path and not os.path.exists(model_path):
        logger.warning(
            f"⚠️ Warning: Local model path '{model_path}' was not found on disk. "
            "Ensure your downloaded checkpoint is properly placed before running inference."
        )

    # Execute the primary PDF executive brief generation pipeline
    logger.info("Starting executive brief generation...")
    generate_executive_brief()
    
    logger.info("✅ Pipeline execution complete. Check the 'deliverables/' folder for your outputs.")

if __name__ == "__main__":
    main()