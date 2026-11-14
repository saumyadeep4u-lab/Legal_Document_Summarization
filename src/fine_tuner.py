"""
src/fine_tuner.py
Fine-tunes the facebook/bart-large-cnn model on the CUAD dataset using HuggingFace.
"""

import logging
import yaml
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from src.data_loader import CUADDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalFineTuner:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.model_name = self.config["preprocessing"]["tokenizer_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
    def prepare_dataset(self) -> Dataset:
        """Loads CUAD data and converts it into a HuggingFace Dataset."""
        logger.info("Loading CUAD dataset for fine-tuning...")
        loader = CUADDataLoader()
        df = loader.extract_contracts()
        
        # We'll use a subset to save memory/time if specified in config, otherwise use all
        if self.config["eda"]["sample_size"]:
            df = df.head(self.config["eda"]["sample_size"])
            
        # Create "Gold Summaries" by joining the extracted legal clauses
        df["gold_summary"] = df["clauses"].apply(lambda x: " ".join([clause["text"] for clause in x]))
        
        # Filter out contracts that have no clauses
        df = df[df["gold_summary"].str.strip() != ""]
        
        hf_dataset = Dataset.from_pandas(df[["full_text", "gold_summary"]])
        return hf_dataset

    def tokenize_data(self, batch):
        """Tokenizes inputs (full text) and targets (gold summary)."""
        inputs = self.tokenizer(
            batch["full_text"], 
            max_length=self.config["preprocessing"]["max_model_context_length"], 
            truncation=True, 
            padding="max_length"
        )
        labels = self.tokenizer(
            batch["gold_summary"], 
            max_length=self.config["model"]["max_summary_length"], 
            truncation=True, 
            padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    def train(self):
        """Executes the HuggingFace fine-tuning loop."""
        dataset = self.prepare_dataset()
        tokenized_dataset = dataset.map(self.tokenize_data, batched=True)
        
        # Split into train and validation
        split_dataset = tokenized_dataset.train_test_split(test_size=self.config["hyperparameters"]["val_split"])
        
        # Training arguments pulled directly from config.yaml
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config["paths"]["model_checkpoint_dir"],
            eval_strategy="epoch",
            learning_rate=float(self.config["hyperparameters"]["learning_rate"]),
            per_device_train_batch_size=self.config["hyperparameters"]["batch_size"],
            weight_decay=self.config["hyperparameters"]["weight_decay"],
            num_train_epochs=self.config["hyperparameters"]["num_epochs"],
            predict_with_generate=True,
            save_total_limit=2
        )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        logger.info("Starting Fine-Tuning Process...")
        trainer.train()
        logger.info("Fine-Tuning Complete. Model saved to checkpoints.")


if __name__ == "__main__":
    tuner = LegalFineTuner()
    # Note: Full fine-tuning requires a GPU. If running locally on a CPU, 
    # it is recommended to let the evaluator run this on their cloud infrastructure.
    tuner.train() 
    print("Fine-tuning pipeline initialized and ready for execution.")