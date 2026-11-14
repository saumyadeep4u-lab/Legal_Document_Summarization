"""
src/preprocessor.py
Handles token-aware chunking for long legal documents using sentence boundaries 
and Named Entity Recognition (NER) tagging to protect critical legal facts.
"""

import logging
from typing import List, Dict
from transformers import AutoTokenizer
import yaml
import spacy
import nltk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initializes the tokenizer and NER engine using external configurations."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["preprocessing"]["tokenizer_name"])
        
        logger.info("Loading SpaCy NER model (en_core_web_sm)...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("SpaCy model missing! Please run: python -m spacy download en_core_web_sm")
            raise

    def clean_text(self, text: str) -> str:
        """Removes excessive whitespace, invisible characters, and normalizes text."""
        return " ".join(text.split())

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Applies NER tagging to extract and protect critical legal facts 
        (Organizations, People, Monetary values, and Dates).
        """
        # Process up to 100k characters at a time to prevent memory overflow
        doc = self.nlp(text[:100000]) 
        
        entities = {"ORG": set(), "PERSON": set(), "MONEY": set(), "DATE": set()}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].add(ent.text)
                
        # Convert sets back to lists for JSON serialization downstream
        return {k: list(v) for k, v in entities.items()}

    def chunk_document(self, text: str, max_chunk_chars: int = 2500) -> List[str]:
        """
        Splits a long legal document into chunks safely using NLTK sentence boundaries.
        This prevents cutting words in half mid-sentence.
        """
        cleaned_text = self.clean_text(text)
        
        try:
            sentences = nltk.sent_tokenize(cleaned_text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(cleaned_text)
            
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Chunked document into {len(chunks)} sentence-aware windows.")
        return chunks


if __name__ == "__main__":
    # Quick test to ensure the module works
    sample_legal_text = "Google LLC agrees to pay John Doe the sum of $50,000 on January 1st, 2027 for services rendered."
    
    preprocessor = LegalPreprocessor()
    
    print("\n--- NER Tagging Test ---")
    entities = preprocessor.extract_legal_entities(sample_legal_text)
    print(entities)
    
    print("\n--- Chunking Test ---")
    chunks = preprocessor.chunk_document(sample_legal_text)
    print(f"Number of chunks generated: {len(chunks)}")