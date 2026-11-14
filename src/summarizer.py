"""
src/summarizer.py
Core AI inference module. Uses facebook/bart-large-cnn.
"""

import logging
import torch
import yaml
import nltk
import re  
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LegalSummarizer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        model_name = self.config["preprocessing"]["tokenizer_name"]
        logger.info(f"Loading AI Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def chunk_document(self, text: str) -> list:
        """Splits document into token-safe chunks using the BART tokenizer."""
        if not text or not text.strip():
            return [""]
        chunk_size = self.config.get("preprocessing", {}).get("chunk_size", 800)
        chunk_overlap = self.config.get("preprocessing", {}).get("chunk_overlap", 128)
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(tokens) <= chunk_size:
            return [text]
            
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            if i + chunk_size >= len(tokens):
                break
        return chunks if chunks else [text]

    def extract_focused_clauses(self, full_text: str, focus_tag: str) -> str:
        if not focus_tag or focus_tag == "Auto-Detect" or focus_tag == "General Legal Terms":
            return full_text
            
        try:
            sentences = nltk.sent_tokenize(full_text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(full_text)
        
        synonym_map = {
            "insurance": ["insurance", "policy", "coverage", "premium", "underwriter", "insured"],
            "renewal": ["renew", "renewal", "extend", "successive", "expire"], 
            "liability": ["liability", "liable", "damages", "cap", "indemnify", "uncapped", "consequential"],
            "price": ["price", "fee", "cost", "charge", "payment", "dollars", "amount"],
            "volume": ["volume", "quantity", "amount", "minimum", "maximum", "quota"],
            "warranty": ["warranty", "warrant", "guarantee", "defect", "repair"], 
            "audit": ["audit", "inspect", "records", "books", "compliance"], 
            "assignment": ["assign", "assignment", "transfer", "delegate", "sublicense"],
            "solicit": ["solicit", "hire", "recruit", "poach", "employee", "personnel", "entice"],
            "compete": ["compete", "competitor", "competitive", "non-compete", "endorse"]
        }
        
        tag_lower = focus_tag.lower()
        search_keywords = []
        for key, synonyms in synonym_map.items():
            if key in tag_lower:
                search_keywords.extend(synonyms)
        if not search_keywords:
            search_keywords = [word for word in tag_lower.split() if len(word) > 3]
                
        relevant_indices = set()
        found_match = False
        
        # Strict 5-sentence window around the target ONLY
        for i, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            if any(re.search(rf"\b{re.escape(kw)}\b", sent_lower) for kw in search_keywords):
                relevant_indices.update(range(max(0, i-2), min(len(sentences), i+3)))
                found_match = True
                
        if not found_match:
            return full_text
            
        return " ".join([sentences[i] for i in sorted(list(relevant_indices))])

    def summarize_chunk(self, chunk_text: str) -> str:
        inputs = self.tokenizer(
            chunk_text, max_length=1024, truncation=True, return_tensors="pt"
        ).to(self.device)
        
        input_token_count = inputs["input_ids"].shape[1]
        configured_min = self.config["model"]["min_summary_length"]
        dynamic_min = min(configured_min, max(20, input_token_count // 3))
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.config["model"]["max_summary_length"],
            min_length=dynamic_min,  
            length_penalty=1.5,      
            num_beams=8,             
            no_repeat_ngram_size=4,  
            repetition_penalty=1.2,  
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_document(self, full_text: str, focus_tag: str = None) -> str:
        logger.info(f"Applying extractive filter for tag: {focus_tag}")
        targeted_text = self.extract_focused_clauses(full_text, focus_tag)
        chunks = self.chunk_document(targeted_text)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i + 1} of {len(chunks)}...")
            summary = self.summarize_chunk(chunk)
            summary = summary.strip()
            if summary and summary[-1] not in ".!?":
                summary += "."
            chunk_summaries.append(summary)
            
        final_summary = " ".join(chunk_summaries)
        return final_summary
