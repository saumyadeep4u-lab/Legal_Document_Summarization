"""
src/evaluator.py
Calculates industry-standard NLP metrics (ROUGE and BERTScore) 
to quantitatively evaluate the summarization engine's performance.
"""

import logging
from rouge_score import rouge_scorer
from bert_score import score
import warnings

# Suppress annoying huggingface warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalEvaluator:
    def __init__(self):
        """Initializes the ROUGE and BERTScore metric engines."""
        logger.info("Initializing NLP Evaluation Engine (ROUGE & BERTScore)...")
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_summaries(self, reference: str, generated: str) -> dict:
        """
        Compares an AI-generated summary against a human ground-truth reference.
        Returns a dictionary of precision, recall, and F1 scores.
        """
        logger.info("Calculating ROUGE metrics...")
        rouge_scores = self.rouge.score(reference, generated)
        
        logger.info("Calculating BERTScore metrics...")
        # Calculate BERTScore (using a smaller, faster model for rapid testing)
        P, R, F1 = score([generated], [reference], lang="en", model_type="distilbert-base-uncased", verbose=False)
        
        return {
            "ROUGE-1 (F1)": round(rouge_scores['rouge1'].fmeasure, 4),
            "ROUGE-2 (F1)": round(rouge_scores['rouge2'].fmeasure, 4),
            "ROUGE-L (F1)": round(rouge_scores['rougeL'].fmeasure, 4),
            "BERTScore (F1)": round(F1.item(), 4)
        }


if __name__ == "__main__":
    evaluator = LegalEvaluator()
    
    # Simulate a human lawyer's perfect summary (Ground Truth)
    human_reference = "The Receiving Party must keep information confidential for 5 years and only use it to evaluate a business transaction."
    
    # The output our AI generated earlier
    ai_generated = "The Receiving Party agrees that it shall strictly hold in confidence and shall not disclose to any third party any Confidential Information of the Disclosing Party. This obligation of confidentiality shall survive for a period of five (5) years from the date of disclosure. The Receiving party may only use the Confidential information for the purpose of evaluating a potential business transaction between the parties."
    
    print("\n=== 📊 NLP PERFORMANCE METRICS ===")
    metrics = evaluator.evaluate_summaries(human_reference, ai_generated)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")