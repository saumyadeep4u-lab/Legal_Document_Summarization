"""
src/validate_10_contracts.py
End-to-end validation script. Extracts 10 real contracts, generates AI summaries,
and validates them against the dataset's gold summaries using ROUGE/BERTScore.
Outputs the results to a CSV file.
"""

import logging
import os
from src.data_loader import CUADDataLoader
from src.summarizer import LegalSummarizer
from src.evaluator import LegalEvaluator
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("--- 🚀 INITIATING 10-CONTRACT BATCH VALIDATION ---")

    loader = CUADDataLoader()
    df = loader.extract_contracts()

    df["gold_summary"] = df["clauses"].apply(lambda x: " ".join([clause["text"] for clause in x]))
    df_10 = df[df["gold_summary"].str.strip() != ""].head(10).reset_index(drop=True)

    summarizer = LegalSummarizer()
    evaluator = LegalEvaluator()

    rouge_1_total, bertscore_total = 0, 0
    results_list = []

    for idx, row in df_10.iterrows():
        title = row['contract_title']
        full_text = row['full_text']
        gold_summary = row['gold_summary']

        logger.info(f"\n📄 Processing Contract {idx + 1}/10: {title}")

        ai_summary = summarizer.summarize_document(full_text)
        metrics = evaluator.evaluate_summaries(reference=gold_summary, generated=ai_summary)

        rouge_1_total += metrics["ROUGE-1 (F1)"]
        bertscore_total += metrics["BERTScore (F1)"]

        logger.info(f"   ↳ ROUGE-1: {metrics['ROUGE-1 (F1)']} | BERTScore: {metrics['BERTScore (F1)']}")

        results_list.append({
            "Contract": title,
            "Gold_Summary_Reference": gold_summary,
            "AI_Generated_Summary": ai_summary,
            "ROUGE-1": metrics["ROUGE-1 (F1)"],
            "BERTScore": metrics["BERTScore (F1)"]
        })

    logger.info("\n==================================================")
    logger.info("✅ 10-CONTRACT VALIDATION COMPLETE")
    logger.info(f"📈 Average ROUGE-1 Score: {round(rouge_1_total / 10, 4)}")
    logger.info(f"📈 Average BERTScore:     {round(bertscore_total / 10, 4)}")
    logger.info("==================================================")

    # SAVE TO CSV
    os.makedirs("data/processed", exist_ok=True)
    results_df = pd.DataFrame(results_list)
    output_path = "data/processed/10_contract_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Saved 10-contract proof to: {output_path}")

if __name__ == "__main__":
    main()
