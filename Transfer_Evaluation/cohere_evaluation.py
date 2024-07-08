import os
import pandas as pd
from collections import defaultdict
from transfer import TransferEvaluator

import cohere
co = cohere.Client('Z0AuLPY1Q2B2n0o3zyntszwvWmBB5MCqnnnuRyNc')


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

    # Load dataset
    task_names = ["wisesight_sentiment", "wongnai_reviews", "generated_reviews_enth"]
    evaluator = TransferEvaluator(task_names, test_score=True)

    results = defaultdict(lambda: defaultdict(float))
    print(f"Evaluating...")
    # Evaluate model
    metrics = evaluator(co, cohere=True)

    results["Cohere"] = {}
    for dataset_name in metrics:
        results["Cohere"].update({
            f"{dataset_name}-Acc": round(metrics[dataset_name]["test"]["accuracy"] * 100, 2),
            f"{dataset_name}-Macro-F1": round(metrics[dataset_name]["test"]["macro avg"]["f1-score"] * 100, 2),
            f"{dataset_name}-Micro-F1": round(metrics[dataset_name]["test"]["weighted avg"]["f1-score"] * 100, 2),
        })

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/cohere_transfer.csv")
    print(results_df)

