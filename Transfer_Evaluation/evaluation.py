import os
import pandas as pd
from collections import defaultdict
from transfer import TransferEvaluator
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

    # Evaluate SentenceTransformer models
    models_to_test = {
        "XLMR": "FacebookAI/xlm-roberta-base",
        "WangchanBERTa": "airesearch/wangchanberta-base-att-spm-uncased",
        "PhayaThaiBERT ": "clicknext/phayathaibert",
        "SimCSE-XLMR": "kornwtp/simcse-model-XLMR",
        "SimCSE-WangchanBERTa": "kornwtp/simcse-model-wangchanberta",
        "SimCSE-PhayaThaiBERT": "kornwtp/simcse-model-phayathaibert",
        "SCT-XLMR": "kornwtp/SCT-model-XLMR",
        "SCT-WangchanBERTa": "kornwtp/SCT-model-wangchanberta",
        "SCT-PhayaThaiBERT": "kornwtp/SCT-model-phayathaibert",
        "ConGen-XLMR": "kornwtp/ConGen-model-XLMR",
        "ConGen-WangchanBERTa": "kornwtp/ConGen-model-wangchanberta",
        "ConGen-PhayaThaiBERT": "kornwtp/ConGen-model-phayathaibert",
        "SCT-KD-XLMR": "kornwtp/SCT-KD-model-XLMR",
        "SCT-KD-WangchanBERTa": "kornwtp/SCT-KD-model-wangchanberta",
        "SCT-KD-PhayaThaiBERT": "kornwtp/SCT-KD-model-phayathaibert",
        "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "distiluse-base-multilingual-cased-v2": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    }

    # Load dataset
    task_names = ["wisesight_sentiment", "wongnai_reviews", "generated_reviews_enth"]
    evaluator = TransferEvaluator(task_names, test_score=True)

    results = defaultdict(lambda: defaultdict(float))
    for model_name, model_path in models_to_test.items():
        print(f"Evaluating {model_name}...")
        # Load model
        model = SentenceTransformer(
            model_path, tokenizer_kwargs={
                "use_fast": False   # WangchanBERTa doesn't support fast tokenizer
            }
        )
        if "WangchanBERTa" in model_name:
            model.max_seq_length = 416
        # Evaluate model
        metrics = evaluator(model, cohere=False)

        results[model_name] = {}
        for dataset_name in metrics:
            results[model_name].update({
                f"{dataset_name}-Acc": round(metrics[dataset_name]["test"]["accuracy"] * 100, 2),
                f"{dataset_name}-Macro-F1": round(metrics[dataset_name]["test"]["macro avg"]["f1-score"] * 100, 2),
                f"{dataset_name}-Micro-F1": round(metrics[dataset_name]["test"]["weighted avg"]["f1-score"] * 100, 2),
            })
        print(results[model_name])

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/transfer.csv")
    print(results_df)