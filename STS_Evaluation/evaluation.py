import os
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator



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
    filepath = "./data/stsbenchmark/sts-test_th.csv"
    thai_sts = pd.read_csv(filepath, header=None).dropna()
    thai_sts.columns = ["text_1","text_2","correlation"]
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=thai_sts["text_1"], 
        sentences2=thai_sts["text_2"], 
        scores=thai_sts["correlation"], 
        batch_size=16, 
        show_progress_bar=False,
    )

    results = defaultdict(lambda: defaultdict(float))
    for model_name, model_path in tqdm(models_to_test.items()):
        print(f"Evaluating {model_name}...")
        # Load model
        model = SentenceTransformer(
            model_path, tokenizer_kwargs={"use_fast": False}    # WangchanBERTa doesn't support fast tokenizer
        )
        if "WangchanBERTa" in model_name:
            model.max_seq_length = 416
        # Evaluate model
        metrics = evaluator(model)
        results[model_name]["spearman_cosine"] = round(metrics["spearman_cosine"] * 100, 2)

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/sts.csv")
    print(results_df)