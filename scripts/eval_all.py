import os
import argparse
import pandas as pd
from thai_sentence_vector_benchmark.benchmark import ThaiSentenceVectorBenchmark
from thai_sentence_vector_benchmark.models import SentenceTransformerModel, BGEModel, CohereV2Model, CohereV3Model, OpenAIModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohere_api_key", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs("./results", exist_ok=True)

    models_to_evaluate = {
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
        "BGE-M3": "BAAI/bge-m3",
        "Cohere-embed-multilingual-v2.0": "embed-multilingual-v2.0",
        "Cohere-embed-multilingual-v3.0": "embed-multilingual-v3.0",
        "Openai-text-embedding-3-large": "text-embedding-3-large",
    }

    benchmark = ThaiSentenceVectorBenchmark()

    results = {}
    for model_name, model_path in models_to_evaluate.items():
        print(f"Running benchmark for {model_name}...")
        if model_name == "BGE-M3":
            model = BGEModel(model_path)
        elif model_name == "Cohere-embed-multilingual-v2.0" and args.cohere_api_key is not None:
            model = CohereV2Model(model_path, api_key=args.cohere_api_key)
        elif model_name == "Cohere-embed-multilingual-v3.0" and args.cohere_api_key is not None:
            model = CohereV3Model(model_path, api_key=args.cohere_api_key)
        elif model_name == "Openai-text-embedding-3-large" and args.openai_api_key is not None:
            model = OpenAIModel(model_path, api_key=args.openai_api_key)
        else:
            model = SentenceTransformerModel(model_path)
        results[model_name] = benchmark(model, batch_size=args.batch_size)

        # Save main results to csv
        main_table = pd.DataFrame(
            {
                model_name: {
                "STS": str(result["STS"]["Spearman_Correlation"]),
                "Text_Classification": f'{result["Text_Classification"]["Average"]["Accuracy"]} / {result["Text_Classification"]["Average"]["F1"]}',
                "Pair_Classification": str(result["Pair_Classification"]["AP"]),
                "Retrieval": f'{result["Retrieval"]["Average"]["R@1"]} / {result["Retrieval"]["Average"]["MRR@10"]}',
                "Average": str(result["Average"]),
                }
            } for model_name, result in results.items()
        )
        main_table.to_csv("./results/main_results.csv", index=False)

        # Save text classification results to csv
        text_classification_table = pd.DataFrame(
            {
                model_name: {
                    dataset_name: f'{value["Accuracy"]} / {value["F1"]}' for dataset_name, value in result["Text_Classification"].items()
                }
            } for model_name, result in results.items()
        )
        text_classification_table.to_csv("./results/text_classification_results.csv", index=False)

        # Save retrieval results to csv
        retrieval_table = pd.DataFrame(
            {
                model_name: {
                    dataset_name: f'{value["R@1"]} / {value["MRR@10"]}' for dataset_name, value in result["Retrieval"].items()
                }
            } for model_name, result in results.items()
        )
        retrieval_table.to_csv("./results/retrieval_results.csv", index=False)