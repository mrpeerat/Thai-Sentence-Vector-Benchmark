import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from thai_sentence_vector_benchmark.benchmark import ThaiSentenceVectorBenchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)    # Ex. FacebookAI/xlm-roberta-base
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(f"./results/{args.model_name}", exist_ok=True)

    if args.task_name is None:
        benchmark = ThaiSentenceVectorBenchmark()
    else:
        benchmark = ThaiSentenceVectorBenchmark(task_names=[args.task_name])

    results = {}
    print(f"Running benchmark for {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    if args.model_name == "WangchanBERTa":
        model.max_seq_length = 416
    results[args.model_name] = benchmark(model, batch_size=args.batch_size)

    # Save main results to csv
    main_table = pd.DataFrame(
        {
            model_name: {
            "STS": str(result["STS"]["Average"]["Spearman_Correlation"]),
            "Text_Classification": f'{result["Text_Classification"]["Average"]["Accuracy"]} / {result["Text_Classification"]["Average"]["F1"]}',
            "Pair_Classification": str(result["Pair_Classification"]["Average"]["AP"]),
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
    text_classification_table.to_csv(f"./results/{args.model_name}/text_classification_results.csv", index=False)

    # Save retrieval results to csv
    retrieval_table = pd.DataFrame(
        {
            model_name: {
                dataset_name: f'{value["R@1"]} / {value["MRR@10"]}' for dataset_name, value in result["Retrieval"].items()
            }
        } for model_name, result in results.items()
    )
    retrieval_table.to_csv(f"./results/{args.model_name}/retrieval_results.csv", index=False)