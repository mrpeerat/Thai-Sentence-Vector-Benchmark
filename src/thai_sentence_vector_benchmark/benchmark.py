from typing import List, Dict
from thai_sentence_vector_benchmark.tasks.sts import STSBenchmark
from thai_sentence_vector_benchmark.tasks.retrieval import RetrievalBenchmark
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel
from thai_sentence_vector_benchmark.tasks.pair_classification import PairClassificationBenchmark
from thai_sentence_vector_benchmark.tasks.text_classification import TextClassificationBenchmark


class ThaiSentenceVectorBenchmark:
    def __init__(self, tasks: List[str] = ("sts", "text_classification", "pair_classification", "retrieval")):
        self.tasks = tasks
        self.benchmarks = {}
        for task in self.tasks:
            if task == "sts":
                self.benchmarks[task] = STSBenchmark()
            elif task == "retrieval":
                self.benchmarks[task] = RetrievalBenchmark()
            elif task == "pair_classification":
                self.benchmarks[task] = PairClassificationBenchmark()
            elif task == "text_classification":
                self.benchmarks[task] = TextClassificationBenchmark()

    def __call__(self, model: SentenceEncodingModel) -> Dict:
        results = {}
        average_result = []
        for task in self.tasks:
            print(f"Running {task} benchmark...")
            result = self.benchmarks[task](model)
            if task == "sts":
                results["STS"] = {"Spearman_Correlation": result["spearman_cosine"]}
                average_result.append(results["STS"]["Spearman_Correlation"])
            elif task == "text_classification":
                results["Text_Classification"] = {dataset_name: {"Accuracy": value["Accuracy"], "F1": value["F1"]} for dataset_name, value in result.items()}
                results["Text_Classification"]["Average"] = {
                    "Accuracy": round(sum([value["Accuracy"] for value in results["Text_Classification"].values()]) / len(results["Text_Classification"]) * 100, 2),
                    "F1": round(sum([value["F1"] for value in results["Text_Classification"].values()]) / len(results["Text_Classification"]) * 100, 2),
                }
                average_result.append(results["Text_Classification"]["Average"]["Accuracy"])
                average_result.append(results["Text_Classification"]["Average"]["F1"])
            elif task == "pair_classification":
                results["Pair_Classification"] = {"AP": result["AP"]}
                average_result.append(results["Pair_Classification"]["AP"])
            elif task == "retrieval":
                results["Retrieval"] = {dataset_name: {"R@1": value["R@1"], "MRR@10": value["MRR@10"]} for dataset_name, value in result.items()}
                results["Retrieval"]["Average"] = {
                    "R@1": round(sum([value["R@1"] for value in results["Retrieval"].values()]) / len(results["Retrieval"]) * 100, 2),
                    "MRR@10": round(sum([value["MRR@10"] for value in results["Retrieval"].values()]) / len(results["Retrieval"]) * 100, 2),
                }
                average_result.append(results["Retrieval"]["Average"]["R@1"])
                average_result.append(results["Retrieval"]["Average"]["MRR@10"])
            results["Average"] = sum(average_result) / len(average_result)
            print(f"{task} benchmark done.")
            print(result)
            print("-" * 100)
        return results