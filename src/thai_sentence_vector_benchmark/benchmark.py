from typing import List, Dict, Optional
from thai_sentence_vector_benchmark.tasks.sts import STSBenchmark
from thai_sentence_vector_benchmark.tasks.retrieval import RetrievalBenchmark
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel
from thai_sentence_vector_benchmark.tasks.pair_classification import PairClassificationBenchmark
from thai_sentence_vector_benchmark.tasks.text_classification import TextClassificationBenchmark


class ThaiSentenceVectorBenchmark:
    def __init__(
            self, 
            task_names: List[str] = ("sts", "text_classification", "pair_classification", "retrieval"),
    ):
        self.task_names = task_names

        self.tasks = {}
        for task_name in self.task_names:
            if task_name == "sts":
                self.tasks[task_name] = STSBenchmark()
            elif task_name == "retrieval":
                self.tasks[task_name] = RetrievalBenchmark()
            elif task_name == "pair_classification":
                self.tasks[task_name] = PairClassificationBenchmark()
            elif task_name == "text_classification":
                self.tasks[task_name] = TextClassificationBenchmark()

    def __call__(
            self, 
            model: SentenceEncodingModel, 
            task_prompts: Optional[Dict] = None,
            batch_size: int = 1024,
    ) -> Dict:
        results = {}
        average_result = []
        for task_name in self.task_names:
            print(f"Running {task_name} benchmark...")
            prompt = task_prompts[task_name] if task_prompts is not None else None
            result = self.tasks[task_name](model, prompt=prompt, batch_size=batch_size)
            if task_name == "sts":
                results["STS"] = {dataset_name: {"Spearman_Correlation": value["Spearman_Correlation"]} for dataset_name, value in result.items()}
                results["STS"]["Average"] = {
                    "Spearman_Correlation": round(sum([value["Spearman_Correlation"] for value in results["STS"].values()]) / len(results["STS"]), 2)
                }
                average_result.append(results["STS"]["Average"]["Spearman_Correlation"])
            elif task_name == "text_classification":
                results["Text_Classification"] = {dataset_name: {"Accuracy": value["Accuracy"], "F1": value["F1"]} for dataset_name, value in result.items()}
                results["Text_Classification"]["Average"] = {
                    "Accuracy": round(sum([value["Accuracy"] for value in results["Text_Classification"].values()]) / len(results["Text_Classification"]), 2),
                    "F1": round(sum([value["F1"] for value in results["Text_Classification"].values()]) / len(results["Text_Classification"]), 2),
                }
                average_result.append(results["Text_Classification"]["Average"]["Accuracy"])
                average_result.append(results["Text_Classification"]["Average"]["F1"])
            elif task_name == "pair_classification":
                results["Pair_Classification"] = {dataset_name: {"AP": value["AP"]} for dataset_name, value in result.items()}
                results["Pair_Classification"]["Average"] = {
                    "AP": round(sum([value["AP"] for value in results["Pair_Classification"].values()]) / len(results["Pair_Classification"]), 2)
                }
                average_result.append(results["Pair_Classification"]["Average"]["AP"])
            elif task_name == "retrieval":
                results["Retrieval"] = {dataset_name: {"R@1": value["R@1"], "MRR@10": value["MRR@10"]} for dataset_name, value in result.items()}
                results["Retrieval"]["Average"] = {
                    "R@1": round(sum([value["R@1"] for value in results["Retrieval"].values()]) / len(results["Retrieval"]), 2),
                    "MRR@10": round(sum([value["MRR@10"] for value in results["Retrieval"].values()]) / len(results["Retrieval"]), 2),
                }
                average_result.append(results["Retrieval"]["Average"]["R@1"])
                average_result.append(results["Retrieval"]["Average"]["MRR@10"])
            results["Average"] = round(sum(average_result) / len(average_result), 2)
            print(f"{task_name} benchmark done.")
            print(result)
            print("-" * 100)
        return results