import numpy as np
import pandas as pd
from typing import Optional
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)


class PairClassificationBenchmark:
    def __init__(self, data_dir: str = "./data/XNLI/xnli.test.tsv"):
        self.data_dir = data_dir
        df = pd.read_csv(self.data_dir, sep="\t")
        df.loc[(df['language']=='th') & (df['gold_label']=='contradiction'),'label'] = '0'
        df.loc[(df['language']=='th') & (df['gold_label']=='entailment'),'label'] = '1'
        self.dataset = df[(df['language']=='th') & (df['gold_label']!='neutral')][['sentence1','sentence2','label']].values.tolist()

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

    @staticmethod
    def ap_score(scores, labels, high_score_more_similar: bool):
        return average_precision_score(labels, scores * (1 if high_score_more_similar else -1))

    def _compute_metrics(self, scores, labels, high_score_more_similar):
        acc, acc_threshold = self.find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(
            scores, labels, high_score_more_similar
        )
        ap = self.ap_score(scores, labels, high_score_more_similar)

        return {
            "accuracy": acc,
            "accuracy_threshold": acc_threshold,
            "f1": f1,
            "f1_threshold": f1_threshold,
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

    def compute_metrics(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
        sentence1, sentence2, labels = zip(*self.dataset)
        sentences1 = list(sentence1)
        sentences2 = list(sentence2)
        labels = [int(x) for x in labels]

        embeddings1 = np.asarray(model.encode(sentences1, prompt=prompt, batch_size=batch_size, show_progress_bar=True))
        embeddings2 = np.asarray(model.encode(sentences2, prompt=prompt, batch_size=batch_size, show_progress_bar=True))
        
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)
        
        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]
        
        labels = np.asarray(labels)
        output_scores = {}
        for short_name, name, scores, reverse in [
            ["cos_sim", "Cosine-Similarity", cosine_scores, True],
            ["manhattan", "Manhattan-Distance", manhattan_distances, False],
            ["euclidean", "Euclidean-Distance", euclidean_distances, False],
            ["dot", "Dot-Product", dot_scores, True],
        ]:
            output_scores[short_name] = self._compute_metrics(scores, labels, reverse)

        return output_scores

    def cal_score(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
        scores = self.compute_metrics(model, prompt=prompt, batch_size=batch_size)
        # Main score is the max of Average Precision (AP)
        main_score = max(scores[short_name]["ap"] for short_name in scores)
        scores["main_score"] = main_score
        return scores

    def __call__(
            self, 
            model: SentenceEncodingModel,
            prompt: Optional[str] = None,
            batch_size: int = 1024,
    ):
        return {
            "xnli": {"AP": round(self.cal_score(model, prompt=prompt, batch_size=batch_size)["cos_sim"]["ap"] * 100, 2)},
        }