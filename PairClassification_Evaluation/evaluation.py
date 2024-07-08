import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)


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


def ap_score(scores, labels, high_score_more_similar: bool):
    return average_precision_score(labels, scores * (1 if high_score_more_similar else -1))


def _compute_metrics(scores, labels, high_score_more_similar):
    
    acc, acc_threshold = find_best_acc_and_threshold(
        scores, labels, high_score_more_similar
    )
    f1, precision, recall, f1_threshold = find_best_f1_and_threshold(
        scores, labels, high_score_more_similar
    )
    ap = ap_score(scores, labels, high_score_more_similar)

    return {
        "accuracy": acc,
        "accuracy_threshold": acc_threshold,
        "f1": f1,
        "f1_threshold": f1_threshold,
        "precision": precision,
        "recall": recall,
        "ap": ap,
    }


def compute_metrics(dataset,model):
    sentence1, sentence2, labels = zip(*dataset)
    sentences1 = list(sentence1)
    sentences2 = list(sentence2)
    labels = [int(x) for x in labels]
    
    sentences = list(set(sentences1 + sentences2))
        
    embeddings = np.asarray(model.encode(sentences))
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    embeddings1 = [emb_dict[sent] for sent in sentences1]
    embeddings2 = [emb_dict[sent] for sent in sentences2]
    
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
        output_scores[short_name] = _compute_metrics(scores, labels, reverse)

    return output_scores


def cal_score(dataset, model):
    scores = compute_metrics(dataset,model)
    # Main score is the max of Average Precision (AP)
    main_score = max(scores[short_name]["ap"] for short_name in scores)
    scores["main_score"] = main_score
    return scores


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
    filepath = "./data/XNLI/xnli.dev.tsv"
    df = pd.read_csv(filepath, sep="\t")
    df.loc[(df['language']=='th') & (df['gold_label']=='contradiction'),'label'] = '0'
    df.loc[(df['language']=='th') & (df['gold_label']=='entailment'),'label'] = '1'
    dev_dataset = df[(df['language']=='th') & (df['gold_label']!='neutral')][['sentence1','sentence2','label']].values.tolist()

    filepath = "./data/XNLI/xnli.test.tsv"
    df = pd.read_csv(filepath, sep="\t")
    df.loc[(df['language']=='th') & (df['gold_label']=='contradiction'),'label'] = '0'
    df.loc[(df['language']=='th') & (df['gold_label']=='entailment'),'label'] = '1'
    test_dataset = df[(df['language']=='th') & (df['gold_label']!='neutral')][['sentence1','sentence2','label']].values.tolist()

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
        dev_score = cal_score(dev_dataset, model)
        results[model_name].update({"Dev": round(dev_score["main_score"] * 100, 2)})
        test_score = cal_score(test_dataset, model)
        results[model_name].update({"Test": round(test_score["main_score"] * 100, 2)})

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/xnli.csv")
    print(results_df)