import os
import numpy as np
import pandas as pd
from tqdm import trange
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
import cohere
co = cohere.Client('Z0AuLPY1Q2B2n0o3zyntszwvWmBB5MCqnnnuRyNc')


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


def compute_metrics_cohere(dataset):
    sentence1, sentence2, labels = zip(*dataset)
    sentences1 = list(sentence1)
    sentences2 = list(sentence2)
    labels = [int(x) for x in labels]
    
    # sentences = list(set(sentences1 + sentences2))
    
    # bs = 96
    # embeddings = []
    # for i in trange(len(sentences)//bs+1):
    #     embeddings.append(co.embed(
    #       texts=sentences[(i*bs):((i+1)*bs)],
    #       model='embed-multilingual-v2.0',
    #     ).embeddings)
        
    bs = 96
    embed1 = []
    embed2 = []
    for i in trange(len(sentence1)//bs+1):
        embed1.append(co.embed(
          texts=sentences1[(i*bs):((i+1)*bs)],
          model='embed-multilingual-v2.0',
        ).embeddings)
        embed2.append(co.embed(
          texts=sentences2[(i*bs):((i+1)*bs)],
          model='embed-multilingual-v2.0',
        ).embeddings)
    
    embeddings1 = np.concatenate(embed1,0)
    embeddings2 = np.concatenate(embed2,0)

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


def cal_score(dataset):
    scores = compute_metrics_cohere(dataset)
    # Main score is the max of Average Precision (AP)
    main_score = max(scores[short_name]["ap"] for short_name in scores)
    scores["main_score"] = main_score
    return scores


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

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
    print(f"Evaluating...")
    # Evaluate model
    dev_score = cal_score(dev_dataset)
    results["Cohere"].update({"Dev": round(dev_score["main_score"] * 100, 2)})
    test_score = cal_score(test_dataset)
    results["Cohere"].update({"Test": round(test_score["main_score"] * 100, 2)})

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/cohere_xnli.csv")
    print(results_df)