import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import cohere

co = cohere.Client('Z0AuLPY1Q2B2n0o3zyntszwvWmBB5MCqnnnuRyNc')



if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

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
    print(f"Evaluating...")
    # Evaluate model
    text1 = thai_sts['text_1'].values.tolist()
    text2 = thai_sts['text_2'].values.tolist()
    label = thai_sts['correlation'].values.tolist()

    bs = 96
    embed1 = []
    embed2 = []
    for i in range(len(text1)//bs+1):
        embed1.append(co.embed(
        texts=thai_sts['text_1'][(i*bs):((i+1)*bs)].values.tolist(),
        model='embed-multilingual-v2.0',
        ).embeddings)
        embed2.append(co.embed(
        texts=thai_sts['text_2'][(i*bs):((i+1)*bs)].values.tolist(),
        model='embed-multilingual-v2.0',
        ).embeddings)

    embed1_final = np.concatenate(embed1, 0)
    embed2_final = np.concatenate(embed2, 0)

    cosine_scores = 1 - (paired_cosine_distances(embed1_final, embed2_final))
    eval_spearman_cosine, _ = spearmanr(label, cosine_scores)

    results["Cohere"]["spearman_cosine"] = round(eval_spearman_cosine * 100, 2)

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/cohere_sts.csv")
    print(results_df)