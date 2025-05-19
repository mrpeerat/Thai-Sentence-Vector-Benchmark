import os
import numpy as np
import matplotlib.pyplot as plt


target_models = {
    "xlm-roberta-base": "XLMR-base",
    "xlm-roberta-large": "XLMR-large",
    "wangchanberta-base-att-spm-uncased": "WangchanBERTa",
    "phayathaibert": "PhayaThaiBERT",
    "e5-mistral-7b-instruct": "E5 Mistral 7B Instruct",
    "gte-Qwen2-7B-instruct": "gte-Qwen2 7B Instruct",
    "GritLM-7B": "GritLM 7B",
    "Meta-Llama-3-8B-Instruct": "Llama3 8B Instruct",
    "llama-3-typhoon-v1.5-8b-instruct": "Typhoon 8B Instruct",
    "simcse-model-phayathaibert": "SimCSE-PhayaThaiBERT",
    "SCT-model-phayathaibert": "SCT-PhayaThaiBERT",
    "ConGen-model-phayathaibert": "ConGen-PhayaThaiBERT",
    "SCT-KD-model-phayathaibert": "SCT-KD-PhayaThaiBERT",
    "paraphrase-multilingual-mpnet-base-v2": "MPNet-multilingual",
    "distiluse-base-multilingual-cased-v2": "DistilUSE-multilingual",
    "bge-m3": "BGE-M3",
}

annotation_offsets = {
    "XLMR-base": [820, 40.5],
    "XLMR-large": [2300, 40.5],
    "WangchanBERTa": [1100, 32.5],
    "PhayaThaiBERT": [1400, 54.6],
    "E5 Mistral 7B Instruct": [20500, 76.5],
    "gte-Qwen2 7B Instruct": [14500, 41],
    "GritLM 7B": [82000, 35.5],
    "Llama3 8B Instruct": [12000, 55],
    "Typhoon 8B Instruct": [16000, 61],
    "SimCSE-PhayaThaiBERT": [1600, 59.2],
    "SCT-PhayaThaiBERT": [1730, 62.4],
    "ConGen-PhayaThaiBERT": [2100, 68],
    "SCT-KD-PhayaThaiBERT": [2000, 64.5],
    "MPNet-multilingual": [670, 68.5],
    "DistilUSE-multilingual": [660, 47.5],
    "BGE-M3": [2350, 75.2],
}

model_sizes = {
    "XLMR-base": 279,
    "XLMR-large": 561,
    "WangchanBERTa": 106,
    "PhayaThaiBERT": 278,
    "E5 Mistral 7B": 7110,
    "gte-Qwen2 7B": 7610,
    "GritLM 7B": 7240,
    "Llama3 8B": 8030,
    "MPNet-multilingual": 278,
    "DistilUSE-multilingual": 135,
    "BGE-M3": 570,
}

embedding_sizes = {
    "XLMR-base": 768,
    "XLMR-large": 1024,
    "WangchanBERTa": 768,
    "PhayaThaiBERT": 768,
    "E5 Mistral 7B": 4096,
    "gte-Qwen2 7B": 3584,
    "GritLM 7B": 4096,
    "Llama3 8B": 4096,
    "MPNet-multilingual": 768,
    "DistilUSE-multilingual": 512,
    "BGE-M3": 1024,
}


def get_model_size(model_name):
    if "xlmr" in model_name.lower():
        return model_sizes["XLMR-base"]
    elif "wangchanberta" in model_name.lower():
        return model_sizes["WangchanBERTa"]
    elif "phayathaibert" in model_name.lower():
        return model_sizes["PhayaThaiBERT"]
    elif "e5" in model_name.lower():
        return model_sizes["E5 Mistral 7B"]
    elif "gte-qwen2" in model_name.lower():
        return model_sizes["gte-Qwen2 7B"]
    elif "gritlm" in model_name.lower():
        return model_sizes["GritLM 7B"]
    elif "llama" in model_name.lower():
        return model_sizes["Llama3 8B"]
    elif "typhoon" in model_name.lower():
        return model_sizes["Llama3 8B"]
    elif "mpnet" in model_name.lower():
        return model_sizes["MPNet-multilingual"]
    elif "distiluse" in model_name.lower():
        return model_sizes["DistilUSE-multilingual"]
    elif "bge-m3" in model_name.lower():
        return model_sizes["BGE-M3"]
    

def get_embedding_size(model_name):
    if "xlmr" in model_name.lower():
        return embedding_sizes["XLMR-base"]
    elif "wangchanberta" in model_name.lower():
        return embedding_sizes["WangchanBERTa"]
    elif "phayathaibert" in model_name.lower():
        return embedding_sizes["PhayaThaiBERT"]
    elif "e5" in model_name.lower():
        return embedding_sizes["E5 Mistral 7B"]
    elif "gte-qwen2" in model_name.lower():
        return embedding_sizes["gte-Qwen2 7B"]
    elif "gritlm" in model_name.lower():
        return embedding_sizes["GritLM 7B"]
    elif "llama" in model_name.lower():
        return embedding_sizes["Llama3 8B"]
    elif "typhoon" in model_name.lower():
        return embedding_sizes["Llama3 8B"]
    elif "mpnet" in model_name.lower():
        return embedding_sizes["MPNet-multilingual"]
    elif "distiluse" in model_name.lower():
        return embedding_sizes["DistilUSE-multilingual"]
    elif "bge-m3" in model_name.lower():
        return embedding_sizes["BGE-M3"]


if __name__ == "__main__":
    folder_path = "./outputs"
    
    models = []
    sizes = {}
    runtimes = {}
    performances = {}
    for file in os.listdir(folder_path):
        if file.endswith(".out"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                for idx, line in enumerate(f):
                    if idx == 0:
                        model_name = line.split("/")[-1].strip()
                        print(model_name)
                        if model_name not in target_models:
                            continue
                        model_name = target_models[model_name]
                        sizes[model_name] = get_model_size(model_name)
                        models.append(model_name)
                    elif line.startswith("{'sts_b': {'Spearman_Correlation': "):
                        sts_result = float(line.replace("{'sts_b': {'Spearman_Correlation': ", "").replace("}}", "").strip())
                    elif line.startswith("{'wisesight': {'Accuracy': "):
                        text_clf_results = []
                        for split_text in line.split("}, "):
                            acc_result, f1_result = split_text.split(", ")
                            acc_result = float(acc_result.split("'Accuracy': ")[-1])
                            f1_result = float(f1_result.split("'F1': ")[-1].split("}}")[0])
                            text_clf_results.append([acc_result, f1_result])
                    elif line.startswith("{'xnli': {'AP': "):
                        pair_clf_result = float(line.replace("{'xnli': {'AP': ", "").replace("}}", "").strip())
                    elif line.startswith("{'xquad': {'R@1': "):
                        retrieval_results = []
                        for split_text in line.split("}, "):
                            r1_result, r5_result, r10_result, mrr_result = split_text.split(", ")
                            r1_result = float(r1_result.split("'R@1': ")[-1])
                            mrr_result = float(mrr_result.split("'MRR@10': ")[-1].split("}}")[0])
                            retrieval_results.append([r1_result, mrr_result])
                    elif line.startswith("Elapsed Time: "):
                        elapsed_time = float(line.replace("Elapsed Time: ", "").replace(" seconds", "").strip())

            text_clf_result = [round(np.mean([acc for acc, f1 in text_clf_results]), 2), round(np.mean([f1 for acc, f1 in text_clf_results]), 2)]
            retrieval_result = [round(np.mean([r for r, mrr in retrieval_results]), 2), round(np.mean([mrr for r, mrr in retrieval_results]), 2)]
            performance = round((sts_result + text_clf_result[0] + text_clf_result[1] + pair_clf_result + retrieval_result[0] + retrieval_result[1]) / 6, 2)
            performances[model_name] = performance
            runtimes[model_name] = elapsed_time

            # with open(os.path.join(folder_path, file), "r") as f:
            #     for idx, line in enumerate(f):
            #         if idx == 0:
            #             model_name = line.split("/")[-1].strip()
            #         elif line.startswith("Elapsed Time: "):
            #             elapsed_time = float(line.replace("Elapsed Time: ", "").replace(" seconds", "").strip())
            #             runtimes[model_name] = elapsed_time

    # print sorted runtimes and performances
    # print(models)
    # print(sizes)
    sizes = [sizes[model_name] for model_name in models]
    runtimes = [runtimes[model_name] for model_name in models]
    performances = [performances[model_name] for model_name in models]
    # print(runtimes)
    # print(performances)
    print([(model, score) for model, score in zip(models, performances)])

    # plot runtimes using scatter plot
    # x-axis: runtime (log-scale), y-axis: performance
    # color: performance (as a heatmap), size: model size

    # label each point with model name
    # font_size = 11
    font_size = 13
    plt.style.use("seaborn-v0_8-pastel")
    plt.figure(figsize=(10, 5))
    for i, txt in enumerate(models):
        if annotation_offsets[txt][0] == 0 and annotation_offsets[txt][1] == 0:
            plt.annotate(txt, (runtimes[i], performances[i]), fontsize=font_size)
        else:
            plt.annotate(txt, (annotation_offsets[txt][0], annotation_offsets[txt][1]), fontsize=font_size)
    plt.scatter(runtimes, performances, c=performances, s=sizes, cmap="viridis", alpha=0.5)
    plt.grid()
    plt.xlim(600, 140000)
    plt.ylim(25, 85)
    plt.xscale("log")
    plt.xlabel("Runtime (seconds)", fontdict={"size": font_size})
    plt.ylabel("Thai Sentence Embedding Benchmark Score", fontdict={"size": font_size})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()