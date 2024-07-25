# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation based on Thai STS-B, Text classification, and Retrieval datasets.

# Motivation
Sentence representation plays a crucial role in NLP downstream tasks such as NLI, text classification, and STS. Recent sentence representation training techniques require NLI or STS datasets. However, no equivalent Thai NLI or STS datasets exist for sentence representation training. 
To address this problem, we create "Thai sentence vector benchmark" to demonstrate that we can train Thai sentence representation without any supervised dataset. 

Our first preliminary results demonstrate that we can train a robust sentence representation model with an unsupervised technique called SimCSE. We show that it is possible to train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the Google Colab (V100), where the performance of [SimCSE-XLM-R](https://huggingface.co/mrp/simcse-model-roberta-base-thai) is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. Our benchmark aims to evaluate the effectiveness of sentence embedding models on Thai zero-shot and transfer learning tasks. The tasks comprise of four tasks: [Semantic ranking on STS-B](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/STS_Evaluation), [text classification (transfer)](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Transfer_Evaluation), [pair classification](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/PairClassification_Evaluation), and [retrieval question answering (QA)](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Retrieval_Evaluation). 

# Install
```
conda create -n thai_sentence_vector_benchmark python==3.11.4
conda activate thai_sentence_vector_benchmark

# Select the appropriate PyTorch version based on your CUDA version
# CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch

pip install -e .
```

# Reproduce the results
```
python scripts/eval_all.py \
--cohere_api_key <YOUR_COHERE_API_KEY> \
--openai_api_key <YOUR_OPENAI_API_KEY>
```

# Usage
```python
from sentence_transformers import SentenceTransformer
from thai_sentence_vector_benchmark.benchmark import ThaiSentenceVectorBenchmark

model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
benchmark = ThaiSentenceVectorBenchmark()
results = benchmark(
  model,
  task_prompts={
    "sts": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "retrieval": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "pair_classification": "Instruct: Retrieve parallel sentences.\nQuery: ",
    "text_classification": "Instruct: Classify the sentiment of the text.\nText: ",
  }
)
>> {
  "STS": {
    "sts_b": {"Spearman_Correlation": float},
    "Average": {"Spearman_Correlation": float},
  },
  "Text_Classification": {
    "wisesight": {"Accuracy": float, "F1": float},
    "wongnai": {"Accuracy": float, "F1": float},
    "generated_reviews": {"Accuracy": float, "F1": float},
    "Average": {"Accuracy": float, "F1": float},
  },
  "Pair_Classification": {
    "xnli": {"AP": float},
    "Average": {"AP": float},
  },
  "Retrieval": {
    "xquad": {"R@1": float, "MRR@10": float},
    "miracl": {"R@1": float, "MRR@10": float},
    "tydiqa": {"R@1": float, "MRR@10": float},
    "Average": {"R@1": float, "MRR@10": float},
  },
  "Average": float,
}
```

# How do we train unsupervised sentence representation?
We provide simple and effective sentence embedding methods that do not require supervised labels (unsupervised learning) as follows: 

## SimCSE
- We use [SimCSE:Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) on multilingual LM models (mBERT, distil-mBERT, XLM-R) and a monolingual model (WangchanBERTa).
- Training data: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ).
- Example: [SimCSE-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb).
- Training Example on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb

## ConGen
- We use the training objective from [ConGen](https://github.com/KornWtp/ConGen) on various PLMs.
- Training data: [scb-mt-en-th-2020](https://medium.com/@onarinlap/scb-mt-en-th-2020-%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B9%81%E0%B8%A3%E0%B8%81%E0%B8%AA%E0%B8%B9%E0%B9%88%E0%B8%AA%E0%B8%B1%E0%B8%87%E0%B9%80%E0%B8%A7%E0%B8%B5%E0%B8%A2%E0%B8%99-machine-translation-%E0%B8%99%E0%B8%B2%E0%B8%99%E0%B8%B2%E0%B8%8A%E0%B8%B2%E0%B8%95%E0%B8%B4%E0%B8%81%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B8%E0%B8%94%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%B9%E0%B8%A5-open-data-fe1c7b9d8271) 
- Example: [ConGen-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/ConGen-Thai.ipynb)

## SCT
- We use the training objective from [SCT](https://github.com/mrpeerat/SCT) on various PLMs.
- Training data: [scb-mt-en-th-2020](https://medium.com/@onarinlap/scb-mt-en-th-2020-%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B9%81%E0%B8%A3%E0%B8%81%E0%B8%AA%E0%B8%B9%E0%B9%88%E0%B8%AA%E0%B8%B1%E0%B8%87%E0%B9%80%E0%B8%A7%E0%B8%B5%E0%B8%A2%E0%B8%99-machine-translation-%E0%B8%99%E0%B8%B2%E0%B8%99%E0%B8%B2%E0%B8%8A%E0%B8%B2%E0%B8%95%E0%B8%B4%E0%B8%81%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B8%E0%B8%94%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%B9%E0%B8%A5-open-data-fe1c7b9d8271) 
- Example: [SCT-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SCT-Thai.ipynb)

### Why do we select these techniques? 
- Easy to train
- Compatible with every model
- Do not require any annotated dataset
- The best sentence representation method (for now) in terms of the performance on STS and downstream tasks (SCT outperformed ConGen and SimCSE in their paper). 

### What about other techniques? 
We also consider other techniques (supervised and unsupervised methods) in this repository. Currently, we have various methods tested on our benchmarks, such as:
- Supervised learning: [sentence-bert](https://github.com/UKPLab/sentence-transformers).
- Multilingual sentence representation alignment: [CL-ReLKT](https://github.com/mrpeerat/CL-ReLKT) (NAACL'22)

# Thai semantic textual similarity benchmark
- We use [STS-B translated ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/stsbenchmark/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate API
- How to evaluate sentence representation: [Easy_Evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/Easy_Evaluation.ipynb)
- How to evaluate sentence representation on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb

| Base Model  | Spearman's Correlation (*100) | Supervised? | Latency(ms) | 
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 44.27  |  |  7.22 &plusmn; 0.53 | 
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 43.95  |  | 11.66 &plusmn; 0.72 |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 63.98  | | 10.95 &plusmn; 0.41 |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 60.95  | | 10.54 &plusmn; 0.33 | 
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 68.28  | | 11.4 &plusmn; 1.01 | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 68.90  | | 10.52 &plusmn; 0.46 |
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 71.35  | | 10.61 &plusmn; 0.62 |
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 74.06  | | 10.64 &plusmn; 0.72 |
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 78.78  | | 10.69 &plusmn; 0.48|
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 77.77  | | 10.86 &plusmn; 0.55| 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 77.89  | | 11.01 &plusmn; 0.62|
| [SCT-Distil-model-phayathaibert-bge-m3]() | 76.71| | |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 79.69  | | 10.79 &plusmn; 0.38 |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 79.20  | | 10.44 &plusmn; 0.5 | 
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 78.90  | | 10.32 &plusmn; 0.31| 
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 76.82  | | 10.91 &plusmn; 0.43|
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 65.37  | :heavy_check_mark: | 9.38 &plusmn; 1.34  |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 80.49  | :heavy_check_mark: |10.93 &plusmn; 0.55 |
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)  | 77.22  | :heavy_check_mark: | 23.5 &plusmn; 3.07|
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 68.03  | :heavy_check_mark: | | | 


# Thai transfer benchmark
- We use [Wisesight](https://huggingface.co/datasets/wisesight_sentiment), [Wongnai](https://huggingface.co/datasets/wongnai_reviews), and [Generated review](https://huggingface.co/datasets/generated_reviews_enth) datasets.
- How to evaluate: [Transfer_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb)

## Wisesight
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 56.12  | 56.60  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 55.86  | 56.65  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 62.07  | 62.76  | 
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 64.17  | 64.39  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 68.59  | 67.73  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 67.47  | 67.62  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 68.51  | 68.97  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 70.80  | 68.60  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 67.73  | 67.75  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 65.78  | 66.17  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 66.64  | 66.94  | 
| [SCT-Distil-model-phayathaibert-bge-m3]() | 67.28 | 67.70| | |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 66.75  | 67.41  |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 67.09  | 67.65  | 
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 67.65  | 68.12  |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 68.62  | 68.92  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.31  | 63.74  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 67.05  | 67.67  | :heavy_check_mark:
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 68.36  | 68.92  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 66.72  | 67.24  | :heavy_check_mark:

## Wongnai
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 34.31  | 35.81  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 37.55  | 38.29  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 40.46  | 38.06  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 40.95  | 37.58  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 37.53  | 38.45  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 42.88  | 44.75  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 47.90  | 47.23  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 54.73  | 49.48  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 46.16  | 47.02  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 48.61  | 44.89  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 48.86  | 48.14  | 
| [SCT-Distil-model-phayathaibert-bge-m3]() | 45.95 | 47.29 | |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 44.95  | 46.57  |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 46.72 | 48.04  | 
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 45.99 | 47.54  |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 47.98  | 49.22  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 37.76  | 40.07  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 45.20  | 46.72  | :heavy_check_mark:
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 51.94  | 52.68  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 46.83  | 48.08  | :heavy_check_mark:


## Generated Review
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 39.11  | 37.27  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 38.72  | 37.56  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 46.27  | 44.22  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 37.37  | 36.72  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 48.76  | 45.14  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 55.93  | 54.19  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 50.39  | 48.65  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 54.90  | 48.36  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 56.76  | 55.50  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 52.33  | 48.41  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 54.35  | 52.23  | 
| [SCT-Distil-model-phayathaibert-bge-m3]() | 58.95 | 57.64| |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 57.93  | 56.66  |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 58.67  | 57.51  | 
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 58.43  | 57.23  |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 59.66  | 58.37  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 50.62  | 48.90  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 57.48  | 56.35  | :heavy_check_mark:
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 59.53  | 58.35  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 57.91  | 56.60  | :heavy_check_mark:

# Thai pair classification benchmark
- We use [XNLI dev and test set](https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip). We drop neutral classes and change from contradiction => 0 and entailment =>1.
- We use the average precision score as the main metric.
- How to evaluate: [XNLI_evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/PairClassification_Evaluation)

| Base Model  | Dev (AP) | Test (AP) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 57.99  | 56.06  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 58.41  | 58.09  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 62.05  | 62.05  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 58.13  | 59.01  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 62.10  | 63.34  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 64.53  | 65.29  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 66.36  | 66.79  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 65.35  | 65.84  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 78.40  | 79.14  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 77.06  | 76.75  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 77.95  | 77.61  | 
| [SCT-Distil-model-phayathaibert-bge-m3]() | 75.18 | 74.83 |  |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 80.68  | 80.98  |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 82.24  | 81.15  | 
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 80.89  | 80.51  |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 76.72  | 76.13  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 65.35  | 64.93  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 84.14  | 84.06  | :heavy_check_mark:
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 79.09  | 79.02  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 60.25  | 61.15  | :heavy_check_mark:


# Thai retrieval benchmark
- We use [XQuAD](https://github.com/google-deepmind/xquad), [MIRACL](https://huggingface.co/datasets/miracl/miracl), and [TyDiQA](https://huggingface.co/datasets/khalidalt/tydiqa-goldp) datasets.
- How to evaluate: [Retrieval_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Retrieval_Evaluation)

## XQuAD
| Base Model  | R@1 | MRR@10 | Supervised? |  Latency(second) | 
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: 
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 18.24  | 27.19  | | 0.61|
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 22.94  | 30.29  | | 1.02 |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 52.02  | 62.94  | | 0.85 |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 53.87  | 65.51  | | 0.81 |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 73.95  | 81.67  | | 0.79 |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 55.29  | 65.23  | | 1.24 |
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 66.30  | 76.14  |  | 1.23 |
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 67.56  | 76.14  |  | 1.19 |
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 68.91  | 78.19  |  | 1.24 |
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 62.27  | 72.53  | | 1.35 |
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 71.43  | 80.18  | | 1.21 |
| [SCT-Distil-model-phayathaibert-bge-m3]() | 80.50 | 86.75 |  |  |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 71.76  | 80.01  | | 1.24 |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 70.92  | 79.59  | | 1.21 |
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 71.85  | 80.33  | | 1.19 |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 85.80  | 90.48  | | 1.3 |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 49.16  | 58.19  | :heavy_check_mark: | 1.05 | 
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 71.26  | 79.63  | :heavy_check_mark: | 1.24 |
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 90.50  | 94.33  | :heavy_check_mark: | 7.22 |
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 82.52 | 87.78  | :heavy_check_mark: | XXX |


## MIRACL
| Base Model  | R@1 | MRR@10 | Supervised? | Latency(second) | 
| ------------- | :-------------: | :-------------: | :-------------: | :-------------:  
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 28.51  | 37.05  | |  4.31 |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 26.19  | 36.11 | | 6.66|
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 34.92  | 47.51  | |  6.17 |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 36.29  | 48.96  | | 6.09 |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 43.25  | 57.28  | | 6.18 |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 28.51  | 40.84  | | 16.29 |
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 35.33  | 48.19  | | 16.0 |
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 37.52  | 51.02  | | 15.8 |
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 40.38  | 51.68  |  | 16.17 |
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 39.43  | 50.61  | | 16.04|
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 45.16  | 56.52  | | 15.82|
| [SCT-Distil-model-phayathaibert-bge-m3]() | 64.80 |  74.46 |  |  |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 43.11  | 55.51  | | 16.4 |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 41.06  | 53.31  | | 15.98 |
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 44.34  | 55.77  | | 15.97 |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 70.40  | 79.33  | | 15.83 |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 17.74  | 27.78  | :heavy_check_mark: | 9.84 |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 38.20  | 49.65  | :heavy_check_mark: | 16.22 |
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 79.67  | 86.68  | :heavy_check_mark: | 91.27 |
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 66.98 | 77.58  | :heavy_check_mark: |  XXX |


## TyDiQA

| Base Model  | R@1 | MRR@10 | Supervised? | Latency(second) | 
| ------------- | :-------------: | :-------------: | :-------------: | :-------------: 
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 44.69  | 51.39  |   |1.6|
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 45.09  | 52.37 | | 2.46 |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 58.06  | 64.72  | | 2.35 | 
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 62.65  | 70.02  | | 2.32 |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 71.43  | 78.16  | | 2.28 |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 49.28  | 58.62  | | 3.15 |
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 58.19  | 68.05  | | 3.21 |
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 63.43  | 71.73  | | 3.21 |
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 56.36  | 65.18  | | 3.3 |
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 56.23  | 65.18  | | 3.18 |
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 58.32  | 67.42  | | 3.21 |
| [SCT-Distil-model-phayathaibert-bge-m3]() | 78.37 |  84.01|  | |
| [ConGen-model-XLMR](https://huggingface.co/kornwtp/ConGen-model-XLMR)  | 60.29  | 68.56  | | 3.28 |
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 59.11  | 67.42  | | 3.19 |
| [ConGen-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-model-phayathaibert)  | 59.24  | 67.69  | | 3.15 |
| [ConGen-BGE_M3-model-phayathaibert](https://huggingface.co/kornwtp/ConGen-BGE_M3-model-phayathaibert)  | 83.36  | 88.29 | | 3.14 |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 32.50  | 42.20  | :heavy_check_mark: | 2.05 |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 54.39  | 63.12  | :heavy_check_mark:|  3.16 |
| [BGE M-3](https://huggingface.co/BAAI/bge-m3)   | 89.12  | 93.43  | :heavy_check_mark: | 20.87 | 
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 85.45 | 90.33  | :heavy_check_mark: | XXX |


# Thank you for the many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)
- [ConGen github](https://github.com/KornWtp/ConGen)
- [SCT github](https://github.com/mrpeerat/SCT)

Acknowledgments:
- Can: proofread
- Charin: proofread + idea

![1_3JJRwT1f2zTK1hx36-qXdg (1)](https://user-images.githubusercontent.com/21156980/139905794-5ce1389f-63e4-4da0-83b8-5b1aa3983222.jpeg)







