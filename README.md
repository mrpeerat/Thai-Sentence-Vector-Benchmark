# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation on Thai STS-B, Text classification, and Retrieval datasets.

# Motivation
Sentence representation plays a crucial role in NLP downstream tasks such as NLI, text classification, and STS. Recent sentence representation training techniques require NLI or STS datasets. However, no equivalent Thai NLI or STS datasets exist for sentence representation training. 
To address this problem, we create "Thai sentence vector benchmark" to demonstrate that we can train Thai sentence representation without any supervised datasets. 

Our first preliminary results demonstrate that we can train a robust sentence representation model with an unsupervised technique called SimCSE. We show that it is possible to train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the Google Colab (V100), where the performance of [SimCSE-XLM-R](https://huggingface.co/mrp/simcse-model-roberta-base-thai) is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. Our benchmark aims to evaluate the effectiveness of sentence embedding models on Thai zero-shot and transfer learning tasks. The tasks comprise of four tasks: [Semantic ranking on STS-B](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/STS_Evaluation), [text classification (transfer)](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Transfer_Evaluation), [pair classification](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/PairClassification_Evaluation), and [retrieval question answering (QA)](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Retrieval_Evaluation). 


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
- Does not require any annotated datasets
- The best sentence representation method (for now) in terms of the performance on STS and downstream tasks (SCT outperformed ConGen and SimCSE in their paper). 

### What about other techniques? 
We also consider other techniques (supervised and unsupervised methods) in this repository. Currently, we have various methods tested on our benchmarks, such as:
- Supervised learning: [sentence-bert](https://github.com/UKPLab/sentence-transformers).
- Multilingual sentence representation alignment: [CL-ReLKT](https://github.com/mrpeerat/CL-ReLKT) (NAACL'22)

# Thai semantic textual similarity benchmark
- We use [STS-B translated ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/stsbenchmark/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate API
- How to evaluate sentence representation: [Easy_Evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/Easy_Evaluation.ipynb)
- How to evaluate sentence representation on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb

| Base Model  | Spearman's Correlation (*100) | Supervised? |
| ------------- | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 44.27  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 43.95  |  
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 60.95  | 
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 68.28  | 
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 63.98  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 68.90  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 71.35  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 74.06  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 78.78  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 77.77  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 77.89  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 79.20  | 
| [ConGen-model-phayathaibert]()  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 65.37  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 80.49  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 68.03  | :heavy_check_mark:


# Thai transfer benchmark
- We use [Wisesight](https://huggingface.co/datasets/wisesight_sentiment), [Wongnai](https://huggingface.co/datasets/wongnai_reviews), and [Generated review](https://huggingface.co/datasets/generated_reviews_enth) datasets.
- How to evaluate: [Transfer_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb)

## Wisesight
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 56.12  | 56.60  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 55.86  | 56.65  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 64.17  | 64.39  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 68.59  | 67.73  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 62.07  | 62.76  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 67.47  | 67.62  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 68.51  | 68.97  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 70.80  | 68.60  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 67.73  | 67.75  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 65.78  | 66.17  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 66.64  | 66.94  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 67.09  | 67.65  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.31  | 63.74  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 67.05  | 67.67  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 67.13  | 67.53  | :heavy_check_mark:

## Wongnai
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 34.31  | 35.81  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 37.55  | 38.29  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 40.95  | 37.58  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 37.53  | 38.45  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 40.46  | 38.06  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 42.88  | 44.75  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 47.90  | 47.23  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 54.73  | 49.48  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 46.16  | 47.02  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 48.61  | 44.89  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 48.86  | 48.14  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 46.72 | 48.04  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 37.76  | 40.07  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 45.20  | 46.72  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | xx.xx  | xx.xx  | :heavy_check_mark:


## Generated Review
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 39.11  | 37.27  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 38.72  | 37.56  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 37.37  | 36.72  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 48.76  | 45.14  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 46.27  | 44.22  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 55.93  | 54.19  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 50.39  | 48.65  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 54.90  | 48.36  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 56.76  | 55.50  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 52.33  | 48.41  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 54.35  | 52.23  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 58.67  | 57.51  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 50.62  | 48.90  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 57.48  | 56.35  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | xx.xx  | xx.xx  | :heavy_check_mark:

# Thai pair classification benchmark
- We use [XNLI dev and test set](https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip). We drop neutral classes and change from contradiction => 0 and entailment =>1.
- We use the average precision score as the main metric.
- How to evaluate: [XNLI_evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/PairClassification_Evaluation)

| Base Model  | Dev (AP) | Test (AP) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 57.99  | 56.06  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 58.41  | 58.09  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 58.13  | 59.01  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 62.10  | 63.34  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 62.05  | 62.05  | 
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 64.53  | 65.29  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 66.36  | 66.79  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 65.35  | 65.84  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 78.40  | 79.14  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 77.06  | 76.75  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 77.95  | 77.61  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 82.24  | 81.15  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 65.35  | 64.93  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 84.14  | 84.06  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 60.25  | 61.15  | :heavy_check_mark:


# Thai retrieval benchmark
- We use [XQuAD](https://github.com/google-deepmind/xquad), [MIRACL](https://huggingface.co/datasets/miracl/miracl), and [TyDiQA](https://huggingface.co/datasets/khalidalt/tydiqa-goldp) datasets.
- How to evaluate: [Retrieval_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Retrieval_Evaluation)

## XQuAD
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 18.24  | 27.19  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 22.94  | 30.29  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 53.87  | 65.51  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 73.95  | 81.67  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 52.02  | 62.94  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 55.29  | 65.23  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 66.30  | 76.14  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 67.56  | 76.14  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 68.91  | 78.19  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 62.27  | 72.53  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 71.43  | 80.18  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 70.92  | 79.59  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 49.16  | 58.19  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 71.26  | 79.63  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 82.52 | 87.78  | :heavy_check_mark:

## MIRACL
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 28.51  | 37.05  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 26.19  | 36.11  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 36.29  | 48.96  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 43.25  | 57.28  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 34.92  | 47.51  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 28.51  | 40.84  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 35.33  | 48.19  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 37.52  | 51.02  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 40.38  | 51.68  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 39.43  | 50.61  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 45.16  | 56.52  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 41.06  | 53.31  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 17.74  | 27.78  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 38.20  | 49.65  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 66.98 | 77.58  | :heavy_check_mark:

## TyDiQA
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/kornwtp/simcse-model-distil-m-bert)  | 44.69  | 51.39  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/kornwtp/simcse-model-m-bert-thai-cased)  | 45.09  | 52.37  |
| [simcse-model-wangchanberta](https://huggingface.co/kornwtp/simcse-model-wangchanberta)  | 62.65  | 70.02  |
| [simcse-model-phayathaibert](https://huggingface.co/kornwtp/simcse-model-phayathaibert)  | 71.43  | 78.16  |
| [simcse-model-XLMR](https://huggingface.co/kornwtp/simcse-model-XLMR)  | 58.06  | 64.72  |
| [SCT-model-XLMR](https://huggingface.co/kornwtp/SCT-model-XLMR)  | 49.28  | 58.62  | 
| [SCT-model-wangchanberta](https://huggingface.co/kornwtp/SCT-model-wangchanberta)  | 58.19  | 68.05  | 
| [SCT-model-phayathaibert](https://huggingface.co/kornwtp/SCT-model-phayathaibert)  | 63.43  | 71.73  | 
| [SCT-Distil-model-XLMR](https://huggingface.co/kornwtp/SCT-KD-model-XLMR)  | 56.36  | 65.18  | 
| [SCT-Distil-model-wangchanberta](https://huggingface.co/kornwtp/SCT-KD-model-wangchanberta)  | 56.23  | 65.18  | 
| [SCT-Distil-model-phayathaibert](https://huggingface.co/kornwtp/SCT-KD-model-phayathaibert)  | 58.32  | 67.42  | 
| [ConGen-model-wangchanberta](https://huggingface.co/kornwtp/ConGen-model-wangchanberta)  | 59.11  | 67.42  | 
| [ConGen-model-phayathaibert]()  | XX.XX  | XX.XX  |
| [ConGen-model-XLMR]()  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 32.50  | 42.20  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 54.39  | 63.12  | :heavy_check_mark:
| [Cohere-embed-multilingual-v2.0](https://cohere.com/embeddings)  | 85.45 | 90.33  | :heavy_check_mark:


# Thank you for the many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)

Acknowledgments:
- Can: proofread
- Charin: proofread + idea

![1_3JJRwT1f2zTK1hx36-qXdg (1)](https://user-images.githubusercontent.com/21156980/139905794-5ce1389f-63e4-4da0-83b8-5b1aa3983222.jpeg)
