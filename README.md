# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation on Thai STS-B and Transfer (text classification) datasets.

# Motivation
Sentence representation plays a crucial role in NLP downstream tasks such as NLI, text classification, and STS. Recent sentence representation training techniques require NLI or STS datasets. However, no equivalent Thai NLI or STS datasets exist for sentence representation training. 
To address this problem, we create "Thai sentence vector benchmark" to demonstrate that we can train Thai sentence representation without any supervised datasets. 

Our first preliminary results demonstrate that we can train a robust sentence representation model with an unsupervised technique called SimCSE. We show that it is possible to train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the Google Colab (V100), where the performance of [SimCSE-XLM-R](https://huggingface.co/mrp/simcse-model-roberta-base-thai) is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. We evaluate the Spearman correlation score of the sentence representations' performance on Thai STS-B (translated version of [STS-B](https://github.com/facebookresearch/SentEval)). In addition, we evaluate the accuracy and F1 scores of Thai text classification datasets [HERE](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb).

# How do we train unsupervised sentence representation?

## SimCSE
- We use [SimCSE:Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) on multilingual LM models (mBERT, distil-mBERT, XLM-R) and a monolingual model (WangchanBERTa).
- Training data: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ).
- Example: [SimCSE-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb).
- Training Example on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb

## ConGen
- We use the training objective from [ConGen](https://github.com/KornWtp/ConGen) on various PLMs.
- Training data: [scb-mt-en-th-2020](https://medium.com/@onarinlap/scb-mt-en-th-2020-%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B9%81%E0%B8%A3%E0%B8%81%E0%B8%AA%E0%B8%B9%E0%B9%88%E0%B8%AA%E0%B8%B1%E0%B8%87%E0%B9%80%E0%B8%A7%E0%B8%B5%E0%B8%A2%E0%B8%99-machine-translation-%E0%B8%99%E0%B8%B2%E0%B8%99%E0%B8%B2%E0%B8%8A%E0%B8%B2%E0%B8%95%E0%B8%B4%E0%B8%81%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B8%E0%B8%94%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%B9%E0%B8%A5-open-data-fe1c7b9d8271) with 
- Example: [ConGen-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/ConGen-Thai.ipynb)

## SCT
- We use the training objective from [SCT](https://github.com/mrpeerat/SCT) on various PLMs.
- Training data: [scb-mt-en-th-2020](https://medium.com/@onarinlap/scb-mt-en-th-2020-%E0%B8%81%E0%B9%89%E0%B8%B2%E0%B8%A7%E0%B9%81%E0%B8%A3%E0%B8%81%E0%B8%AA%E0%B8%B9%E0%B9%88%E0%B8%AA%E0%B8%B1%E0%B8%87%E0%B9%80%E0%B8%A7%E0%B8%B5%E0%B8%A2%E0%B8%99-machine-translation-%E0%B8%99%E0%B8%B2%E0%B8%99%E0%B8%B2%E0%B8%8A%E0%B8%B2%E0%B8%95%E0%B8%B4%E0%B8%81%E0%B8%B1%E0%B8%9A%E0%B8%8A%E0%B8%B8%E0%B8%94%E0%B8%82%E0%B9%89%E0%B8%AD%E0%B8%A1%E0%B8%B9%E0%B8%A5-open-data-fe1c7b9d8271) with 
- Example: [SCT-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SCT-Thai.ipynb)

### Why do we use these techniques? 
- Easy to train
- Compatible with every model
- Does not require any annotated dataset
- The best sentence representation method (for now) in terms of the performance on STS and downstream tasks (SCT outperformed ConGen and SimCSE in their paper). 

### What about other techniques? 
We also consider other techniques in this repository. Currently, we have many methods tested on our benchmarks :)
- Supervised learning: [sentence-bert](https://github.com/UKPLab/sentence-transformers).
- Multilingual sentence representation alignment: [CL-ReLKT](https://github.com/mrpeerat/CL-ReLKT) (NAACL'22)

# Thai semantic textual similarity benchmark
- We use [STS-B translated ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate API.
- How to evaluate sentence representation: [SentEval.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/SentEval.ipynb) 
- For the easy-to-implement version: [Easy_Evaluation.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/STS_Evaluation/Easy_Evaluation.ipynb)
- How to evaluate sentence representation on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb

| Base Model  | Spearman's Correlation (*100) | Supervised? |
| ------------- | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 38.84  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 39.26  | 
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | 52.66  | 
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 62.60  | 
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | 66.21  |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | 76.56  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.50  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 80.11  | :heavy_check_mark:


# Thai transfer benchmark
- We use [Wisesight](https://huggingface.co/datasets/wisesight_sentiment), [Wongnai](https://huggingface.co/datasets/wongnai_reviews), and [Generated review](https://huggingface.co/datasets/generated_reviews_enth) datasets.
- How to evaluate: [Transfer_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/Transfer_Evaluation.ipynb)

## Wisesight
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 55.37  | 55.92  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 56.72  | 56.95  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | 62.22  | 63.06  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 61.96  | 62.48  | 
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | 65.07  | 65.28  |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | 67.84  | 68.31  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.31  | 63.74  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 67.05  | 67.67  | :heavy_check_mark:

## Wongnai
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 36.56  | 37.31  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 39.63  | 38.96  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | 41.38  | 37.46  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 44.11  | 40.34  | 
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | 41.32  | 41.57 |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | 47.22  | 48.63  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 37.76  | 40.07  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 45.20  | 46.72  | :heavy_check_mark:


## Generated Review
| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 38.29  | 37.10  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 38.30  | 36.63  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | 46.63  | 42.60  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 42.93  | 42.81  | 
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | 49.81  | 47.94 |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | 58.00 | 56.80  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 50.62  | 48.90  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 57.48  | 56.35  | :heavy_check_mark:


# Thai retrieval benchmark
- We use [XQuAD](https://github.com/google-deepmind/xquad), [MIRACL](https://huggingface.co/datasets/miracl/miracl), and [TyDiQA](https://huggingface.co/datasets/khalidalt/tydiqa-goldp) datasets.
- How to evaluate: [Retrieval_Evaluation](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/tree/main/Retrieval_Evaluation)

## XQuAD
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | XX.XX  | XX.XX  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | XX.XX  | XX.XX  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | XX.XX  | XX.XX  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:

## MIRACL
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | XX.XX  | XX.XX  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | XX.XX  | XX.XX  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | XX.XX  | XX.XX  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:

## TyDiQA
| Base Model  | R@1 | MRR@10 | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | XX.XX  | XX.XX  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | XX.XX  | XX.XX  |
| [simcse-model-wangchanberta](https://huggingface.co/mrp/simcse-model-wangchanberta)  | XX.XX  | XX.XX  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-simcse-model-roberta-base-thai](https://huggingface.co/kornwtp/ConGen-simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  |
| [ConGen-paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/kornwtp/ConGen-paraphrase-multilingual-mpnet-base-v2/tree/main)  | XX.XX  | XX.XX  |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:


# Thank you for the many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)

Acknowledgments:
- Can: proofread
- Charin: proofread + idea

![1_3JJRwT1f2zTK1hx36-qXdg (1)](https://user-images.githubusercontent.com/21156980/139905794-5ce1389f-63e4-4da0-83b8-5b1aa3983222.jpeg)
