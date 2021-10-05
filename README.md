# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation on Thai STS-B.

# Motivation
Sentence representations play a crucial role in NLP downstream tasks such as NLI, text classification, and STS.
Recent techniques for train sentence representations require NLI or STS datasets. 
However, Thai NLI or STS datasets are not available to train a sentence representation.
To address this problem, we train a sentence representation with an unsupervised technique call SimCSE.
We can train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the google collab (V100) where the performance of SimCSE-XLM-R is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. We evaluate the Spearman correlation score of a representation's performance on Thai STS-B.

# How do we train unsupervised sentence representation?
- We use [SimCSE:Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) and training models with multilingual LM models (mBERT, distil-mBERT, XLM-R) 
- Training data: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ)
- Example: [SimCSE-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb)

# Why SimCSE?
- Easy to train
- Work with every model
- Not require any label datasets
- The performance of XLM-R (unsupervised) and m-Distil-BERT (train on > 1B sentences) is similar (<1% correlation)

# How to train supervised?
- We recommend [sentence-bert](https://github.com/UKPLab/sentence-transformers), which you can train with NLI, STS, triplet, contrastive, etc.

# Benchmark
- We use [STS-B translation ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate.
- How to evaluate sentence representation: [SentEval.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb) 

| Base Model  | Spearman' Cor (*100) | Supervised? |
| ------------- | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 38.84  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 39.26  | 
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 62.60  | 
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.50  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 80.11  | :heavy_check_mark:

# Google Colab
- Evaluation: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb
- Training Example: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb

You can pull requests with results to show your model in the benchmark table!!!!.

# Thank you many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)
