# Thai-Sentence-Vector-Benchmark
Benchmark for Thai sentence representation on Thai STS-B and Transfer (text classification) datasets.

# Motivation
Sentence representation plays a crucial role in NLP downstream tasks such as NLI, text classification, and STS. Recent sentence representation training techniques require NLI or STS datasets.  However, there are no equivalent Thai NLI or STS datasets for sentence representation training.
To address this problem, we train a sentence representation model with an unsupervised technique called SimCSE.

We show that it is possible to train SimCSE with 1.3 M sentences from Wikipedia within 2 hours on the Google Colab (V100) where the performance of [SimCSE-XLM-R](https://huggingface.co/mrp/simcse-model-roberta-base-thai) is similar to [mDistil-BERT<-mUSE](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) (train on > 1B sentences).

Moreover, we provide the Thai sentence vector benchmark. We evaluate the Spearman correlation score of the sentence representations’ performance on Thai STS-B (translated version of [STS-B](https://github.com/facebookresearch/SentEval)). In addition, we evalute the accuracy and F1 scores of Thai text classification datasets [HERE]().

# How do we train unsupervised sentence representation?
- We use [SimCSE:Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf) on multilingual LM models (mBERT, distil-mBERT, XLM-R) and a monolingual model (WangchanBERTa).
- Training data: [Thai Wikipedia](https://github.com/PyThaiNLP/ThaiWiki-clean/releases/tag/20210620?fbclid=IwAR2_CtHJ_6od9z5-0hsolwcNYJH03e5qk_XXkoxDpOQivmo8QreYFQS3JuQ).
- Example: [SimCSE-Thai.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb).
- Training Example on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SimCSE-Thai.ipynb
## Why SimCSE?
- Easy to train
- Compatible with every model
- Does not require any annotated dataset
- The performance of XLM-R (unsupervised) and m-Distil-BERT (supervised and trained on > 1B sentences) are similar (1% difference in correlation).

# What about Supervised Learning?
- We recommend [sentence-bert](https://github.com/UKPLab/sentence-transformers), which you can train with NLI, STS, triplet loss, contrastive loss, etc.

# Thai semantic textual similarity benchmark
- We use [STS-B translated ver.](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/sts-test_th.csv) in which we translate STS-B from [SentEval](https://github.com/facebookresearch/SentEval) by using google-translate.
- How to evaluate sentence representation: [SentEval.ipynb](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb) 
- For the easy-to-implement version: [Easy_Evaluation.ipynb]()
- How to evaluate sentence representation on Google Colab: https://colab.research.google.com/github/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/SentEval.ipynb

| Base Model  | Spearman's Correlation (*100) | Supervised? |
| ------------- | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | 38.84  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | 39.26  | 
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | 62.60  | 
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | 63.50  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | 80.11  | :heavy_check_mark:

# Thai transfer benchmark
- We use [Wisesight](https://huggingface.co/datasets/wisesight_sentiment), [Wongnai](https://huggingface.co/datasets/wongnai_reviews), and [Generated review](https://huggingface.co/datasets/generated_reviews_enth) datasets.
- How to evaluate: [Transfer_Evaluation]()

| Base Model  | Acc (*100) | F1 (*100, weighted) | Supervised? |
| ------------- | :-------------: | :-------------: | :-------------: |
| [simcse-model-distil-m-bert](https://huggingface.co/mrp/simcse-model-distil-m-bert)  | XX.XX  | XX.XX  |
| [simcse-model-m-bert-thai-cased](https://huggingface.co/mrp/simcse-model-m-bert-thai-cased)  | XX.XX  | XX.XX  |
| [simcse-model-roberta-base-thai](https://huggingface.co/mrp/simcse-model-roberta-base-thai)  | XX.XX  | XX.XX  | 
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)  | XX.XX  | XX.XX  | :heavy_check_mark:


# Thank you many codes from
- [Sentence-transformer (Sentence-BERT)](https://github.com/UKPLab/sentence-transformers)
- [SimCSE github](https://github.com/princeton-nlp/SimCSE)

Acknowledgments:
- Can: proofread
- Charin: proofread + idea

![1_3JJRwT1f2zTK1hx36-qXdg (1)](https://user-images.githubusercontent.com/21156980/139905794-5ce1389f-63e4-4da0-83b8-5b1aa3983222.jpeg)
