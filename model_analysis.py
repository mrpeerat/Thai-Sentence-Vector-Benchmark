import os
import requests
from tqdm import tqdm
from tokenizers import Tokenizer
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

import cohere
co = cohere.Client('Z0AuLPY1Q2B2n0o3zyntszwvWmBB5MCqnnnuRyNc')

tokenizer_url = "https://storage.googleapis.com/cohere-public/tokenizers/embed-multilingual-v2.0.json"
# tokenizer_url = "https://storage.googleapis.com/cohere-public/tokenizers/embed-multilingual-v3.0.json"
response = requests.get(tokenizer_url)  


def get_model_size(model):
    # Model size (MB)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    return size_all_mb, num_params


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

    models_to_test = {
        "XLMR": "FacebookAI/xlm-roberta-base",
        "WangchanBERTa": "airesearch/wangchanberta-base-att-spm-uncased",
        "PhayaThaiBERT ": "clicknext/phayathaibert",
        "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "distiluse-base-multilingual-cased-v2": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    }

    for model_name, model_path in tqdm(models_to_test.items()):
        print(f"Evaluating {model_name}...")
        # Vocabulary size
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # Get number of thai words in vocab
        thai_vocab_count = 0
        for word_str, word_id in tokenizer.get_vocab().items():
            # Check if the word contains Thai characters
            if any([ord("ก") <= ord(c) <= ord("ฮ") for c in word_str]):
                # print(f"Thai word: {word_str}")
                thai_vocab_count += 1
        print(f"Vocab size: {len(tokenizer.get_vocab())}")
        print(f"Thai Vocab size: {thai_vocab_count}")
        # Model size
        # Load model
        model = AutoModel.from_pretrained(model_path)
        model_size, num_params = get_model_size(model)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Number of parameters: {num_params}")

    # Cohere API
    print(f"Evaluating Cohere...")
    # Load Cohere tokenizer
    tokenizer = Tokenizer.from_str(response.text)
    # Get number of thai words in vocab
    thai_vocab_count = 0
    for word_str, word_id in tokenizer.get_vocab().items():
        # Check if the word contains Thai characters
        if any([ord("ก") <= ord(c) <= ord("ฮ") for c in word_str]):
            # print(f"Thai word: {word_str}")
            thai_vocab_count += 1
    print(f"Total Vocab size: {len(tokenizer.get_vocab())}")
    print(f"Thai Vocab size: {thai_vocab_count}")