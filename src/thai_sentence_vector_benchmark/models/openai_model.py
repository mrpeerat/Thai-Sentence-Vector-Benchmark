import time
import tiktoken
import numpy as np
from typing import List
from tqdm import trange
from openai import OpenAI
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class OpenAIModel(SentenceEncodingModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, texts: List[str], batch_size: int = 1024, show_progress_bar: bool = False, max_length: int = 8192, delay: int = 60, **kwargs):
        embeds = []
        for i in trange(len(texts) // batch_size + 1, disable=not show_progress_bar):
            batch = texts[i : i + batch_size]
            # Limit input sequence length to not exceed 8192 tokens
            tokens = self.encoding.encode_batch(batch)
            batch = [token[:max_length] for token in tokens]
            embed = np.array(
                [d.embedding for d in self.client.embeddings.create(input=batch, model=self.model_name).data]
            )
            embeds.append(embed)
            # Delay to avoid rate limit
            time.sleep(delay)
        return np.concatenate(embeds)