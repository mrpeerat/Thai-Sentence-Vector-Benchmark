import numpy as np
from typing import List
from tqdm import trange
from FlagEmbedding import BGEM3FlagModel
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class BGEModel(SentenceEncodingModel):
    def __init__(self, model_name: str, use_fp16: bool = False, max_seq_length: int = 1024):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = BGEM3FlagModel(model_name,  use_fp16=use_fp16) 

    def encode(self, texts: List[str], batch_size: int = 1024, **kwargs):
        embeds = self.model.encode(
            sentences=texts,
            batch_size=batch_size,
            max_length=self.max_seq_length,

        )["dense_vecs"]
        return embeds