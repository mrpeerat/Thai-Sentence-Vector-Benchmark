import numpy as np
from typing import List
from tqdm import trange
from FlagEmbedding import BGEM3FlagModel
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class BGEModel(SentenceEncodingModel):
    def __init__(self, model_name: str, use_fp16: bool = False):
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name,  use_fp16=use_fp16) 

    def encode(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False, **kwargs):
        embeds = []
        for i in trange(len(texts) // batch_size + 1, disable=not show_progress_bar):
            embed = self.model.encode(
                sentences=texts[(i * batch_size) : ((i + 1) * batch_size)],
                batch_size=batch_size,

            )["dense_vecs"]
            embeds.append(embed)
        return np.concatenate(embeds)