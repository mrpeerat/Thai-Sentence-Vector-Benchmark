import cohere
import numpy as np
from typing import List
from tqdm import trange
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class CohereV2Model(SentenceEncodingModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = cohere.Client(api_key)

    def encode(self, texts: List[str], batch_size: int = 1024, show_progress_bar: bool = False, **kwargs):
        embeds = []
        for i in trange(len(texts) // batch_size + 1, disable=not show_progress_bar):
            embed = self.client.embed(
                texts=texts[(i * batch_size) : ((i + 1) * batch_size)],
                model=self.model_name,
            ).embeddings
            embeds.append(embed)
        return np.concatenate(embeds)
    

class CohereV3Model(SentenceEncodingModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = cohere.Client(api_key)

    def encode(self, texts: List[str], batch_size: int = 1024, show_progress_bar: bool = False, input_type: str = "search_document", **kwargs):
        embeds = []
        for i in trange(len(texts) // batch_size + 1, disable=not show_progress_bar):
            embed = self.client.embed(
                texts=texts[(i * batch_size) : ((i + 1) * batch_size)],
                model=self.model_name,
                input_type=input_type,
            ).embeddings
            embeds.append(embed)
        return np.concatenate(embeds)