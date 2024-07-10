from typing import List
from sentence_transformers import SentenceTransformer
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class SentenceTransformerModel(SentenceEncodingModel):
    def __init__(self, model_name: str, max_seq_length: int = None):
        self.model_name = model_name
        # self.model = SentenceTransformer(
        #     model_name, tokenizer_kwargs={"use_fast": False}
        # )
        self.model = SentenceTransformer(
            model_name
        )
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(self, texts: List[str], batch_size: int = 1024, show_progress_bar: bool = False, normalize_embeddings: bool = False, **kwargs):
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings
        )