import numpy as np
from typing import List
from abc import ABC, abstractmethod



class SentenceEncodingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        pass