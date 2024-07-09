import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel



class STSBenchmark:
    def __init__(self, data_dir: str = "./data/stsbenchmark/sts-test_th.csv"):
        self.data_dir = data_dir
        df = pd.read_csv(self.data_dir, header=None).dropna()
        df.columns = ["text_1","text_2","correlation"]
        self.texts_1 = df["text_1"].tolist()
        self.texts_2 = df["text_2"].tolist()
        self.labels = df["correlation"].tolist()

    def __call__(
            self, 
            model: SentenceEncodingModel,
    ):
        text_1_embeds = model.encode(self.texts_1, show_progress_bar=True)
        text_2_embeds = model.encode(self.texts_2, show_progress_bar=True)
        cosine_scores = 1 - (paired_cosine_distances(text_1_embeds, text_2_embeds))
        eval_spearman_cosine, _ = spearmanr(self.labels, cosine_scores)
        return {
            "spearman_cosine": round(eval_spearman_cosine * 100, 2),
        }