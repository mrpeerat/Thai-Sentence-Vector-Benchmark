from transfer import Transfer

from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 

task_names = ['wisesight_sentiment','wongnai_reviews','generated_reviews_enth']
transfer_run = Transfer(model,task_names)

transfer_run.run()

