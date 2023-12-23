import os
from transfer import Transfer
from sentence_transformers import SentenceTransformer
import cohere
co = cohere.Client('YOUR API KEY!')

task_names = ['wisesight_sentiment','wongnai_reviews','generated_reviews_enth']
transfer_run = Transfer(co,task_names,cohere=True)

print(f"Model:Cohere")
transfer_run.run()

