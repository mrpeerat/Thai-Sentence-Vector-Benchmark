import os
from transfer import Transfer

from sentence_transformers import SentenceTransformer
model_name = 'mrp/simcse-model-wangchanberta'
# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)

task_names = ['wisesight_sentiment','wongnai_reviews','generated_reviews_enth']
transfer_run = Transfer(model,task_names)

print(f"Model:{model_name}")
transfer_run.run()

