import os
import datasets
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer
from collections import defaultdict

import cohere
cohere_model = "embed-multilingual-v3.0"
co = cohere.Client('Z0AuLPY1Q2B2n0o3zyntszwvWmBB5MCqnnnuRyNc')
# response = co.models.list()
# print(response)


def eval_xquad(model):
    dataset = datasets.load_dataset("xquad","xquad.th")

    all_doc = set(dataset['validation']['context'])
    all_doc = {c:i for i,c in enumerate(all_doc)}

    question_contextid_context = []
    for item in dataset['validation']:
        question = item['question']
        doc = item['context']
        question_contextid_context.append([all_doc[doc],question])
        
    df_question = pd.DataFrame(question_contextid_context, columns =['doc_id','question'])
    df_document = pd.DataFrame(zip(list(all_doc.values()),list(all_doc.keys())), columns =['doc_id','document'])

    context_id = df_document['doc_id'].to_list()    
    context_all = co.embed(
      texts=df_document['document'].to_list(),
      model=cohere_model,
      input_type="search_document",
    ).embeddings

    question_id = df_question['doc_id'].to_list()
    question_all = co.embed(
      texts=df_question['question'].to_list(),
      model=cohere_model,
      input_type="search_query",
    ).embeddings

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all,context_all)
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        idx_search = list(index_edit).index(question_id[idx])
        if idx_search == 0:
            hit_1+=1
            hit_5+=1
            hit_10+=1
        elif idx_search < 5:
            hit_5+=1
            hit_10+=1
        elif idx_search < 10:
            hit_10+=1  
        if idx_search < 10:
            mrr_score += (1/(idx_search+1))
    hit_1/=len(question_all)
    hit_5/=len(question_all)
    hit_10/=len(question_all)
    mrr_score/=len(question_all)
    return {
        "Hit@1": hit_1,
        "Hit@5": hit_5,
        "Hit@10": hit_10,
        "MRR@10": mrr_score,
    }


def eval_miracl(model):
    dataset = datasets.load_dataset("miracl/miracl", "th")

    queries = []
    answers = []
    docs = []
    for data in dataset['dev']: 
        query = data['query']
        positive_passages = data['positive_passages']
        negative_passages = data['negative_passages']
        
        queries.append(query)
        answers.append([x['text'] for x in positive_passages])
    
        docs += [x['text'] for x in positive_passages]
        docs += [x['text'] for x in negative_passages]
    docs = list(set(docs))  

    doc_embeddings = co.embed(
      texts=docs,
      model=cohere_model,
      input_type="search_document",
    ).embeddings
    question_embeddings = co.embed(
      texts=queries,
      model=cohere_model,
      input_type="search_query",
    ).embeddings

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    mrr_score = 0
    sim_score = np.inner(question_embeddings, doc_embeddings)
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        doc_sorted = [docs[i] for i in index]
        answer_idx = [doc_sorted.index(a) for a in answers[idx]] # cal index for each answer
        final_idx_search = min(answer_idx) # since we have multiple answers, we find the min index! 
        if final_idx_search == 0:
            hit_1+=1
            hit_5+=1
            hit_10+=1
        elif final_idx_search < 5:
            hit_5+=1
            hit_10+=1
        elif final_idx_search < 10:
            hit_10+=1  
        if final_idx_search < 10:
            mrr_score += (1/(final_idx_search+1))
    hit_1/=len(question_embeddings)
    hit_5/=len(question_embeddings)
    hit_10/=len(question_embeddings)
    mrr_score/=len(question_embeddings)
    return {
        "Hit@1": hit_1,
        "Hit@5": hit_5,
        "Hit@10": hit_10,
        "MRR@10": mrr_score,
    }


def eval_tydiqa(model):
    dataset = datasets.load_dataset("chompk/tydiqa-goldp-th", trust_remote_code=True)

    all_doc = set(dataset['validation']['context'])
    all_doc = {c:i for i,c in enumerate(all_doc)}

    question_contextid_context = []
    for item in dataset['validation']:
        question = item['question']
        doc = item['context']
        question_contextid_context.append([all_doc[doc],question])
        
    df_question = pd.DataFrame(question_contextid_context, columns =['doc_id','question'])
    df_document = pd.DataFrame(zip(list(all_doc.values()),list(all_doc.keys())), columns =['doc_id','document'])

    context_id = df_document['doc_id'].to_list()    
    context_all = co.embed(
      texts=df_document['document'].to_list(),
      model=cohere_model,
      input_type="search_document",
    ).embeddings

    question_id = df_question['doc_id'].to_list()
    question_all = co.embed(
      texts=df_question['question'].to_list(),
      model=cohere_model,
      input_type="search_query",
    ).embeddings

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    mrr_score = 0
    context_id = np.array(context_id)
    sim_score = np.inner(question_all,context_all)
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        idx_search = list(index_edit).index(question_id[idx])
        if idx_search == 0:
            hit_1+=1
            hit_5+=1
            hit_10+=1
        elif idx_search < 5:
            hit_5+=1
            hit_10+=1
        elif idx_search < 10:
            hit_10+=1  
        if idx_search < 10:
            mrr_score += (1/(idx_search+1))
    hit_1/=len(question_all)
    hit_5/=len(question_all)
    hit_10/=len(question_all)
    mrr_score/=len(question_all)
    return {
        "Hit@1": hit_1,
        "Hit@5": hit_5,
        "Hit@10": hit_10,
        "MRR@10": mrr_score,
    }


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

    results = defaultdict(lambda: defaultdict(float))
    print(f"Evaluating...")
    # Evaluate model
    xquad_results = eval_xquad(co)
    miracl_results = eval_miracl(co)
    tydiqa_results = eval_tydiqa(co)
    results["Cohere"].update(
        {
            "XQuAD_R@1": round(xquad_results["Hit@1"] * 100, 2),
            "XQuAD_MRR@10": round(xquad_results["MRR@10"] * 100, 2),
            "MIRACL_R@1": round(miracl_results["Hit@1"] * 100, 2),
            "MIRACL_MRR@10": round(miracl_results["MRR@10"] * 100, 2),
            "TyDiQA_R@1": round(tydiqa_results["Hit@1"] * 100, 2),
            "TyDiQA_MRR@10": round(tydiqa_results["MRR@10"] * 100, 2),
        }
    )

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/cohere_v3_retrieval.csv")
    print(results_df)