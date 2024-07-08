import os
import datasets
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel


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

    init_time = time()
    context_id = df_document['doc_id'].to_list()    
    context_all = model.encode(df_document['document'].to_list())['dense_vecs']
    indexing_time = time() - init_time

    init_time = time()
    question_id = df_question['doc_id'].to_list()
    question_all = model.encode(df_question['question'].to_list())['dense_vecs']

    sim_score = np.inner(question_all,context_all)
    inference_time = time() - init_time

    top_1 = 0
    top_5 = 0
    top_10 = 0
    mrr_score = 0
    context_id = np.array(context_id)
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        idx_search = list(index_edit).index(question_id[idx])
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < 10:
            mrr_score += (1/(idx_search+1))
    top_1/=len(question_all)
    top_5/=len(question_all)
    top_10/=len(question_all)
    mrr_score/=len(question_all)
    return {
        "Hit@1": top_1,
        "Hit@5": top_5,
        "Hit@10": top_10,
        "MRR@10": mrr_score,
        "Indexing_Time": indexing_time,
        "Inference_Time": inference_time,
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

    init_time = time()
    doc_embeddings = model.encode(docs)['dense_vecs']
    indexing_time = time() - init_time

    init_time = time()
    question_embeddings = model.encode(queries)['dense_vecs']

    sim_score = np.inner(question_embeddings, doc_embeddings)
    inference_time = time() - init_time

    top_1 = 0
    top_5 = 0
    top_10 = 0
    mrr_score = 0
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        doc_sorted = [docs[i] for i in index]
        answer_idx = [doc_sorted.index(a) for a in answers[idx]] # cal index for each answer
        final_idx_search = min(answer_idx) # since we have multiple answers, we find the min index! 
        if final_idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif final_idx_search < 5:
            top_5+=1
            top_10+=1
        elif final_idx_search < 10:
            top_10+=1  
        if final_idx_search < 10:
            mrr_score += (1/(final_idx_search+1))
    top_1/=len(question_embeddings)
    top_5/=len(question_embeddings)
    top_10/=len(question_embeddings)
    mrr_score/=len(question_embeddings)
    return {
        "Hit@1": top_1,
        "Hit@5": top_5,
        "Hit@10": top_10,
        "MRR@10": mrr_score,
        "Indexing_Time": indexing_time,
        "Inference_Time": inference_time,
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

    init_time = time()
    context_id = df_document['doc_id'].to_list()    
    context_all = model.encode(df_document['document'].to_list())['dense_vecs']
    indexing_time = time() - init_time

    init_time = time()
    question_id = df_question['doc_id'].to_list()
    question_all = model.encode(df_question['question'].to_list())['dense_vecs']

    sim_score = np.inner(question_all,context_all)
    inference_time = time() - init_time

    top_1 = 0
    top_5 = 0
    top_10 = 0
    mrr_score = 0
    context_id = np.array(context_id)
    status_bar = enumerate(sim_score)
    for idx,sim in status_bar:
        index = np.argsort(sim)[::-1]
        index_edit = [context_id[x] for x in index]
        idx_search = list(index_edit).index(question_id[idx])
        if idx_search == 0:
            top_1+=1
            top_5+=1
            top_10+=1
        elif idx_search < 5:
            top_5+=1
            top_10+=1
        elif idx_search < 10:
            top_10+=1  
        if idx_search < 10:
            mrr_score += (1/(idx_search+1))
    top_1/=len(question_all)
    top_5/=len(question_all)
    top_10/=len(question_all)
    mrr_score/=len(question_all)
    return {
        "Hit@1": top_1,
        "Hit@5": top_5,
        "Hit@10": top_10,
        "MRR@10": mrr_score,
        "Indexing_Time": indexing_time,
        "Inference_Time": inference_time,
    }


if __name__ == "__main__":
    os.makedirs("./test_results", exist_ok=True)

    results = defaultdict(lambda: defaultdict(float))
    print(f"Evaluating...")
    # Load model
    model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True) 
    # Evaluate model
    xquad_results = eval_xquad(model)
    miracl_results = eval_miracl(model)
    tydiqa_results = eval_tydiqa(model)
    results["bge"].update(
        {
            "XQuAD_R@1": round(xquad_results["Hit@1"] * 100, 2),
            "XQuAD_MRR@10": round(xquad_results["MRR@10"] * 100, 2),
            "XQuAD_Indexing_Time": round(xquad_results["Indexing_Time"], 2),
            "XQuAD_Inference_Time": round(xquad_results["Inference_Time"], 2),
            "MIRACL_R@1": round(miracl_results["Hit@1"] * 100, 2),
            "MIRACL_MRR@10": round(miracl_results["MRR@10"] * 100, 2),
            "MIRACL_Indexing_Time": round(miracl_results["Indexing_Time"], 2),
            "MIRACL_Inference_Time": round(miracl_results["Inference_Time"], 2),
            "TyDiQA_R@1": round(tydiqa_results["Hit@1"] * 100, 2),
            "TyDiQA_MRR@10": round(tydiqa_results["MRR@10"] * 100, 2),
            "TyDiQA_Indexing_Time": round(tydiqa_results["Indexing_Time"], 2),
            "TyDiQA_Inference_Time": round(tydiqa_results["Inference_Time"], 2),
        }
    )

    # Save results to csv
    results_df = pd.DataFrame(results)
    # Transpose the dataframe
    results_df = results_df.T
    results_df.to_csv("./test_results/bge_retrieval.csv")
    print(results_df)