import datasets
import numpy as np
import pandas as pd
from typing import List
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel



class RetrievalBenchmark:
    def __init__(self, dataset_names: List[str] = ("xquad", "miracl", "tydiqa")):
        self.dataset_names = dataset_names
        self.datasets = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "xquad":
                self.datasets[dataset_name] = datasets.load_dataset("xquad","xquad.th")
            elif dataset_name == "miracl":
                self.datasets[dataset_name] = datasets.load_dataset("miracl/miracl", "th")
            elif dataset_name == "tydiqa":
                self.datasets[dataset_name] = datasets.load_dataset("chompk/tydiqa-goldp-th", trust_remote_code=True)

    def eval_xquad(self, model: SentenceEncodingModel, batch_size: int = 1024):
        dataset = self.datasets["xquad"]

        all_doc = set(dataset['validation']['context'])
        all_doc = {c: i for i, c in enumerate(all_doc)}

        question_contextid_context = []
        for item in dataset['validation']:
            question = item['question']
            doc = item['context']
            question_contextid_context.append([all_doc[doc], question])
            
        df_question = pd.DataFrame(question_contextid_context, columns=['doc_id', 'question'])
        df_document = pd.DataFrame(zip(list(all_doc.values()), list(all_doc.keys())), columns=['doc_id', 'document'])

        context_id = df_document['doc_id'].to_list()    
        context_all = model.encode(df_document['document'].to_list(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_document")

        question_id = df_question['doc_id'].to_list()
        question_all = model.encode(df_question['question'].to_list(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_query")

        sim_score = np.inner(question_all, context_all)

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
                top_1 += 1
                top_5 += 1
                top_10 += 1
            elif idx_search < 5:
                top_5 += 1
                top_10 += 1
            elif idx_search < 10:
                top_10 += 1  
            if idx_search < 10:
                mrr_score += (1 / (idx_search + 1))
        top_1 /= len(question_all)
        top_5 /= len(question_all)
        top_10 /= len(question_all)
        mrr_score /= len(question_all)
        return {
            "R@1": top_1,
            "R@5": top_5,
            "R@10": top_10,
            "MRR@10": mrr_score,
        }

    def eval_miracl(self, model: SentenceEncodingModel, batch_size: int = 1024):
        dataset = self.datasets["miracl"]

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

        doc_embeddings = model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_document")

        question_embeddings = model.encode(queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_query")

        sim_score = np.inner(question_embeddings, doc_embeddings)

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
                top_1 += 1
                top_5 += 1
                top_10 += 1
            elif final_idx_search < 5:
                top_5 += 1
                top_10 += 1
            elif final_idx_search < 10:
                top_10 += 1  
            if final_idx_search < 10:
                mrr_score += (1 / (final_idx_search + 1))
        top_1 /= len(question_embeddings)
        top_5 /= len(question_embeddings)
        top_10 /= len(question_embeddings)
        mrr_score /= len(question_embeddings)
        return {
            "R@1": top_1,
            "R@5": top_5,
            "R@10": top_10,
            "MRR@10": mrr_score,
        }

    def eval_tydiqa(self, model: SentenceEncodingModel, batch_size: int = 1024):
        dataset = self.datasets["tydiqa"]

        all_doc = set(dataset['validation']['context'])
        all_doc = {c: i for i, c in enumerate(all_doc)}

        question_contextid_context = []
        for item in dataset['validation']:
            question = item['question']
            doc = item['context']
            question_contextid_context.append([all_doc[doc], question])
            
        df_question = pd.DataFrame(question_contextid_context, columns=['doc_id', 'question'])
        df_document = pd.DataFrame(zip(list(all_doc.values()), list(all_doc.keys())), columns=['doc_id', 'document'])

        context_id = df_document['doc_id'].to_list()    
        context_all = model.encode(df_document['document'].to_list(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_document")

        question_id = df_question['doc_id'].to_list()
        question_all = model.encode(df_question['question'].to_list(), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, input_type="search_query")

        sim_score = np.inner(question_all,context_all)

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
                top_1 += 1
                top_5 += 1
                top_10 += 1
            elif idx_search < 5:
                top_5 += 1
                top_10 += 1
            elif idx_search < 10:
                top_10 += 1  
            if idx_search < 10:
                mrr_score += (1 / (idx_search + 1))
        top_1 /= len(question_all)
        top_5 /= len(question_all)
        top_10 /= len(question_all)
        mrr_score /= len(question_all)
        return {
            "R@1": top_1,
            "R@5": top_5,
            "R@10": top_10,
            "MRR@10": mrr_score,
        }

    def __call__(
            self, 
            model: SentenceEncodingModel,
            batch_size: int = 1024,
    ):
        results = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "xquad":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_xquad(model, batch_size=batch_size).items()}
            elif dataset_name == "miracl":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_miracl(model, batch_size=batch_size).items()}
            elif dataset_name == "tydiqa":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_tydiqa(model, batch_size=batch_size).items()}
        return results