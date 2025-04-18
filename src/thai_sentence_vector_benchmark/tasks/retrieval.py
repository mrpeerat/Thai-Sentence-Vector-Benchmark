import datasets
import numpy as np
import pandas as pd
from typing import List, Optional
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class RetrievalBenchmark:
    def __init__(self, dataset_names: List[str] = (
                                                  "xquad", 
                                                  "miracl", 
                                                  "tydiqa",
                                                  "iapp-wikiqa",
                                                  "mldr",
                                                  "thai-wikiqa",
                                                  "wangchanx-legalrag")):
        self.dataset_names = dataset_names
        self.datasets = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "xquad":
                self.datasets[dataset_name] = datasets.load_dataset("xquad","xquad.th")
            elif dataset_name == "miracl":
                self.datasets[dataset_name] = datasets.load_dataset("miracl/miracl", "th")
            elif dataset_name == "tydiqa":
                self.datasets[dataset_name] = datasets.load_dataset("chompk/tydiqa-goldp-th", trust_remote_code=True)
            else:
                try:
                    huggingface_name = f"kornwtp/{dataset_name}-tha-qaretrieval"
                    self.datasets[dataset_name] = datasets.load_dataset(huggingface_name)
                except:
                    raise NotImplementedError

    def compute_metric(self, question_all, question_id, context_id, sim_score):
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

    def eval_xquad(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
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
        context_all = model.encode(df_document['document'].to_list(), prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        question_id = df_question['doc_id'].to_list()
        question_all = model.encode(df_question['question'].to_list(), prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        sim_score = np.inner(question_all, context_all)

        return self.compute_metric(question_all, question_id, context_id, sim_score)

    def eval_miracl(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
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

        doc_embeddings = model.encode(docs, prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        question_embeddings = model.encode(queries, prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

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

    def eval_mldr(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
        dataset = self.datasets["mldr"]

        queries = []
        answers = []
        docs = []
        for data in dataset['test']: 
            query = data['query']
            positive_passages = data['positive_passages']
            negative_passages = data['negative_passages']
            
            queries.append(query)
            answers.append([x['text'] for x in positive_passages])
        
            docs += [x['text'] for x in positive_passages]
            docs += [x['text'] for x in negative_passages]
        docs = list(set(docs))  

        doc_embeddings = model.encode(docs, prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        question_embeddings = model.encode(queries, prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

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

    def eval_tydiqa(self, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
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
        context_all = model.encode(df_document['document'].to_list(), prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        question_id = df_question['doc_id'].to_list()
        question_all = model.encode(df_question['question'].to_list(), prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

        sim_score = np.inner(question_all,context_all)

        return self.compute_metric(question_all, question_id, context_id, sim_score)

    def eval_retrieval(self, dataset_name, model: SentenceEncodingModel, prompt: Optional[str] = None, batch_size: int = 1024):
        dataset = self.datasets[dataset_name]
        if dataset_name == "wangchanx-legalrag":
            queries, answers, docs = [], [], []
            for data in dataset['test']:
                query = data["question"]
                positive_contexts = [d["text"] for d in data["positive_contexts"]]
                if len(data["hard_negative_contexts"]) > 0:
                    hard_negative_contexts = [d["text"] for d in data["hard_negative_contexts"]]

                queries.append(query)
                answers.append(positive_contexts)

                docs += positive_contexts
                if len(data["hard_negative_contexts"]) > 0:
                    docs += hard_negative_contexts

            docs = list(set(docs))

            doc_embeddings = model.encode(docs, prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

            question_embeddings = model.encode(queries, prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

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
        else:
            if "test" in dataset.keys():
                data_split = "test"
            elif "validation" in dataset.keys():
                data_split = "validation"
            elif "train" in dataset.keys():
                data_split = "train"

            all_doc = set(dataset[data_split]['context'])
            all_doc = {c: i for i, c in enumerate(all_doc)}

            question_contextid_context = []
            for item in dataset[data_split]:
                question = item['question']
                doc = item['context']
                question_contextid_context.append([all_doc[doc], question])
                
            df_question = pd.DataFrame(question_contextid_context, columns=['doc_id', 'question'])
            df_document = pd.DataFrame(zip(list(all_doc.values()), list(all_doc.keys())), columns=['doc_id', 'document'])

            context_id = df_document['doc_id'].to_list()    
            context_all = model.encode(df_document['document'].to_list(), prompt=None, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

            question_id = df_question['doc_id'].to_list()
            question_all = model.encode(df_question['question'].to_list(), prompt=prompt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

            sim_score = np.inner(question_all, context_all)

            return self.compute_metric(question_all, question_id, context_id, sim_score)


    def __call__(
            self, 
            model: SentenceEncodingModel,
            prompt: Optional[str] = None,
            batch_size: int = 1024,
    ):
        results = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "xquad":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_xquad(model, prompt=prompt, batch_size=batch_size).items()}
            elif dataset_name == "miracl":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_miracl(model, prompt=prompt, batch_size=batch_size).items()}
            elif dataset_name == "tydiqa":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_tydiqa(model, prompt=prompt, batch_size=batch_size).items()}
            elif dataset_name == "mldr":
                results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_mldr(model, prompt=prompt, batch_size=batch_size).items()}
            else:
                try:
                    results[dataset_name] = {k: round(v * 100, 2) for k, v in self.eval_retrieval(dataset_name, model, prompt=prompt, batch_size=batch_size).items()}
                except:
                    raise NotImplementedError
            
        return results