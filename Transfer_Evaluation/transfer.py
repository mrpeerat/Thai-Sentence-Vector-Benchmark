from tqdm import trange
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np


class TransferEvaluator:
    def __init__(self, tasks, train_score=False, val_score=False, test_score=True):
        self.tasks = tasks
        self.train_score = train_score
        self.val_score = val_score
        self.test_score = test_score

    def wisesight_preprocess(self,data):
        X_train = data['train']['texts']
        y_train = data['train']['category']
        
        X_val = data['validation']['texts']
        y_val = data['validation']['category']
        
        X_test = data['test']['texts']
        y_test = data['test']['category']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def wongnai_preprocess(self,data):
        X_train = data['train']['review_body']
        y_train = data['train']['star_rating']
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
        
        X_test = data['test']['review_body']
        y_test = data['test']['star_rating']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def generated_preprocess(self,data):
        X_train = [text['th'] for text in data['train']['translation']]
        y_train = data['train']['review_star']
        
        X_val = [text['th'] for text in data['validation']['translation']]
        y_val = data['validation']['review_star']
        
        X_test = [text['th'] for text in data['test']['translation']]
        y_test = data['test']['review_star']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def load_data(self):
        datasets={}
        for name in self.tasks:
            datasets.update({name: load_dataset(name)})
        return datasets

    def __call__(self, model, cohere=False, bge=False):
        datasets = self.load_data()

        results = {}
        for key in self.tasks:
            data = datasets[key]
            if key == 'wisesight_sentiment':
                X_train, y_train, X_val, y_val, X_test, y_test =  self.wisesight_preprocess(data)
            elif key == 'wongnai_reviews':
                X_train, y_train, X_val, y_val, X_test, y_test =  self.wongnai_preprocess(data)
            elif key == 'generated_reviews_enth':
                X_train, y_train, X_val, y_val, X_test, y_test =  self.generated_preprocess(data)
            else:
                raise Exception(f"Key Error:{key}")
                
            if cohere:
                bs = 96
                X_train_encode = []
                X_val_encode = []
                X_test_encode = []
                for i in trange(len(X_train)//bs+1):
                    texts = X_train[(i*bs):((i+1)*bs)]
                    if len(texts) == 0: continue
                    X_train_encode.append(model.embed(
                      texts=texts,
                      model='embed-multilingual-v2.0',
                    ).embeddings)
                for i in trange(len(X_val)//bs+1):
                    texts=X_val[(i*bs):((i+1)*bs)]
                    if len(texts) == 0: continue
                    X_val_encode.append(model.embed(
                      texts=texts,
                      model='embed-multilingual-v2.0',
                    ).embeddings)
                for i in trange(len(X_test)//bs+1):
                    texts=X_test[(i*bs):((i+1)*bs)]
                    if len(texts) == 0: continue
                    X_test_encode.append(model.embed(
                      texts=texts,
                      model='embed-multilingual-v2.0',
                    ).embeddings)
                X_test_encode = np.concatenate(X_test_encode,0)
                X_train_encode = np.concatenate(X_train_encode,0)
                X_val_encode = np.concatenate(X_val_encode,0)
            elif bge:
                X_train_encode = model.encode(X_train,batch_size=12,return_dense=True, return_sparse=False, return_colbert_vecs=False)
                X_val_encode = model.encode(X_val,batch_size=12,return_dense=True, return_sparse=False, return_colbert_vecs=False)
                X_test_encode = model.encode(X_test,batch_size=12,return_dense=True, return_sparse=False, return_colbert_vecs=False)

                X_train_encode = X_train_encode['dense_vecs']
                X_val_encode = X_val_encode['dense_vecs']
                X_test_encode = X_test_encode['dense_vecs']
            else:
                X_train_encode = model.encode(X_train, batch_size=12, show_progress_bar=True)
                # X_val_encode = model.encode(X_val, batch_size=12, show_progress_bar=True)
                X_test_encode = model.encode(X_test, batch_size=12, show_progress_bar=True)

            print("Training classification head...") 
            text_clf = LinearSVC(class_weight='balanced', verbose=0)
            text_clf.fit(X_train_encode, y_train)
            
            results[key] = {}
            if self.train_score:
                train_predicted = text_clf.predict(X_train_encode)
                results[key]["train"] = classification_report(y_train, train_predicted, digits=4, output_dict=True)
            # if self.val_score:
            #     val_predicted = text_clf.predict(X_val_encode)
            #     results[key]["val"] = classification_report(y_val, val_predicted, digits=4, output_dict=True)
            if self.test_score:
                test_predicted = text_clf.predict(X_test_encode)
                results[key]["test"] = classification_report(y_test, test_predicted, digits=4, output_dict=True)
            
        return results