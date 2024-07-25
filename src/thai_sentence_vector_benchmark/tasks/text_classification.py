import datasets
from time import time
from sklearn.svm import LinearSVC
from typing import List, Optional
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from thai_sentence_vector_benchmark.models.baseclass import SentenceEncodingModel


class TextClassificationBenchmark:
    def __init__(self, dataset_names: List[str] = ("wisesight", "wongnai", "generated_reviews")):
        self.dataset_names = dataset_names
        self.datasets = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "wisesight":
                self.datasets[dataset_name] = datasets.load_dataset("wisesight_sentiment")
            elif dataset_name == "wongnai":
                self.datasets[dataset_name] = datasets.load_dataset("wongnai_reviews")
            elif dataset_name == "generated_reviews":
                self.datasets[dataset_name] = datasets.load_dataset("generated_reviews_enth")

    def get_wisesight_dataset(self):
        dataset = self.datasets["wisesight"]

        X_train = dataset['train']['texts']
        y_train = dataset['train']['category']

        X_val = dataset['validation']['texts']
        y_val = dataset['validation']['category']
        
        X_test = dataset['test']['texts']
        y_test = dataset['test']['category']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_wongnai_dataset(self):
        dataset = self.datasets["wongnai"]

        X_train = dataset['train']['review_body']
        y_train = dataset['train']['star_rating']
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
        
        X_test = dataset['test']['review_body']
        y_test = dataset['test']['star_rating']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_generated_reviews_dataset(self):
        dataset = self.datasets["generated_reviews"]

        X_train = [text['th'] for text in dataset['train']['translation']]
        y_train = dataset['train']['review_star']
        
        X_val = [text['th'] for text in dataset['validation']['translation']]
        y_val = dataset['validation']['review_star']
        
        X_test = [text['th'] for text in dataset['test']['translation']]
        y_test = dataset['test']['review_star']
        return X_train, y_train, X_val, y_val, X_test, y_test

    def __call__(
            self, 
            model: SentenceEncodingModel,
            prompt: Optional[str] = None,
            batch_size: int = 1024,
    ):
        results = {}
        for dataset_name in self.dataset_names:
            if dataset_name == "wisesight":
                X_train, y_train, _, _, X_test, y_test = self.get_wisesight_dataset()
            elif dataset_name == "wongnai":
                X_train, y_train, _, _, X_test, y_test = self.get_wongnai_dataset()
            elif dataset_name == "generated_reviews":
                X_train, y_train, _, _, X_test, y_test = self.get_generated_reviews_dataset()

            # Train classification head
            train_embeds = model.encode(X_train, prompt=prompt, batch_size=batch_size, show_progress_bar=True)
            text_clf = LinearSVC(class_weight='balanced', verbose=0)
            print("Training classification head...") 
            init_time = time()
            text_clf.fit(train_embeds, y_train)
            print(f"Training time: {time() - init_time:.2f}s")

            # Evaluate
            test_embeds = model.encode(X_test, prompt=prompt, batch_size=batch_size, show_progress_bar=True)
            test_predicted = text_clf.predict(test_embeds)
            result = classification_report(y_test, test_predicted, output_dict=True)
            results[dataset_name] = {
                "Accuracy": round(result["accuracy"] * 100 , 2),
                "F1": round(result["weighted avg"]["f1-score"] * 100 , 2),
            }
        return results