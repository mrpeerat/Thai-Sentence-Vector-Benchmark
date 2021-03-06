from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

class Transfer:
    def __init__(self, model, tasks, train_score=False, val_score=False, test_score=True):
        self.model = model
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

    def run(self):
        datasets = self.load_data()
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

            X_train_encode = self.model.encode(X_train)
            X_val_encode = self.model.encode(X_val)
            X_test_encode = self.model.encode(X_test)
            
            text_clf = LinearSVC(class_weight='balanced')
            text_clf.fit(X_train_encode, y_train)
            
            if self.train_score:
                print(f"Dataset: {key} Set: Train")
                train_predicted = text_clf.predict(X_train_encode)
                print(classification_report(y_train, train_predicted, digits=4))
            if self.val_score:
                print(f"Dataset: {key} Set: Validate")
                val_predicted = text_clf.predict(X_val_encode)
                print(classification_report(y_val, val_predicted, digits=4))
            if self.test_score:
                print(f"Dataset: {key} Set: Test")
                test_predicted = text_clf.predict(X_test_encode)
                print(classification_report(y_test, test_predicted, digits=4))
            
            print('*'*50)