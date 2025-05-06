import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utilitiesFun as ut

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

class ChurnModelling:
    def __init__(self, set_seed=123):
        self.seed = set_seed

    def extract_features(self, data_train, data_test):
        x_train = pd.DataFrame(data_train.drop(columns=['Churn']))
        x_test = pd.DataFrame(data_test.drop(columns=['Churn']))

        return x_train, x_test

    def extract_label(self, data_train, data_test):
        y_train = data_train['Churn']
        y_test = data_test['Churn']
        
        return y_train, y_test

    def standScaling_features(self, x_train, x_test):
        # x_train:
        scaler=StandardScaler()
        x_train=x_train.drop(columns=['last_PurchaseDate', 'first_PurchaseDate'])
        x_train=pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
        # x_test:
        x_test=x_test.drop(columns=['last_PurchaseDate', 'first_PurchaseDate'])
        x_test=pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

        return(x_train, x_test)
    
    def baseline_random(self, y_test):
        prob = round(y_test.sum()/y_test.shape[0], 4)
        length = y_test.shape[0]
        # random prediction:
        baseline_random = np.random.choice([0, 1], size=length, p=[1-prob, prob])

        return baseline_random
    
    def baseline_all_zeros(self, y_test):
        # all 0 prediction:
        baseline_zero = [0 for i in y_test]

        return baseline_zero
    
    def baseline_all_ones(self, y_test):
        # all 0 prediction:
        baseline_zero = [1 for i in y_test]

        return baseline_zero

    def logit_modelling(self, x_train, y_train, x_test):
        # training:
        logit_model = LogisticRegression(random_state=123)
        logit_model.fit(x_train, y_train)
        # predicting:
        logit_pred = logit_model.predict(x_test)

        return(logit_pred)

    def lasso_modelling(self, x_train, y_train, x_test, cv_lasso):
        # training:
        lasso_cv = LassoCV(random_state=123, cv=cv_lasso)         # parameter
        lasso_cv.fit(x_train, y_train)
        # predicting:
        lasso_pred = lasso_cv.predict(x_test)

        return(lasso_pred)
        
    def bayes_modelling(self, x_train, y_train, x_test):
        # training:
        bayes_model = GaussianNB()
        bayes_model.fit(x_train, y_train)
        # prediction:
        bayes_pred = bayes_model.predict(x_test)

        return(bayes_pred)
    
    def knn_modelling(self, x_train, y_train, x_test, range_knn, cv_knn):
        # choosing best k_neighbors:
        precision_scorer = make_scorer(f1_score, zero_division=0)
        scores = []

        for k in range (1, range_knn):
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, x_train, y_train, cv=cv_knn, scoring=precision_scorer)
            scores.append(np.mean(score))
        best_k = np.argmax(scores)+1

        # training:
        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        knn_model.fit(x_train, y_train)
        # prediction:
        knn_pred = knn_model.predict(x_test)

        return(knn_pred)

    def tree_modelling(self, x_train, y_train, x_test, range_tree, cv_tree):
        # choosing best depth:
        precision_scorer = make_scorer(f1_score, zero_division=0)

        scores = []
        for i in range(1, range_tree):
            tree_model = DecisionTreeClassifier(max_depth=i)
            score = cross_val_score(tree_model, x_train, y_train, cv=cv_tree, scoring=precision_scorer)
            scores.append(np.mean(score))
        best_depth=np.argmax(scores)+3

        # training:
        tree_model = DecisionTreeClassifier(max_depth=best_depth)
        tree_model.fit(x_train, y_train)
        # prediction:
        tree_pred = tree_model.predict(x_test)

        return(tree_pred)
    
    def choosing_threshold(self, predictions, y_test):
        # creation of a f1_score matrix for each 0.005 alpha
        results=[]
        for alpha in np.arange(0, 1, 0.005):
            pred_new = [-1]*len(predictions)
            for i in range(0, len(predictions)):
                if predictions[i] > alpha:
                    pred_new[i]=1
                else:
                    pred_new[i]=0
            results.append((f1_score(y_test, pred_new, zero_division=0.0), alpha))
        results=pd.DataFrame(results, columns=['f1', 'alpha'])
        # finding best alpha:
        alpha=results[results['f1']==results['f1'].max()]['alpha'].max()
        # cutting with the chosen alpha:
        for i in range(0, len(predictions)):
                if predictions[i] > alpha:
                    predictions[i]=1
                else:
                    predictions[i]=0

        return predictions
    
    def score_matrix(self, models, y_test):
        # f1_score and precision matrix for each model used
        scoreMatrix_test = []
        for name, predictions in models.items():
            precision = precision_score(y_test, predictions, zero_division=0.0)
            f1 = f1_score(y_test, predictions, zero_division=0.0)
            scoreMatrix_test.append((name, precision, f1))
        scoreMatrix_test=pd.DataFrame(scoreMatrix_test, columns=['model', 'precision', 'f1'])

        return scoreMatrix_test
    

    def modelling(self, data_train, data_test, cv_lasso=3, range_knn=26, cv_knn=3, range_tree=36, cv_tree=3):
        # features and labels:
        x_train, x_test = self.extract_features(data_train, data_test)
        y_train, y_test = self.extract_label(data_train, data_test)

        # scaling features:
        x_train, x_test = self.standScaling_features(x_train, x_test)

        # training and predict:
        models = {
            "Baseline Random": self.baseline_random(y_test),
            "Baseline Zeros": self.baseline_all_zeros(y_test),
            "Baseline Ones": self.baseline_all_ones(y_test),
            "Logistic Regression": self.logit_modelling(x_train, y_train, x_test),
            "Lasso CV": self.lasso_modelling(x_train, y_train, x_test, cv_lasso),
            "Naive Bayes": self.bayes_modelling(x_train, y_train, x_test),
            "K-Nearest Neighbors": self.knn_modelling(x_train, y_train, x_test, range_knn, cv_knn),
            "Decision Tree": self.tree_modelling(x_train, y_train, x_test, range_tree, cv_tree)
        } 

        # checking if predictions need to be cutted with a threshold:
        for name, predictions in models.items():
            if not(all(x in [0, 1] for x in predictions)):
                predictions = self.choosing_threshold(predictions, y_test)
        
        # creating score matrix:
        score_matrix_test = self.score_matrix(models, y_test)
        

        return models, y_test, score_matrix_test


    def confusion_matrix(self, models, y_test):
        dict_confusion_matrix = {}
        for name, predictions in models.items():
            dict_confusion_matrix[name] = pd.DataFrame(confusion_matrix(y_test, predictions))

        return dict_confusion_matrix

    def roc_curve_plot(self, models, y_test):
        dict_roc_auc = {}
        # calculating roc/auc for each model
        for name, predictions in models.items():
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            auc_score = auc(fpr, tpr)
            dict_roc_auc[name] = {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score}
        
        return dict_roc_auc
        
    