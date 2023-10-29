import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pyodbc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

server = 'your server name' 
database = 'your database' 
username = 'your username' 
password = 'your password'  
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

query = "your query" 
data = pd.read_sql(query, cnxn)

class ReadQuery(BaseEstimator, TransformerMixin):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        server = 'your server name' 
        database = 'your database' 
        username = 'your username' 
        password = 'your password'  
        cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
        cursor = cnxn.cursor()

        file_name = "your query"
        df = pd.read_sql(self.file_name, cnxn)
        
        return df

class DataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, df, y=None):
        return self
        
    def transform(self, df, y=None):
        data_train_test = data[(data["Title"] == "Champions") | (data["Title"] == "Loyals")].drop(["UserID","OrderCount","Recency", "Frequency", "Monetary"], axis = 1)
        data_validation = data[data["Title"] == "Can to be Loyals"]


        data_train_test1 = data_train_test.replace({"title" : "Champions"}, 0).replace({"title" : "Loyals"}, 1)


        x = data_train_test1.iloc[:,:-1]
        mm = MinMaxScaler()
        X_mm = pd.DataFrame(mm.fit_transform(x), columns = x.columns)
        data_train_test2 = pd.concat([X_mm, data_train_test1.iloc[:,-1]], axis = 1)


        X = data_train_test2.iloc[:,:-1]
        Y = data_train_test2.iloc[:,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
        
        #smote = SMOTE()
        #x = df_train.iloc[:,:-1]
        #y = df_train.iloc[:,-1]
        #x_smote, y_smote = smote.fit_resample(x,y)
        
        return X_train, X_test, Y_train, Y_test



class Model(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, rfmTable1, y=None):
        return self
        
    def transform(self, rfmTable1, y=None):
        DT_model = DecisionTreeClassifier(criterion = "gini", max_depth = 12)
        DT_model.fit(X_train, Y_train)
        y_pred_dt = DT_model.predict(X_test)


        RF_model = RandomForestClassifier(n_estimators = ,criterion = , max_depth = )
        RF_model.fit(X_train, Y_train)
        y_pred_rf = RF_model.predict(X_test)


        KNN_model = KNeighborsClassifier(n_neighbors = 5, weights = "distance", algorithm = "auto", metric = "minkowski", p = 3)
        KNN_model.fit(X_train, Y_train)
        y_pred_knn = KNN_model.predict(X_test)

        ## Decision Tree
        cm_dt = confusion_matrix(Y_test, y_pred_dt)
        acc_dt = accuracy_score(Y_test, y_pred_dt)
        pr_dt = precision_score(Y_test, y_pred_dt)
        re_dt = recall_score(Y_test, y_pred_dt)
        f1_dt = f1_score(Y_test, y_pred_dt)

        ## Random Forest
        confusion_matrix(Y_test, y_pred_rf)
        accuracy_score(Y_test, y_pred_rf)
        precision_score(Y_test, y_pred_rf)
        recall_score(Y_test, y_pred_rf)
        f1_score(Y_test, y_pred_rf)

        ## Rknn
        cm_knn = confusion_matrix(Y_test, y_pred_knn)
        acc_knn = accuracy_score(Y_test, y_pred_knn)
        pr_knn = precision_score(Y_test, y_pred_knn)
        re_knn = recall_score(Y_test, y_pred_knn)
        f1_knn = f1_score(Y_test, y_pred_knn)


        DT_eval = {"Accuracy": acc_dt, 
              "Precision": pr_dt,
              "Recall": re_dt,
              "F1": f1_dt}
        RF_eval = {"Accuracy": acc_rf, 
              "Precision": pr_rf,
              "Recall": re_rf,
              "F1": f1_rf}
        KNN_eval = {"Accuracy": acc_knn, 
              "Precision": pr_knn,
              "Recall": re_knn,
              "F1": f1_knn}

        DT_evals = pd.DataFrame(DT_eval, index = ["DT"])
        RF_evals = pd.DataFrame(RF_eval, index = ["RF"])
        KNN_evals = pd.DataFrame(KNN_eval, index = ["KNN"])

        eval_final = pd.concat([DT_evals,RF_evals,KNN_evals], axis = 0)
        
        data_validation1 = data_validation.iloc[:,:-1]
        y_pred_dt_valid = DT_model.predict(data_validation1) 
        y_pred_rf_valid = RF_model.predict(data_validation1) 
        y_pred_knn_valid = KNN_model.predict(data_validation1) 

        return y_pred_dt_valid, y_pred_rf_valid, y_pred_knn_valid
    
    
    
    data_pipeline = Pipeline([
    ('ReadQuery', ReadQuery("SELECT userid, questionpaymentdate FROM CON.FactQuestion WHERE questionpayment = 1 and questionpaymentdate >= '2023-03-01';")),
    ('DataPreparation', DataPreparation()),
    ('Model', Model()))
])
        
data_pipeline.fit_transform(None)