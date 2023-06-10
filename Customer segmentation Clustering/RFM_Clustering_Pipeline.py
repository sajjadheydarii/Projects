# %% Import the Libraries

import pyodbc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# %% Import the Data

class ReadQuery(BaseEstimator, TransformerMixin):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        server = 'server_address, port_number' 
        database = 'database_name' 
        username = 'user_name' 
        password = 'your_password'  
        cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
        cursor = cnxn.cursor()
        file_name = "SELECT userid, questionpaymentdate FROM database;"
        df = pd.read_sql(self.file_name, cnxn)
        return df
    

# %% Data Preparation

class DataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, df, y=None):
        return self
        
    def transform(self, df, y=None):
        
        df['date'] = pd.DatetimeIndex(df.questionpaymentdate).date
        df = df.drop("questionpaymentdate", axis = 1)
        df_recency = df.groupby(['userid'],as_index=False)['date'].max()
        df_recency.columns = ['userid','Last_Purchase_Date']
        df_recency['Recency'] = df_recency.Last_Purchase_Date.apply(lambda x:(dt.date(2023,3,21) - x).days)
        df_recency.drop(columns=['Last_Purchase_Date'],inplace=True)
        rfmTable = df.groupby('userid').agg({ 'userid': lambda x: len(x)}) # Frequency
        rfmTable.rename(columns={'userid': 'frequency'}, inplace = True)
        RFM_Table = df_recency.merge(rfmTable,left_on='userid',right_on='userid')
        RFM_Table = RFM_Table.set_index("userid")
        sc = StandardScaler()
        RFM_Table_sc = pd.DataFrame(sc.fit_transform(RFM_Table), columns = RFM_Table.columns)
        rfmTable1 = RFM_Table.reset_index().drop(RFM_Table_sc[(RFM_Table_sc["frequency"] < -3) | (RFM_Table_sc["frequency"] > 3)].index, axis = 0).set_index("userid")
        
        return rfmTable1
    
    
# %% Modelling

class Model(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, rfmTable1, y=None):
        return self
        
    def transform(self, rfmTable1, y=None):
        X = rfmTable1
        mm = MinMaxScaler()
        X_mm = pd.DataFrame(mm.fit_transform(X), columns = X.columns)
        kmeans_cluster = KMeans(n_clusters = 5, max_iter = 300)
        y_kmeans = kmeans_cluster.fit_predict(X_mm)  
        y_kmeans_df = pd.DataFrame(y_kmeans, columns = ["cluster"])
        clustered_rfmTable_km = pd.concat([rfmTable1, y_kmeans_df], axis = 1)

        return clustered_rfmTable_km


# %% Save to 

class SaveExcel(BaseEstimator, TransformerMixin):
    def __init__(self, file_name):
        self.file_name = file_name
        
    def fit(self, clustered_rfmTable_km, y=None):
        return self
        
    def transform(self, clustered_rfmTable_km, y=None):
        clustered_rfmTable_km.to_excel(self.file_name, index=False)
        return clustered_rfmTable_km    
    
    
# %% Build the Pipeline

data_pipeline = Pipeline([
    ('ReadQuery', ReadQuery("ELECT userid, questionpaymentdate FROM database;")),
    ('DataPreparation', DataPreparation()),
    ('Model', Model()),
    ('SaveExcel', SaveExcel("Cluster_Result.xlsx"))
])


# %% Run the Pipeline

data_pipeline.fit_transform(None)