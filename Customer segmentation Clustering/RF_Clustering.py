# %% Import the Libraries
import pyodbc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import numpy.stats as ns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score


# %% Import the Dataset (From SQL Server)
server = 'server_address, port_number' 
database = 'database_name' 
username = 'user_name' 
password = 'your_password'  
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
query = "SELECT * FROM Database;"
df = pd.read_sql(query, cnxn)

# %% Data Preparation

# Data Integration

""" 1 """
df['date'] = pd.DatetimeIndex(df.questionpaymentdate).date
df = df.drop("questionpaymentdate", axis = 1)

""" 2 """
df_recency = df.groupby(['userid'],as_index=False)['date'].max()
df_recency.columns = ['userid','Last_Purchase_Date']
df_recency['Recency'] = df_recency.Last_Purchase_Date.apply(lambda x:(dt.date(2023,3,21) - x).days)
df_recency.drop(columns=['Last_Purchase_Date'],inplace=True)

""" 3 """
rfmTable = df.groupby('userid').agg({ 'userid': lambda x: len(x)}) # Frequency

rfmTable.rename(columns={'userid': 'frequency'}, inplace = True)

""" 4 """
RFM_Table = df_recency.merge(rfmTable,left_on='userid',right_on='userid').set_index("userid")

# EDA
fig = plt.figure(figsize = (8,8))
sns.heatmap(RFM_Table.corr(), vmin = -1, vmax = 1, annot = True)


def check_skew(df_skew, column):
    skew = ns.skew(df_skew[column])
    skewtest = ns.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return
plt.subplot(3, 1, 1)
check_skew(RFM_Table,'Recency')
plt.subplot(3, 1, 2)
check_skew(RFM_Table,'Frequency')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)

# Handle Missing Values
rfmTable.isnull().sum()

# Feature Scaling for Outlier Detection
""" Z-Score """
sc = StandardScaler()
RFM_Table_sc = pd.DataFrame(sc.fit_transform(rfmTable), columns = rfmTable.columns)


# Handle Outliers
RFM_Table.boxplot()

""" With Z-Score """
RFM_Table_sc[(RFM_Table_sc["frequency"] < -3) | (RFM_Table_sc["frequency"] > 3)].index
rfmTable1 = RFM_Table.reset_index().drop(RFM_Table_sc[(RFM_Table_sc["frequency"] < -3) | (RFM_Table_sc["frequency"] > 3)].index, axis = 0).set_index("userid")

# %% Training the Dataset on RFM Model with considering Frequency and Monetary

# Determining RFM Quantiles
quantiles = rfmTable1.quantile(q = [0.25,0.5,0.75]).to_dict()

# Creating the RFM segmentation table

""" Arguments (x = value, p = recency, frequency, k = quartiles dict) """
def FClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
rfmTable1['F_Quantile'] = rfmTable1['frequency'].apply(FClass, args=('frequency',quantiles,))
rfmTable1['R_quantile'] = rfmTable1['Recency'].apply(RScore, args=('Recency',quantiles))
rfmTable1['RFClass'] = rfmTable1.F_Quantile.map(str) + rfmTable1.R_quantile.map(str)

# Getting the total score for each User
rfmTable1['Total Score'] = rfmTable1['F_Quantile'] + rfmTable1['R_quantile']

# %% Training the RFM results on Clustering Algorithms


# K-Means
""" Using MinMax to employ K-Means """
mm = MinMaxScaler()
X_mm = pd.DataFrame(mm.fit_transform(rfmTable1), columns = rfmTable1.columns)

""" Elbow Method for finding the best number of clusters in K-Means """
WCSS = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X_mm)
    WCSS.append(kmeans.inertia_)

fig = plt.figure(figsize = (8,8))
plt.plot(range(1,20), WCSS)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.xticks([j for j in range(1,20)])
plt.show()

""" determining the cluster number with Silhouette method """
for i in range(2,20):
    model = KMeans(n_clusters = i, init = "k-means++")
    preds = model.fit_predict(X_mm)
    centers = model.cluster_centers_

    score = silhouette_score(X_mm, preds).round(3)
    print("Untuk k={}, silhouette score= {}".format(i, score))
    
""" Training the Model """
kmeans_cluster = KMeans(n_clusters = 4, max_iter = 200)
y_kmeans = kmeans_cluster.fit_predict(X_mm)  

""" Final Result """
y_kmeans_df = pd.DataFrame(y_kmeans, columns = ["cluster"])
clustered_rfmTable_km = pd.concat([rfmTable1, y_kmeans_df], axis = 1)


# DBSCAN
""" Validate Epsilon as a Hyperparameter"""

neighbors = NearestNeighbors(n_neighbors = 6)
neighbors_fit = neighbors.fit(X_mm)
distances, indices = neighbors_fit.kneighbors(X_mm)

fig = plt.figure(figsize = (6,6))
distances = np.sort(distances, axis = 0)
distances = distances[:,1]
plt.plot(distances)

""" Training the Algorithm """
y_dbscan = DBSCAN(eps = 0.2,min_samples = 6).fit_predict(X_mm)

""" Final Result """
y_dbscan_df = pd.DataFrame(y_dbscan, columns = ["cluster"])
clustered_rfmTable_dbs = pd.concat([rfmTable1, y_dbscan_df], axis = 1)


# BIRCH (The Balance Iterative Reducing and Clustering using Hierarchies)

""" Define the model """
y_birch = Birch(branching_factor = 50, n_clusters = None, threshold = 0.1).fit_predict(X_mm)

""" Final Result """
clustered_rfmTable_bir = pd.concat([rfmTable1, y_birch], axis = 1)


# %% Evaluation of Clusters

# Davis-Bouldin
db_score_km = davies_bouldin_score(X_mm, y_kmeans)

db_score_dbs = davies_bouldin_score(X_mm, y_dbscan)

db_score_bir = davies_bouldin_score(X_mm, y_birch)

# %% visualising clusters

fig = plt.figure(figsize = (8,8))
plt.scatter(X_mm[y_kmeans == 0, 0], X_mm[y_kmeans == 0, 1], s = 10, c = "red", label = "Cluster0")
plt.scatter(X_mm[y_kmeans == 1, 0], X_mm[y_kmeans == 1, 1], s = 10, c = "green", label = "Cluster1")
plt.scatter(X_mm[y_kmeans == 2, 0], X_mm[y_kmeans == 2, 1], s = 10, c = "darkviolet", label = "Cluster2")
plt.scatter(X_mm[y_kmeans == 3, 0], X_mm[y_kmeans == 3, 1], s = 10, c = "blue", label = "Cluster3")

#visualising centroids
plt.scatter(kmeans_cluster.cluster_centers_[:,0], kmeans_cluster.cluster_centers_[:,1], s = 100, c = "yellow", label = "centroid")

#Better visualisation
plt.title("Clusters of customers")
plt.xlabel("Frequency")
plt.ylabel("Recency")
plt.legend()
plt.show()


# %% Save Final Result
clustered_rfmTable_km.to_excel("Cluster_result.xlsx")  




