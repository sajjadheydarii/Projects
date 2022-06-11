"""
Created on Subject: Diagnosis of Patients with borderline personality disorder using machine learning techniques

@author: Sajad Heydari
"""
#
#%% Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from math import log2
from numpy import asarray

#%% Import the Dataset
data = pd.read_csv("total_data.csv").iloc[0:27,1:].drop("Hand", axis = 1)

#%% EDA
## Correlation Matrix
d1 = data.iloc[:,1:]
d2 = data.iloc[:,0]
d_t = pd.concat([d1,d2], axis = 1)

fig = plt.figure(figsize = (25,25))
sns.heatmap(d_t.corr(), vmin = -1, vmax = 1)

## Frequency of Age Values
fig = plt.figure(figsize = (10,10))
sns.countplot(x = "Age", data = data)

## Table Analysis
#1
data.iloc[:,[1,40,41]].groupby(["Sex"]).sum()

#2
df = data
df["drug_total"] = df["drug-1"] + df["drug-2"]

df.iloc[:,[1,-3,-2,-1]].groupby(["Sex"]).sum()

#3
# Set CSS properties for th elements in dataframe
th_props = [
  ('font-size', '14px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9')
  ]

# Set CSS properties for td elements in dataframe
td_props = [
  ('font-size', '14px'),
  ('text-align', 'center'),
  ('color', '#6d6d6d')
  
  ]

# Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

# Set colormap equal to seaborns light green color palette
cm = sns.light_palette("pink", as_cmap=True)

# Set colors and styles on DataFrame

(data.iloc[:,[1,51,52,53,54,55]].groupby(["Sex"]).sum().style
  .background_gradient(cmap=cm, subset=data.iloc[:,[51,52,53,54,55]].columns)
  .format({'CTQ-PhysAbuse': "{:}", 'CTQ-EmotAbuse': "{:}", 'CTQ-SexAbuse': "{:}", 'CTQ-PhysNeglect': "{:}", 'CTQ-EmotNeglect': "{:}"})
  .set_table_styles(styles))

#%% Data Preparation
##Pre-Processing
"""
We do not need any pre-processing ways.
Because our dataset is clean. But, I comment realted codes.
"""
#data.dtypes
#data.isnull().sum()
#fig = plt.figure(figsize = (30,15)) data.boxplot()
#data.duplicated().sum()

##Feature Selection
"""
We do not need any Feature Selection ways.
"""

#Feature Scaling
X = data.iloc[:,2:56]
binary = data.iloc[:,[1,-1,-2]]
Y = data.iloc[:,0]

sc = StandardScaler()
X_sc = sc.fit_transform(X)

df_sc = pd.DataFrame(X_sc, columns = X.columns)
data1 = pd.concat([df_sc, binary, Y], axis = 1)

#%% Splittting the Dataset into the Training set and the Test set
X_split = data1.iloc[:,:-1]
Y_split = data1.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X_split, Y_split, test_size = 0.4, random_state = 0)

#%% Training the Classification Models on the Training set
"""
We want to use various algorithms Such as:
    1. ElasticNet
    2. Lasso
    3. Adaptive Lasso  
"""
##ElasticNet
classifier = LogisticRegression(penalty = "elasticnet", solver = "saga", l1_ratio = 0.2)
classifier.fit(X_train, Y_train)

##Lasso
classifier1 = LogisticRegression(penalty = "l1", solver = "liblinear", random_state = 0)
classifier1.fit(X_train, Y_train)

##AdaptiveLasso
def Adaptive_LASSO(X_train, y_train, max_iterations = 1000, lasso_iterations = 10, alpha = 0.1, tol = 0.001, max_error_up = 5):

  #set checks
  higher = float('inf')
  lower = 0

  #set lists
  iterations_list = []

  #set variables
  X_train = X_train
  y_train = y_train

  #set constants
  alpha = alpha
  tol = tol
  max_iter = max_iterations
  n_lasso_iterations = lasso_iterations

  #Build mathematical formula
  g = lambda w: np.sqrt(np.abs(w))    #w is w
  gprime = lambda w: 1 / (2 * np.sqrt(np.abs(w)) + np.finfo(float).eps)

  n_samples, n_features = X_train.shape
  p_obj = lambda w: 1 / (2 * n_samples) * np.sum((y_train - np.dot(X_train, w)) ** 2) + alpha * np.sum(g(w))   # w is equal to beta

  #weighted features for feature selection
  weights = np.ones(n_features)
  X_w = X_train / weights[np.newaxis, :]
  X_w = np.nan_to_num(X_w)
  X_w = np.round(X_w, decimals = 3)

  y_train = np.nan_to_num(y_train)

  #Running lasso for once iteration
  adaptive_lasso = Lasso(alpha = alpha, fit_intercept = False)
  adaptive_lasso.fit(X_w, y_train)

  #Running lasso for multiple iteration
  for k in range(n_lasso_iterations):
    X_w = X_train / weights[np.newaxis, :]
    adaptive_lasso = Lasso(alpha = alpha, fit_intercept = False)
    adaptive_lasso.fit(X_w, y_train)
    coef_ = adaptive_lasso.coef_ / weights
    weights = gprime(coef_)

    iterations_list.append(k)

  return adaptive_lasso


#Final Model of Adaptive Lasso
model = Adaptive_LASSO(X_train, Y_train, max_iterations = 1000, lasso_iterations = 10, alpha = 0.1, tol = 0.001, max_error_up = 5)

#%% Predict the Test set Results
"""
We want to Predict the test set results of various algorithms Such as:
    1. ElasticNet
    2. Lasso
    3. Adaptive Lasso  
"""

##ElasticNet
y_pred = classifier.predict(X_test)

##Lasso
y_pred1 = classifier1.predict(X_test)

##Adaptive Lasso
y_pred2 = model.predict(X_test)
y_pred2 = np.ceil(y_pred2)

#%% Algoritms Evaluation methods
"""
We want to use evaluation methods such as:
    1. Confusion Matrix
    2. Accuracy Score
    3. Precision and Recall
"""
##ElasticNet
confusion_matrix(Y_test, y_pred)
accuracy_score(Y_test, y_pred)
precision_score(Y_test, y_pred)
recall_score(Y_test, y_pred)

##Lasso
confusion_matrix(Y_test, y_pred1)
accuracy_score(Y_test, y_pred1)
precision_score(Y_test, y_pred1)
recall_score(Y_test, y_pred1)

##AdaptiveLasso
confusion_matrix(Y_test, y_pred2)
accuracy_score(Y_test, y_pred2)
precision_score(Y_test, y_pred2)
recall_score(Y_test, y_pred2)

#%% Post-Processing
"""
We want to use Post-Processing methods such as:
    1. Logaritmic Loss
    2. Brier Score Loss
    3. ROC Curve
    4. Entropy (just for Y_test, not algorithms)
    5. ROC_AUC_SCORE
"""
#1
##ElasticNet
probs = classifier.predict_proba(X_test)

loss = log_loss(Y_test, probs)

##Lasso
probs1 = classifier1.predict_proba(X_test)

loss1 = log_loss(Y_test, probs1)

##AdaptiveLasso
"""
TOTALLY NOT DEFINED DUE TO STRUCTURE OF THE ALGORITHM WHICH NOT ACCEPT predict_proba METHOD
"""

#2
##ElasticNet
probs.reshape(-1,1)
probs_b_0 = probs[:,0]
loss_b_0 = brier_score_loss(Y_test, probs_b_0)

probs_b_1 = probs[:,1]
loss_b_1 = brier_score_loss(Y_test, probs_b_1)

print(loss_b_0, loss_b_1)

##Lasso
probs_b1_0 = probs1[:,0]
loss_b1_0 = brier_score_loss(Y_test, probs_b1_0)

probs_b1_1 = probs1[:,1]
loss_b1_1 = brier_score_loss(Y_test, probs_b1_1)

print(loss_b1_0, loss_b1_1)

##AdaptiveLasso
"""
TOTALLY NOT DEFINED DUE TO STRUCTURE OF THE ALGORITHM WHICH NOT ACCEPT predict_proba METHOD
"""

#3
##ElasticNet
fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for ElasticNet (using binary prediction)")

pyplot.show()


fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, probs_b_0)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for ElasticNet (using probabilistics prediction)")

pyplot.show()

##Lasso
fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, y_pred1)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for Lasso (using binary prediction)")

pyplot.show()

fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, probs_b1_0)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for Lasso (using probabilistics prediction)")

pyplot.show()

fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, probs_b1_1)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for Lasso (using probabilistics prediction)")

pyplot.show()

##AdaptiveLasso
"""
TOTALLY NOT DEFINED DUE TO STRUCTURE OF THE ALGORITHM WHICH NOT ACCEPT predict_proba METHOD
"""
fig = plt.figure(figsize = (10,10))
fpr, tpr, thresholds = roc_curve(Y_test, y_pred2)

pyplot.plot([0,1],[0,1], linestyle = "--", color = "red")
pyplot.plot(fpr,tpr, color = "blue")

plt.title("ROC Curve for Adaptive Lasso (using binary prediction)")

pyplot.show()

#4
def entropy(p):
  return -sum([p[i] * log2(p[i]) for i in range(len(p))])

x = asarray(Y_test) + 1e-15
entropy(x)

#5
##ElasticNet
print(roc_auc_score(Y_test, y_pred), roc_auc_score(Y_test,probs_b_0), roc_auc_score(Y_test,probs_b_1))

##Lasso
print(roc_auc_score(Y_test, y_pred1), roc_auc_score(Y_test,probs_b1_0), roc_auc_score(Y_test,probs_b1_1))
