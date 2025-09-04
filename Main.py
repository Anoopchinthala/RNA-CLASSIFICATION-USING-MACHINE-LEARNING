#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# In[2]:

data = pd.read_csv("data_train.tsv", delimiter="\t")
data1 = pd.read_csv("data_test.tsv", delimiter="\t")
data = pd.concat([data, data1])
data = data.sample(frac=0.2, random_state=0)
data["Class"].value_counts()

# In[3]:

train_data = data["Sequence"]
train_label = data["Class"]
print(train_data)

# In[4]:

max_length = 0
min_length = 99999
length_list = []
for i in train_data.values:
    length = len(i)
    length_list.append(length)
    if length > max_length:
        max_length = length
    
    if length < min_length:
        min_length = length

print(min_length, max_length)

# In[5]:

sns.distplot(length_list)
plt.xlim([0, 8000])
plt.xlabel('Sequence count')

# In[6]:

LENGTH = 27610
list_data = []
for seq in tqdm(train_data.values):
    temp_list = [0]*LENGTH
    for ind in range(len(seq)):
        if ind >= LENGTH:
            break
        temp_list[ind] = ord(seq[ind])
    list_data.append(temp_list)
print("Data Ready")

# In[7]:

new_train_data = pd.DataFrame(list_data)
print(new_train_data)

# In[8]:

X_train, X_test, y_train, y_test = train_test_split(new_train_data, train_label, test_size=0.50, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# In[9]:

#DECISION TREE
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_train, y_train)

# In[10]:

dt_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_fp, dt_tp, _ = roc_curve(y_test, dt_pred)

print("Decision tree Accuracy:", dt_acc*100, "%")
print("Decision tree MSE:", dt_mse)
print("Decision tree MAE:", dt_mae)
print("Decision tree r2:", dt_r2)
print("Decision tree f1:", dt_f1)

# In[11]:

#decision tree confusion matrix
conf_matrix = confusion_matrix(y_test, dt_pred) 
plt.figure(figsize =(6, 6)) 
sns.heatmap(conf_matrix, xticklabels = ['PCT / mRNA', 'LNC'],  
            yticklabels = ['PCT / mRNA', 'LNC'], annot = True); 
plt.title("Confusion Matrix (Decision Tree)") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 

# In[12]:

#ROC od Decision Tree
plt.subplots(1, figsize=(10,10))
plt.title('ROC - Decision Tree')
plt.plot(dt_fp, dt_tp)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[13]:

#RANDOM FOREST
rf_clf = RandomForestClassifier(random_state = 0)
rf_clf.fit(X_train, y_train)

# In[14]:

rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_fp, rf_tp, _ = roc_curve(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc*100, "%")
print("Random Forest MSE:", rf_mse)
print("Random Forest MAE:", rf_mae)
print("Random Forest r2:", rf_r2)
print("Random Forest f1:", rf_f1)

# In[15]:

#Confusion matrix of random forest
conf_matrix = confusion_matrix(y_test, rf_pred) 
plt.figure(figsize =(6, 6)) 
sns.heatmap(conf_matrix, xticklabels = ['PCT / mRNA', 'LNC'],  
            yticklabels = ['PCT / mRNA', 'LNC'], annot = True); 
plt.title("Confusion Matrix (Random Forest)") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()

# In[16]:

#ROC of Random forest
plt.subplots(1, figsize=(10,10))
plt.title('ROC - Random Forest')
plt.plot(rf_fp, rf_tp)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[17]:

#LOGISTIC REGRESSION
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

# In[18]:

lr_pred = lr_clf.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_fp, lr_tp, _ = roc_curve(y_test, lr_pred)

print("LR Classifier Accuracy:", lr_acc*100, "%")
print("LR Classifier MSE:", lr_mse)
print("LR Classifier MAE:", lr_mae)
print("LR Classifier r2:", lr_r2)
print("LR Classifier f1:", lr_f1)

# In[19]:

#CONFUSION matrix of logistic regression
conf_matrix = confusion_matrix(y_test, lr_pred) 
plt.figure(figsize =(6, 6)) 
sns.heatmap(conf_matrix, xticklabels = ['PCT / mRNA', 'LNC'],  
            yticklabels = ['PCT / mRNA', 'LNC'], annot = True); 
plt.title("Confusion Matrix (LR)") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 

# In[20]:

#ROC of logistic regression
plt.subplots(1, figsize=(10,10))
plt.title('ROC - Logistic Regression')
plt.plot(lr_fp, lr_tp)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[21]:

from xgboost import XGBClassifier

# In[23]:

from xgboost import XGBClassifier

# In[25]:

import xgboost as xgb

# In[26]:

from xgboost import XGBClassifier

# In[27]:

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_fp, xgb_tp, _ = roc_curve(y_test, xgb_pred)

print("XGB Classifier Accuracy:", xgb_acc*100, "%")
print("XGB Classifier MSE:", xgb_mse)
print("XGB Classifier MAE:", xgb_mae)
print("XGB Classifier r2:", xgb_r2)
print("XGB Classifier f1:", xgb_f1)

# In[28]:

#Confusion matrix of XG Boost
conf_matrix = confusion_matrix(y_test, xgb_pred) 
plt.figure(figsize =(6, 6)) 
sns.heatmap(conf_matrix, xticklabels = ['PCT / mRNA', 'LNC'],  
            yticklabels = ['PCT / mRNA', 'LNC'], annot = True); 
plt.title("Confusion Matrix (XG Boost)") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 

# In[29]:

#ROC of XG Boost
plt.subplots(1, figsize=(10,10))
plt.title('ROC - XG Boost')
plt.plot(xgb_fp, xgb_tp)
plt.plot([0, 1], ls="--")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[31]:

#Metrics
print("Accuracy of decision tree:",dt_acc)
print("Accuracy of random forest:",rf_acc)
print("Accuracy of logistic regression:",lr_acc)
print("Accuracy of XG Boost:",xgb_acc)
plt.figure(figsize=(20, 10)) 
accplot = sns.barplot(x=['DT', 'RF', 'LR', 'XGB'], y=[dt_acc, rf_acc, lr_acc, xgb_acc])
accplot.set_ylim(0.70, 0.85)
accplot.set_title("Accuracy Comparison")
accplot.set_xlabel("Models")
accplot.set_ylabel("Accuracy")

# In[32]:

#metrics comparion on mse
print("Mean Squared Error of decision tree:",dt_mse)
print("Mean Squared Error of random forest:",rf_mse)
print("Mean Squared Errorof logistic regression:",lr_mse)
print("Mean Squared Errorof XG Boost:",xgb_mse)
plt.figure(figsize=(20, 10)) 
mseplot = sns.barplot(x=['DT', 'RF', 'LR', 'XGB'], y=[dt_mse, rf_mse, lr_mse, xgb_mse])
mseplot.set_title("MSE Comparison")
mseplot.set_xlabel("Models")
mseplot.set_ylabel("MSE")

# In[33]:

#metrics comparision on mae
print("Mean Absolute Error of decision tree:",dt_mae)
print("Mean Absolute  Error of random forest:",rf_mae)
print("Mean Absolute  Errorof logistic regression:",lr_mae)
print("Mean Absolute Errorof XG Boost:",xgb_mae)
plt.figure(figsize=(20, 10)) 
maeplot = sns.barplot(x=['DT', 'RF', 'LR', 'XGB'], y=[dt_mae, rf_mae, lr_mae, xgb_mae])
maeplot.set_title("MAE Comparison")
maeplot.set_xlabel("Models")
maeplot.set_ylabel("MAE")

# In[34]:

#metric comparision on r2 score
print("r2 score of decision tree:",dt_r2)
print("r2 scoreof random forest:",rf_r2)
print("r2 score of logistic regression:",lr_r2)
print("r2 score of XG Boost:",xgb_r2)
plt.figure(figsize=(20, 10)) 
r2plot = sns.barplot(x=['DT', 'RF', 'LR', 'XGB'], y=[dt_r2, rf_r2, lr_r2, xgb_r2])
r2plot.set_title("r2 Comparison")
r2plot.set_xlabel("Models")
r2plot.set_ylabel("r2")

# In[36]:

#metric comparion on f1 score

print("f1 score of decision tree:",dt_f1)
print("f1 scoreof random forest:",rf_f1)
print("f1 score of logistic regression:",lr_f1)
print("f1 score of XG Boost:",xgb_f1)

plt.figure(figsize=(20, 10)) 
f1plot = sns.barplot(x=['DT', 'RF', 'LR', 'XGB'], y=[dt_f1, rf_f1, lr_f1, xgb_f1])
f1plot.set_title("f1 Comparison")
f1plot.set_xlabel("Models")
f1plot.set_ylabel("f1")

# In[37]:

plt.subplots(1, figsize=(10,10))
plt.title('ROC Comparison')
plt.plot(dt_fp, dt_tp)
plt.plot(rf_fp, rf_tp)
plt.plot(lr_fp, lr_tp)
plt.plot(xgb_fp, xgb_tp)
plt.plot([0, 1], ls="--")

plt.legend(['DT', 'RF', 'LR', 'XGB', 'Base'])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[ ]:




