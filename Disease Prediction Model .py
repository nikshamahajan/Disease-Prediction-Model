#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


get_ipython().system('pip install shap')


# In[2]:


# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


train=pd.read_csv("Training.csv")
test=pd.read_csv("Testing.csv")


# In[4]:


#Dataset Loading and Checking
train.head()


# In[5]:


test.head()


# In[6]:


print("Number of observations in training set:", train.shape[0])
print("Number of features in training set:", train.shape[1])
print("Number of unique diseases in training set:", train['prognosis'].nunique())
print("\nInformation about the dataset:")
print(train.info())


# In[7]:


#Data Preprocessing
train=train.drop(["Unnamed: 133"],axis=1)


# In[8]:


train.prognosis.value_counts()


# In[9]:


train.isna().sum()


# In[10]:


test.isna().sum()


# In[11]:


P = train[["prognosis"]]
X = train.drop(["prognosis"],axis=1)
Y = test.drop(["prognosis"],axis=1)


# In[12]:


xtrain,xtest,ytrain,ytest = train_test_split(X,P,test_size=0.2,random_state=42)


# In[13]:


#Model - Random Forest
rf= RandomForestClassifier(random_state=42)
model_rf = rf.fit(xtrain,ytrain)
tr_pred_rf = model_rf.predict(xtrain)
ts_pred_rf = model_rf.predict(xtest)

print("training accuracy is:",accuracy_score(ytrain,tr_pred_rf))
print("testing accuracy is:",accuracy_score(ytest,ts_pred_rf))


# In[14]:


#Hyperparameter Optimization with Grid Search
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.transform(Y)

# Flatten the target variable using ravel()
P_flat = train['prognosis'].values.ravel()

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, P_flat)

# Display best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)


# In[15]:


#Model Evaluation after Hyperparameter Tuning
xtrain_scaled, xtest_scaled, ytrain_scaled, ytest_scaled = train_test_split(X_scaled, P_flat, test_size=0.2, random_state=42)

# Instantiate RandomForestClassifier with the best hyperparameters from GridSearchCV
best_rf = RandomForestClassifier(random_state=42, **grid_search.best_params_)

# Fit the model on the training set
best_rf.fit(xtrain_scaled, ytrain_scaled)

# Make predictions on the training and testing sets
tr_pred_rf_scaled = best_rf.predict(xtrain_scaled)
ts_pred_rf_scaled = best_rf.predict(xtest_scaled)

# Evaluate the model
print("Training accuracy after hyperparameter tuning:", accuracy_score(ytrain_scaled, tr_pred_rf_scaled))
print("Testing accuracy after hyperparameter tuning:", accuracy_score(ytest_scaled, ts_pred_rf_scaled))


# In[16]:


# Confusion Matrix
conf_matrix = confusion_matrix(ytest_scaled, ts_pred_rf_scaled)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(ytest_scaled, ts_pred_rf_scaled))


# In[18]:


# Create a SHAP explainer
explainer = shap.TreeExplainer(best_rf)

# Calculate SHAP values
shap_values = explainer.shap_values(X_scaled)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_scaled)


# In[ ]:




