#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
parkinsons_data = pd.read_csv("parkinsons.data")
parkinsons_data.head()


# In[2]:


print(parkinsons_data.shape)
parkinsons_data.info()


# In[3]:


parkinsons_data.isnull().sum()


# In[5]:


parkinsons_data.describe()


# In[6]:


parkinsons_data.drop(columns = ["name"]).groupby("status").mean()


# In[7]:


X = parkinsons_data.drop(columns = ["name", "status"], axis = 1)
Y = parkinsons_data["status"]
print("X\n",X)
print("Y\n",Y)


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[9]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


# Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel="linear")
svm_model.fit(X_train_scaled, Y_train)


# In[12]:


# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)
rf_model.fit(X_train_scaled, Y_train)


# In[13]:


# Logistic Regression Classifier
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, Y_train)


# In[14]:


# k-Nearest Neighbors (k-NN) Classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, Y_train)


# In[15]:


# Evaluate Models
def evaluate_model(model, X_train, X_test, Y_train, Y_test, model_name):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(test_predictions, Y_test)
    print(f"\n{model_name} Model:")
    print("Accuracy: ", test_accuracy)


# In[16]:


# Evaluate all models
evaluate_model(svm_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "SVM")
evaluate_model(rf_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "Random Forest")
evaluate_model(lr_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "Logistic Regression")
evaluate_model(knn_model, X_train_scaled, X_test_scaled, Y_train, Y_test, "k-NN")


# In[17]:


# Example Prediction
input_data = parkinsons_data.iloc[42].drop(["status", "name"]).values
input_data_reshaped = input_data.reshape(1, -1)
input_data_scaled = scaler.transform(input_data_reshaped)


# In[18]:


svm_prediction = svm_model.predict(input_data_scaled)
rf_prediction = rf_model.predict(input_data_scaled)
lr_prediction = lr_model.predict(input_data_scaled)
knn_prediction = knn_model.predict(input_data_scaled)


# In[19]:


print("\nPrediction for SVM:")
print("The person has Parkinson's Disease." if svm_prediction[0] == 1 else "The person does not have Parkinson's Disease.")

print("\nPrediction for Random Forest:")
print("The person has Parkinson's Disease." if rf_prediction[0] == 1 else "The person does not have Parkinson's Disease.")

print("\nPrediction for Logistic Regression:")
print("The person has Parkinson's Disease." if lr_prediction[0] == 1 else "The person does not have Parkinson's Disease.")

print("\nPrediction for k-Nearest Neighbors:")
print("The person has Parkinson's Disease." if knn_prediction[0] == 1 else "The person does not have Parkinson's Disease.")


# In[ ]:




