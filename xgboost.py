
# coding: utf-8

# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset = pd.read_csv('C:\\Users\\VIPUL\\Downloads\\Polynomial-Linear-Regression-master (1)\\Polynomial-Linear-Regression-master\\Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
print(dataset.head())


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[11]:


from xgboost import XGBClassifier
cls=XGBClassifier()
cls.fit(X_train,y_train)


# In[12]:


y_pred=cls.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[13]:


#Accuracy of model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100,'%')

