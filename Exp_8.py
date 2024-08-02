#!/usr/bin/env python
# coding: utf-8

# # Experiment-8

# In[1]:


#pip install pgmpy


# In[12]:


import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings


# In[3]:


heart = pd.read_csv(r"C:\Users\acer\Downloads\ml_dataset\Exp_8.csv")


# In[4]:


heart=heart.replace('?',np.nan)


# In[5]:


print(heart.head())


# In[15]:


model=BayesianModel([('age','trestbps'),('age','fbs'),('sex','trestbps'),('exang','trestbps'),('trestbps','target'),
                     ('fbs','target'),('target','restecg'),('target','thalach'),('target','chol')])


# In[7]:


model.fit(heart,estimator=MaximumLikelihoodEstimator)


# In[8]:


heart_infer=VariableElimination(model)


# In[9]:


q=heart_infer.query(variables=['target'],evidence={'age':40})


# In[10]:


print(q)


# In[11]:


q1=heart_infer.query(variables=['target'],evidence={'age':40,
                                                    'sex':1,
                                                   'trestbps':140,
                                                   'chol':211})
print(q1)


# In[ ]:




