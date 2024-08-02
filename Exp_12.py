#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()
iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['species']=iris.target


# In[3]:


plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
sns.scatterplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)', hue='species')
plt.title('IRIS Dataset-Scatter Plot')
plt.subplot(1,3,2)
iris_df[['sepal length (cm)','sepal width (cm)']].plot(ax=plt.gca())
plt.title('IRIS Dataset-Line Chart')
plt.subplot(1,3,3)
iris_df['sepal length (cm)'].plot(kind='hist',bins=20,ax=plt.gca())
plt.title('IRIS Dataset-Histogram')


# In[4]:


wine = pd.read_csv(r"C:\Users\acer\Downloads\ml_dataset\Exp_12.csv")


# In[5]:


print(wine.dtypes)


# In[6]:


fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Scatter Plot using seaborn
sns.scatterplot(x='rating', y='price', data=wine, ax=axs[0])
axs[0].set_title('Scatter Plot (Rating Vs Price)')
axs[0].set_xlabel('Rating')
axs[0].set_ylabel('Price')

# Line Chart using matplotlib
axs[1].plot(wine['rating'], wine['price'], color='blue', marker='o', linestyle='-', label='Int vs Float')
axs[1].set_title('Line Chart (Rating Vs Price)')
axs[0].set_xlabel('Rating')
axs[0].set_ylabel('Price')
axs[1].legend()

# Histogram using seaborn
sns.histplot(wine['rating'], bins=3, ax=axs[2], color='blue', label='Rating', kde=False, stat='frequency')
sns.histplot(wine['price'], bins=3, ax=axs[2], color='red', label='Price', kde=False, stat='frequency')
axs[2].set_title('Histogram (Rating Vs Price)')
axs[0].set_xlabel('Rating')
axs[0].set_ylabel('Price')
axs[2].legend()

# Adjust layout and show plots
plt.tight_layout()
plt.show()


# In[ ]:




