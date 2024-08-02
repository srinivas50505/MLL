#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sixv
import sys
sys.modules['sklearn.externals.six'] = six


# In[2]:


import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import numpy as np


# In[3]:


data = pd.read_csv(r'C:\Users\acer\Downloads\ml_dataset\Exp_4.csv')
print("The first 5 values of data is \n",data.head())


# In[4]:


X = data.iloc[:,:-1]
print("\nThe first 5 values of Train data is \n",X.head())


# In[5]:


y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is \n",y.head())


# In[6]:


le_outlook = LabelEncoder()
X.Outlook =  le_outlook.fit_transform(X.Outlook)
le_Temperature = LabelEncoder()
X.Temperature =  le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity =  le_Humidity.fit_transform(X.Humidity)
le_Windy = LabelEncoder()
X.Windy =  le_Windy.fit_transform(X.Windy)

print("\nNow the Train data is",X.head())


# In[7]:


le_PlayTennis = LabelEncoder()
y =  le_PlayTennis.fit_transform(y)
print("\nNow the Train data is\n",y)


# In[8]:


classifier = DecisionTreeClassifier()
classifier.fit(X,y)


# In[9]:


def labelEncoderForInput(list1):
    list1[0] =  le_outlook.transform([list1[0]])[0]
    list1[1] =  le_Temperature.transform([list1[1]])[0]
    list1[2] =  le_Humidity.transform([list1[2]])[0]
    list1[3] =  le_Windy.transform([list1[3]])[0]
    return [list1]

inp = ["Rainy","Mild","High","False"]
inp1=["Rainy","Cool","High","False"]
pred1 = labelEncoderForInput(inp1)
y_pred = classifier.predict(pred1)
y_pred = np.ravel(y_pred)
print("\n for input {0}, we obtain {1}".format(inp1, le_PlayTennis.inverse_transform(y_pred[0])))


# In[15]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your dataset
data = pd.read_csv(r'C:\Users\acer\Downloads\ml_dataset\Exp_4.csv')
print("The first 5 values of data is \n", data.head())

# Separate features (X) and target variable (y)
X = data.drop(columns=['PlayTennis'])
y = data['PlayTennis']

# Initialize LabelEncoders
le_outlook = LabelEncoder()
le_Temperature = LabelEncoder()
le_Humidity = LabelEncoder()
le_Windy = LabelEncoder()
le_PlayTennis = LabelEncoder()

# Fit and transform categorical features in X
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])
X['Temperature'] = le_Temperature.fit_transform(X['Temperature'])
X['Humidity'] = le_Humidity.fit_transform(X['Humidity'])
X['Windy'] = le_Windy.fit_transform(X['Windy'])

# Fit and transform target variable y
y = le_PlayTennis.fit_transform(y)

# Initialize and train the Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# Function to perform label encoding for input
def labelEncoderForInput(list1):
    # Convert categorical values to encoded integers
    list1[0] = le_outlook.transform([list1[0]])[0]
    list1[1] = le_Temperature.transform([list1[1]])[0]
    list1[2] = le_Humidity.transform([list1[2]])[0]
    list1[3] = le_Windy.transform([list1[3]])[0]
    return list1

inp = ["Rainy", "Mild", "High", "False"]
inp1 = ["Rainy", "Cool", "High", "False"]

# Encoding inp1
pred1 = labelEncoderForInput(inp1)

# Making prediction
pred1 = np.array(pred1).reshape(1, -1)  # Reshape to a 2D array
y_pred = classifier.predict(pred1)

# Printing the result
print("\nFor input {0}, we obtain {1}".format(inp1, le_PlayTennis.inverse_transform([y_pred])[0]))


# In[ ]:




