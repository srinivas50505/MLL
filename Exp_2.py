#!/usr/bin/env python
# coding: utf-8

# In[21]:


import csv
print(pandas.read_csv(r'C:\Users\acer\Downloads\ml_dataset\Exp_2.csv'))
h=['0'for i in range(6)]
with open(r"C:\Users\acer\Downloads\ml_dataset\Exp_2.csv") as f:
    data=csv.reader(f)
    data=list(data)
    
    for i in data:
        if i[-1]=="Yes":
            for j in range(6):
                if h[j]=='0':
                    h[j]=i[j]
                elif h[j]!=i[j]:
                    h[j]='?'

    print("\n",h)


# In[ ]:




