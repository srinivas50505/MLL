import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
data=pd.read_csv("tennisdata.csv")
print(data.head())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

le_outlook=LabelEncoder()
x.Outlook=le_outlook.fit_transform(x.Outlook)

le_temperature=LabelEncoder()
x.Temperature=le_outlook.fit_transform(x.Temperature)

le_humidity=LabelEncoder()
x.Humidity=le_outlook.fit_transform(x.Humidity)

le_windy=LabelEncoder()
x.Windy=le_outlook.fit_transform(x.Windy)

le_playtennis=LabelEncoder()
y=le_playtennis.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
classifier=GaussianNB()
classifier.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
print("Accuarcy : ",accuracy_score(classifier.predict(x_test),y_test))
