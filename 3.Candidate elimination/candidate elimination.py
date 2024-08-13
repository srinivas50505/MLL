import numpy as np 
import pandas as pd

data = pd.read_csv("trainingdata.csv")
concepts = np.array(data.iloc[:, :-1]) 
target = np.array(data.iloc[:, -1])     


specific_h = concepts[0].copy()
general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]


for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for j in range(len(specific_h)):
            if h[j] != specific_h[j]:
                specific_h[j] = "?"
                general_h[j][j] = "?"
    elif target[i] == "No":
        for j in range(len(specific_h)):
            if h[j] != specific_h[j]:
                general_h[j][j] = specific_h[j]
            else:
                general_h[j][j] = "?"


general_h = [h for h in general_h if h != ["?" for _ in range(len(specific_h))]]

print("Specific Hypothesis is:", specific_h)
print("General Hypotheses are:", general_h)
