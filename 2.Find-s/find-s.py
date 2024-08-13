import pandas as pd
import numpy as np

data = pd.read_csv("trainingdata.csv")
concepts = np.array(data)[:, :-1]
target = np.array(data)[:, -1]

def train(concepts, target):
   
    for i, val in enumerate(target):
        if val == "Yes":
            specific = concepts[i].copy()
            print("Specific hypothesis initialized as:", specific)
            break

    if specific is not None:
        for i, val in enumerate(concepts):
            if target[i] == "Yes":
                for j in range(len(specific)):
                    if val[j] != specific[j]:
                        specific[j] = "?"
                    else:
                        pass
                print(specific)

    return specific
specific_hypothesis = train(concepts, target)
print("Final specific hypothesis:", specific_hypothesis)
