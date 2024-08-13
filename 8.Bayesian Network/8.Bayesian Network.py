import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings
heart= pd.read_csv("heartdisease.csv")
model=BayesianModel([
    ("age","trestbps"),("age","fbs"),("sex","trestbps"),("trestbps","target"),
    ("fbs","target"),("target","restecg"),("target","thalach"),("target","chol")
])
model.fit(heart,estimator=MaximumLikelihoodEstimator)
heart_infer=VariableElimination(model)
q=heart_infer.query(variables=["target"],evidence={"age":40})
print(q)
q1=heart_infer.query(variables=["target"],evidence={"age":40,"sex":1,
                                              "trestbps":140,
                                             "chol":211})
print(q1)
