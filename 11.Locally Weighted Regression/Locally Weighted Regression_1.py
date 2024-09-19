import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
x=np.linspace(0,10,100)
y=np.sin(x)+np.random.normal(scale=0.2,size=x.shape)
tau=0.1
predictions=[]
for xi in x:
    weights=np.exp(-(x-xi)**2/2*tau)
    coefficients=np.polyfit(x,y,1,w=weights)
    prediction=np.polyval(coefficients,xi)
    predictions.append(prediction)
plt.scatter(x,y,label="datapoints")
plt.plot(x,predictions,color="red",label="LWR")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LWR")
plt.legend()
plt.show()
