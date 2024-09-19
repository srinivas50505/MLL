from sklearn.neural_network import MLPRegressor
import numpy as np
x=np.array([[2,9],[1,3],[3,5]],dtype=float)
y=np.array([[92],[86],[89]],dtype=float)
x=x/np.max(x,axis=0)
y=y/100
model = MLPRegressor(hidden_layer_sizes=(5,),activation="logistic",
                     solver="adam",learning_rate_init=0.001,
                     max_iter=5000)
model.fit(x,y.ravel())
pred= model.predict(x).reshape(-1,1)
print("predition:\n",np.round(pred,2))
print("loss:",np.round(np.mean(np.square(y-pred)),4))
