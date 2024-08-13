import numpy as np


X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)  


X = X / np.amax(X, axis=0) 
y = y / 100                 


inputSize = 2
hiddenSize = 3
outputSize = 1
learning_rate = 0.01  


W1 = np.random.randn(inputSize, hiddenSize) * 0.1   
W2 = np.random.randn(hiddenSize, outputSize) * 0.1  

def sigmoid(s):
    return 1 / (1 + np.exp(-s)) 

def sigmoidPrime(s):
    return s * (1 - s)  


def forward(X, W1, W2):
    z = np.dot(X, W1)         
    z2 = sigmoid(z)            
    z3 = np.dot(z2, W2)       
    o = sigmoid(z3)            
    return o


def backward(X, y, W1, W2, o):
    o_error = y - o                          
    o_delta = o_error * sigmoidPrime(o)     
    z2_error = o_delta.dot(W2.T)            
    z2_delta = z2_error * sigmoidPrime(np.dot(X, W1))  
    
    W1 += learning_rate * X.T.dot(z2_delta)  
    W2 += learning_rate * np.dot(sigmoid(np.dot(X, W1)).T, o_delta) 
    
    return W1, W2


for i in range(1000):  
    o = forward(X, W1, W2)
    W1, W2 = backward(X, y, W1, W2, o)

# Results
print("\nInput: \n" + str(X))
print("\nActual Output: \n" + str(y))
print("\nPredicted Output: \n" + str(forward(X, W1, W2)))
print("\nLoss: \n" + str(np.mean(np.square(y - forward(X, W1, W2)))))  
