import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 100)  
y = np.sin(X) + np.random.normal(scale=0.2, size=X.shape)  
tau = 1.0
predictions = np.zeros_like(X)

for i in range(len(X)):
    xi = X[i]
    weights = np.exp(-((X - xi) ** 2) / (2 * tau ** 2))
    W = np.diag(weights)
    X_w = np.vstack([np.ones(len(X)), X]).T
    theta = np.linalg.pinv(X_w.T @ W @ X_w) @ X_w.T @ W @ y
    predictions[i] = np.hstack([1, xi]) @ theta

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')  
plt.plot(X, predictions, color='red', label='LWR Fit')  
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
