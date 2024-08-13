import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Correct and Wrong Predictions:\n")
for i in range(len(y_test)):
    actual = y_test[i]
    predicted = y_pred[i]
    if actual == predicted:
        print(f"Correct: Actual={iris.target_names[actual]}, Predicted={iris.target_names[predicted]}")
    else:
        print(f"Wrong: Actual={iris.target_names[actual]}, Predicted={iris.target_names[predicted]}")


accuracy = accuracy_score(y_test, y_pred)


print("Accuracy:",accuracy)

