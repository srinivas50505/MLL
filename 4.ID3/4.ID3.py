import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('tennisdata.csv')
df_encoded = df.copy()
for column in df_encoded.columns:
    df_encoded[column] = df_encoded[column].astype('category').cat.codes


X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
tree_rules = export_text(clf, feature_names=list(X.columns))
print("Decision Tree:\n", tree_rules)

new_sample = pd.DataFrame({'Outlook': [0], 'Temperature': [1], 'Humidity': [0], 'Windy': [0]})
prediction = clf.predict(new_sample)
result = 'Yes' if prediction[0] == 1 else 'No'
print(f"Prediction for the new sample: {result}")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
