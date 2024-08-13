import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'iris.csv'
iris_df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal_length', y='sepal_width', hue='species', palette='viridis')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(data=iris_df, x=iris_df.index, y='sepal_length', hue='species', palette='viridis')
plt.title('Line Plot of Sepal Length over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=iris_df, x='petal_length', bins=30, kde=True, hue='species', palette='viridis')
plt.title('Histogram of Petal Lengths')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()



file_path1 = 'winemag-data_first150k.csv'
wine_reviews = pd.read_csv(file_path1)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=wine_reviews, x='points', y='price')
plt.title('Scatter Plot of Wine Points vs Price')
plt.xlabel('Points')
plt.ylabel('Price')
plt.show()

wine_reviews_sorted = wine_reviews.sort_values('price')

plt.figure(figsize=(10, 6))
sns.lineplot(data=wine_reviews_sorted, x=wine_reviews_sorted.index, y='points')
plt.title('Line Plot of Wine Points Sorted by Price')
plt.xlabel('Index')
plt.ylabel('Points')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data=wine_reviews, x='price', bins=30, kde=True)
plt.title('Histogram of Wine Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
