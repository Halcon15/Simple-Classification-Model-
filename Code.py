import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Create a DataFrame for better manipulation
data = pd.DataFrame(
    data=iris.data, columns=iris.feature_names
)
data['target'] = iris.target

# Selecting sepal length and sepal width for simplicity
X = data[['sepal length (cm)', 'sepal width (cm)']]
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
for i, color in zip(range(3), ['blue', 'red', 'green']):
    plt.scatter(X_test[y_test == i]['sepal length (cm)'], X_test[y_test == i]['sepal width (cm)'], 
                color=color, label=iris.target_names[i])

plt.title('Logistic Regression: Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.grid()
plt.show()
