from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Iris dataset
df = pd.read_csv("./data/data_banknote_authentication.csv")
X, y = df.drop(columns=['Classifier']), df['Classifier']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ID3 decision tree classifier (uses entropy as the splitting criterion)
id3_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Train the model
id3_tree.fit(X_train, y_train)

# Make predictions
y_pred = id3_tree.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")