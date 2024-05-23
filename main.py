import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load the data
data = pd.read_csv('data/wdbc.data', header=None)

# Name the columns
column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
data.columns = column_names

# Drop the ID column, as it is not relevant for classification
data.drop('ID', axis=1, inplace=True)

# Change the target variable to binary: 0 (Benign) and 1 (Malignant)
data['Diagnosis'] = data['Diagnosis'].map({'B': 0, 'M': 1})

# Separate the input variables (features) and the target variable
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate a model using cross-validation and print classification report and confusion matrix
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Prediction on the test data using cross-validation
    y_pred = cross_val_predict(model, X_test, y_test, cv=10)
    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n\n")

# KNN
print("KNN Results")
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, X_train, X_test, y_train, y_test)

# Decision Tree
print("Decision Tree Results")
dt = DecisionTreeClassifier(random_state=42)
evaluate_model(dt, X_train, X_test, y_train, y_test)

# SVM
print("SVM Results")
svm = SVC(kernel='linear', random_state=42)
evaluate_model(svm, X_train, X_test, y_train, y_test)

# Naive Bayes
print("Naive Bayes Results")
nb = GaussianNB()
evaluate_model(nb, X_train, X_test, y_train, y_test)