import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)