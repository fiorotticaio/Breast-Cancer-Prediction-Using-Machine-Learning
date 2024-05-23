# Breast Cancer Classification Project

## Overview

This project aims to classify breast cancer as malignant or benign using various machine learning algorithms. The dataset used contains various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

## Features

- **ID**: Identifier
- **Diagnosis**: 0 (Benign), 1 (Malignant)
- **Features**: 30 numeric features such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Algorithms Used

- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes

## Installation

To install the necessary packages, use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository
```bash
git clone https://github.com/yourusername Breast-Cancer-Prediction-Using-Machine-Learning.git
cd Breast-Cancer-Prediction-Using-Machine-Learning
```
2. Run the script
```bash
python main.py
```

## File Structure

```
.
├── data
│   ├── wdbc.data
├── main.py
├── lazy_predict.py
├── requirements.txt
└── README.md
``` 

## Output

The `main` script evaluates the performance of each model using cross-validation and outputs the classification report and confusion matrix for each algorithm.

Example output:
```
K-Nearest Neighbors (KNN)
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        71
           1       0.95      0.95      0.95        43

    accuracy                           0.96       114
   macro avg       0.96      0.96      0.96       114
weighted avg       0.96      0.96      0.96       114

Confusion Matrix:
[[69  2]
 [ 2 41]]
```

The `lazy_predict` script uses the `LazyPredict` library to evaluate the performance of multiple models at once.

Example output:
```
100%|██████████| 30/30 [00:00<00:00,  1.07it/s]
                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  Time Taken
Model
KNeighborsClassifier               0.96               0.96     0.96      0.96        0.02
SVC                                0.96               0.96     0.96      0.96        0.02
NuSVC                              0.96               0.96     0.96      0.96        0.02
DecisionTreeClassifier             0.95               0.95     0.95      0.95        0.01
ExtraTreeClassifier                0.95               0.95     0.95      0.95        0.01
RandomForestClassifier             0.95               0.95     0.95      0.95        0.12
ExtraTreesClassifier               0.95               0.95     0.95      0.95        0.09
AdaBoostClassifier                 0.95               0.95     0.95      0.95        0.08
GradientBoostingClassifier         0.95               0.95     0.95      0.95        0.07
LGBMClassifier                     0.95               0.95     0.95      0.95        0.04
XGBClassifier                      0.95               0.95     0.95      0.95        0.06
BaggingClassifier                  0.95               0.95     0.95      0.95        0.03
CalibratedClassifierCV             0.95               0.95     0.95      0.95        0.04
LinearSVC                          0.95               0.95     0.95      0.95        0.02
RidgeClassifierCV                  0.95               0.95     0.95      0.95        0.01
RidgeClassifier                    0.95               0.95     0.95      0.95        0.01
LogisticRegression                 0.95               0.95     0.95      0.95        0.02
SGDClassifier                      0.95               0.95     0.95      0.95        0.01
Perceptron                         0.95               0.95     0.95      0.95        0.01
PassiveAggressiveClassifier        0.95               0.95     0.95      0.95        0.01
BernoulliNB                        0.95               0.95     0.95      0.95        0.01
GaussianNB                         0.95               0.95     0.95      0.95        0.01
LinearDiscriminantAnalysis         0.95               0.95     0.95      0.95        0.01
QuadraticDiscriminantAnalysis      0.95               0.95     0.95      0.95        0.01
LabelPropagation                   0.95               0.95     0.95      0.95        0.02
LabelSpreading                     0.95               0.95     0.95      0.95        0.02
NearestCentroid                    0.95               0.95     0.95      0.95        0.01
DummyClassifier                    0.61               0.50     0.50      0.61        0.01
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.