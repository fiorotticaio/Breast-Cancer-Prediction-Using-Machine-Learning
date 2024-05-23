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
├── requirements.txt
└── README.md
``` 

## Output

The script evaluates the performance of each model using cross-validation and outputs the classification report and confusion matrix for each algorithm.

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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.