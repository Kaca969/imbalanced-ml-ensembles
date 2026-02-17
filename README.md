# Imbalanced ML Ensembles

Predictive modeling on imbalanced clinical tabular data using ensemble learning and class-balancing strategies.

---

## Overview

This project investigates classification performance under class imbalance using ensemble models and robust evaluation metrics.

The primary objective is to improve minority class detection beyond simple accuracy by applying:

- Resampling techniques
- Ensemble learning methods
- Hyperparameter optimization
- Feature importance analysis

---

## Methodology

### Data Preparation

- Exploratory Data Analysis (EDA)
- Correlation matrix visualization
- Stratified train/test split
- Standardization using StandardScaler
- Manual oversampling of the minority class (training set only)

### Models Implemented

- Random Forest
- AdaBoost
- Stacking Classifier
  - Base learners: Random Forest + SVM
  - Meta-learner: Logistic Regression

Hyperparameters were optimized using GridSearchCV.

---

## Evaluation Metrics

Model performance was assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curves

Special emphasis was placed on F1-score and ROC-AUC due to class imbalance.

---

## Results (Test Set)

Best performance achieved by the Stacking classifier:

- Accuracy ≈ 0.83
- F1-score ≈ 0.72
- ROC-AUC ≈ 0.89–0.90

Feature selection preserved competitive performance (F1 ≈ 0.70), demonstrating model robustness under dimensionality reduction.

---

## Feature Analysis

- Feature importance computed via Random Forest
- Top features visualized
- Feature selection performed using SelectFromModel
- Comparative evaluation between full feature set and reduced feature set

---

## Dataset

Heart Failure Clinical Records Dataset

Expected location:

data/heart_failure_clinical_records_dataset.csv

The dataset is not included in this repository.

---

## Installation

pip install -r requirements.txt

---

## Run

python train_imbalanced_ensembles.py

---

## Key Focus Areas

- Imbalanced classification
- Ensemble learning
- Robust evaluation strategies
- Practical machine learning workflow
