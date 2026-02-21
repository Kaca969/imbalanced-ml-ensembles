Imbalanced ML Ensembles

Predictive modeling on imbalanced clinical tabular data using ensemble learning and class-balancing strategies

Overview

This project examines how ensemble models perform on heavily imbalanced clinical tabular datasets. In real-world medical data, minority classes often represent the most critical outcomes, making standard accuracy insufficient as an evaluation metric. The goal of this project is to improve minority class detection using a combination of resampling techniques, ensemble learning methods, and appropriate evaluation metrics.

Methodology
Data Preparation

The workflow begins with exploratory data analysis and correlation visualization to better understand feature relationships. The dataset is split using stratified train/test splitting to preserve class distribution. Feature scaling is performed using StandardScaler. To address class imbalance, manual oversampling of the minority class is applied exclusively to the training set in order to prevent data leakage.

Models

Three ensemble-based classifiers were implemented and compared:

Random Forest — used as a strong baseline model with built-in feature importance evaluation.

AdaBoost — designed to iteratively focus on samples that are harder to classify.

Stacking Classifier — combines Random Forest and SVM as base learners, with Logistic Regression as the meta-learner.

All models were tuned using GridSearchCV to optimize hyperparameters.

Evaluation

Because of class imbalance, accuracy alone does not provide meaningful insight. Model performance is evaluated primarily using:

F1-score

ROC-AUC

Additional metrics include precision, recall, confusion matrices, and ROC curves to provide a more comprehensive performance analysis.

Results

The Stacking classifier achieved the best overall performance:

Accuracy ≈ 0.83

F1-score ≈ 0.72

ROC-AUC ≈ 0.89–0.90

Feature selection using SelectFromModel (based on Random Forest feature importances) resulted in only a slight decrease in F1-score to approximately 0.70. This indicates that the model does not rely excessively on any single feature, suggesting good generalization capability.

Dataset

The project uses the Heart Failure Clinical Records Dataset, expected at:
data/heart_failure_clinical_records_dataset.csv

The dataset is not included in the repository.

Setup
pip install -r requirements.txt
python train_imbalanced_ensembles.py

Key Takeaways

This project demonstrates a complete machine learning pipeline tailored for imbalanced clinical data. It highlights the importance of appropriate metrics, careful handling of class distribution, and the performance benefits of ensemble and stacking methods. The workflow serves as a practical reference for real-world classification problems where detecting the minority class is essential.
