Imbalanced ML Ensembles
Predictive modeling on imbalanced clinical tabular data using ensemble learning and class-balancing strategies.
Overview
This project looks at how well ensemble models handle classification when the data is heavily imbalanced — which is pretty common in clinical datasets. The core goal is to push minority class detection beyond what simple accuracy can tell you, using a combination of resampling, ensemble methods, and proper evaluation.
Methodology
Data Preparation
The pipeline starts with some exploratory analysis and correlation visualization to get a feel for the data. From there: stratified train/test splitting to preserve class ratios, standardization with StandardScaler, and manual oversampling of the minority class on the training set only (to avoid any data leakage).
Models
Three classifiers were implemented and compared:

Random Forest — solid baseline with built-in feature importance
AdaBoost — focuses iteratively on harder-to-classify samples
Stacking Classifier — Random Forest and SVM as base learners, with Logistic Regression as the meta-learner

All models were tuned using GridSearchCV.
Evaluation
Given the class imbalance, accuracy alone doesn't mean much. The evaluation leans on F1-score and ROC-AUC as the primary metrics, alongside precision, recall, confusion matrices, and ROC curves for a fuller picture.
Results
The Stacking classifier came out on top:

Accuracy ≈ 0.83
F1-score ≈ 0.72
ROC-AUC ≈ 0.89–0.90

Feature selection (via SelectFromModel based on Random Forest importances) brought the F1 down only slightly to around 0.70, which suggests the model isn't overly dependent on any single feature — a good sign for generalization.
Dataset
Heart Failure Clinical Records Dataset — expected at data/heart_failure_clinical_records_dataset.csv. Not included in the repo.
Setup
bashpip install -r requirements.txt
python train_imbalanced_ensembles.py
Key Takeaways
The project works through a fairly complete ML workflow — from handling messy class distributions to picking the right metrics and squeezing performance out of stacked ensembles. It's a good reference for anyone dealing with real-world clinical data where the minority class is exactly the one you can't afford to miss.
