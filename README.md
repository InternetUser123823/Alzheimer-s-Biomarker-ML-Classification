# Overview
The goal of this project was to develop a classification model that is capable of predicting Alzheimer's disease status based on clinical and MRI-derived features by training and comparing four models: logistic regression, decision trees, random forests, and XGBoost. Each model went through multiple iterations, as documented in the run logs attached. [[should i mention results here next?]]

# Dataset
The dataset used in this project was obtained from Kaggle:
"Alzheimer Features" by Baris Dincer.

This publicly available dataset contains clinical and MRI-derived features used for Alzheimer's disease classification, including age, education level, MMSE, CDR, estimated total intracranial volume (eTIV), normalized whole brain volume (nWBV), and atlas scaling factor (ASF).

Note that this dataset only contains about 374 records. Because some features were missing, this was addressed via mean imputation.

The dataset appears to be derived from the Open Access Series of Imaging Studies (OASIS) dataset, although the Kaggle page does not provide an official citation. The Kaggle page can be accessed here:
https://www.kaggle.com/datasets/brsdincer/alzheimer-features/data

# Methods
Logistic Regression — a linear model estimating the probability of AD based on a weighted combination of features. Used as a baseline; also tested with polynomial feature expansion.

Decision Tree — a single tree that splits data on feature thresholds to separate AD from non-AD cases. Offers more interpretability but higher variance than the other two models.

Random Forest — an ensemble of many decision trees, each trained on a random subset of the data and features, with predictions averaged together. Reduces the variance/instability of a single decision tree.

XGBoost - an ensemble of many decision trees, with each subsequent tree trained on the residual errors of the previous tree, rather than independently, allowing sequential correction of earlier mistakes.

All four models were evaluated using stratified 5-fold cross-validation and compared on F1 score, ROC-AUC, and confusion matrix breakdown, with class-weighted training and threshold tuning applied to address the importance of correctly identifying AD cases.

# Results
- Accuracy:
- Precision:
- Recall:
- F1 Score:
- Any interesting findings

# Files
Logistic regression model: logistic_regression.py    
Decision tree model: decision_tree.py    
Random forest model modified with threshold search: rf_test_threshold.py    
Random forest model: random_forest.py   
XGBoost model: xg_boost.py          
XGBoost model modified with evaluation set early stopping: xg_boost_eval.py    
Project description: README.md    
Dependencies: requirements.txt    

# How to Run
Commands shown for VSCode on macOS; use `python` instead of `python3` if that's how Python 3 is set up on your system.           
pip install -r requirements.txt          
python3 logistic_regression.py           
python3 decision_tree.py            
python3 random_forest.py          
python3 rf_test_threshold.py           

# Future Improvements
- Hyperparameter tuning
- Cross-validation
- Feature engineering

## Note
This project was developed as part of my personal learning so I could get more familiar with machine learning models, along with some common ML workflows.
