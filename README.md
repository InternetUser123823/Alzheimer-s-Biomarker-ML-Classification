# Overview
The goal of this project was to develop a classification model that is capable of predicting Alzheimer's disease status based on clinical and MRI-derived features by training and comparing four models: logistic regression, decision trees, random forests, and XGBoost. Each model went through multiple iterations, as documented in the run logs attached. In conclusion, the XGBoost model had the highest predictive performance overall; nevertheless, the Logistic Regression model provided a reasonable alternative that prioritized recall over precision, which may be more applicable depending on the context.

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
| Feature | Logistic Regression (Coefficient) | Decision Tree | Random Forest | XGBoost |
|---------|----------------------------------:|--------------:|--------------:|---------:|
| **MMSE** | **−0.805** | **0.673** | **0.480** | **0.460** |
| **nWBV** | −0.317 | 0.077 | 0.162 | 0.096 |
| **M/F** | +0.296 | 0.089 | 0.081 | 0.167 |
| **EDUC** | −0.201 | 0.094 | 0.095 | 0.094 |
| **Age** | −0.157 | 0.067 | 0.092 | 0.092 |
| **SES** | +0.050 | 0.000 | 0.089 | 0.091 |        


| Model               | Test F1 | Accuracy | ROC-AUC | Recall (AD) |
|----------------------|---------|----------|---------|-------------|
| Logistic Regression  | 0.814   | 0.80     | 0.926   | 0.96        |
| Decision Tree        | 0.766   | 0.80     | 0.905   | 0.72        |
| Random Forest        | 0.793   | 0.79     | 0.935   | 0.92        |
| XGBoost              | 0.83    | 0.84     | 0.945   | 0.88        |

Full results — including cross-validation stability, confusion matrix breakdowns, and best hyperparameters per run — are available in Model_Performance_Comparison.md.

# Conclusion
The XGBoost model demonstrated a higher test F1 score (0.83) and the highest CV F1 mean (0.864) compared to the other models, indicating that it had the strongest overall predictive performance. This could potentially be due to XGBoost's boosting mechanism, where each subsequent tree is trained to correct the residual errors of the previous tree, rather than being trained independently (like in Random Forest). This, paired with a high learning rate, could also explain why the XGBoost model required far fewer trees to achieve a strong performance (n=34) compared to the Random Forest model (n=100). However, the CV-to-test drop-off for the XGBoost and Random Forest models, a decrease of ~0.034 and ~0.039, respectively, was notably larger than the drop-off for the Logistic Regression and Decision Tree models, a decrease of ~0.002 and ~0.003, respectively, indicating the XGBoost model's advantage in a higher predictive performance ceiling rather than in generalizing more consistently from the CV to the test data.

Similarly, the XGBoost model showed a higher ROC-AUC score (0.945) compared to the other models, indicating that its predictive advantage holds across a threshold-independent metric, not just with the F1 score. In contrast, the Logistic Regression model achieved a higher recall for the AD class (0.96), despite having a lower CV and test F1 score and ROC-AUC than the XGBoost model. This reflects a precision-recall trade-off: the model identifies fewer false negative cases (1) at the cost of identifying more false positives (10). In a clinical screening context, where failing to identify an AD case (a false negative) is often considered more costly than a false alarm, this trade-off may make the Logistic Regression model a more reasonable choice compared to the XGBoost model despite its lower predictive performance overall.

Finally, when considering feature importance, MMSE appears as the most predictive feature across all four models. This result is consistent with expectations, as the MMSE (Mini-Mental State Examination) is a well-established clinical cognitive screening tool used to diagnose dementia severity in real practice. Additionally, it is important to note that though the nWBV feature appears as the second most influential feature in both the Logistic Regression and Random Forest models, it is notably weaker in both the Decision Tree and XGBoost models. Regarding the decision tree model, this result could be attributed to a shallow depth combined with an already high information gain from the initial MMSE split, leaving the tree increasingly unable to utilize the nWBV signal in its remaining nodes. For the XGBoost model, this result could be explained by the gamma threshold at each node, which could limit the ability for a secondary feature like nWBV to reduce a sufficient amount of error. Additionally, after the first MMSE split that captures most of the signal, the nWBV feature may not be useful in correcting the residual errors of the previous trees. 

In conclusion, while the XGBoost model had the highest predictive performance overall, the Logistic Regression model provides a reasonable alternative that prioritizes recall over precision, which may be more applicable depending on the context.

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
python3 xg_boost.py          
python3 xg_boost_eval.py

# Limitations/Future Improvements
A limitation to note is the modest dataset size (~374 patients); therefore, the conclusions drawn in this project should be treated as suggestive rather than definitive, and would benefit from validation on a larger, independent dataset. Additionally, mean imputation of the missing data artificially shrinks the variance of the feature, as each imputed value is identical to the mean. Instead, the models would have benefited from imputation alternatives, such as KNN imputation or iterative/regression-based imputation. Finally, more extensive feature engineering could have been implemented; because the dataset used in this project was relatively small, interaction and derived terms were not included to reduce the possibility of overfitting, but with more samples, these terms could be tested to see if they optimize predictive performance.

## Note
This project was developed as part of my personal learning so I could get more familiar with machine learning models, along with some common ML workflows.
