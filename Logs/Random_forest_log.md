# Random Forest — Run Log

## Run 1

**Best Model:** RF_n100_depth4_split20_leaf5

**CV F1 Mean:** 0.804  
**CV F1 SD:** 0.025  
**Avg Nodes/Tree:** 18.7  
**N Estimators:** 100  
**Depth:** 4  
**Split:** 20  
**Leaf:** 5  
**Max Features:** 2  

**Confusion Matrix:**

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 27          | 4           |
| Actual 1     | 4           | 21          |

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.87      | 0.87   | 0.87     | 31      |
| 1     | 0.84      | 0.84   | 0.84     | 25      |
| Accuracy | | | 0.86 | 56 |

**ROC-AUC:** 0.928

**Feature Importances:**

| Feature | Importance |
|---------|------------|
| MMSE    | 0.572 |
| nWBV    | 0.162 |
| M/F     | 0.081 |
| SES     | 0.070 |
| Age     | 0.062 |
| EDUC    | 0.053 |

**Conclusion:** The random forest CV F1 SD score is notably lower than the best tree model so far (0.025 compared to 0.046), highlighting the impact of the ensemble advantage in the random forest. Additionally, the CV F1 and ROC-AUC scores are comparable to those of the best logistic regression model (0.804 vs 0.812 for CV F1, 0.928 vs 0.926 for ROC-AUC). However, when analyzing their confusion matrices, it is evident that the random forest model is much more balanced between false negatives and false positives (4 and 4 cases, respectively), compared to the best logistic regression model (1 and 10 cases, respectively). Thus, investigations to minimize the incidence of false negatives in the random forest could be further explored.


---

## Run 2

**Changes:**
- Narrowed grid search for n_estimators_list, depths, splits, leaves, max_features_list (to reduce CPU load)
- Created threshold search to find optimal threshold for the best random forest based on its folds

**Best Model:** RF_n100_depth4_split20_leaf5_mf2

**CV F1 Mean:** 0.822  
**CV F1 SD:** 0.032  
**Avg Nodes/Tree:** 18.7  
**N Estimators:** 100  
**Depth:** 4  
**Split:** 20  
**Leaf:** 5  
**Max Features:** 2  
**Best Threshold:** 0.5  

**Confusion Matrix:**

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 27          | 4           |
| Actual 1     | 4           | 21          |

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.87      | 0.87   | 0.87     | 31      |
| 1     | 0.84      | 0.84   | 0.84     | 25      |
| Accuracy | | | 0.86 | 56 |

**ROC-AUC:** 0.928

**Feature Importances:**

| Feature | Importance |
|---------|------------|
| MMSE    | 0.572 |
| nWBV    | 0.162 |
| M/F     | 0.081 |
| SES     | 0.070 |
| Age     | 0.062 |
| EDUC    | 0.053 |

**Conclusion:** Combining threshold search with a narrowed grid search retained the same parameters as in Run 1. The optimal threshold was found to be 0.5, the random forest’s default, indicating that its probabilities were already well calibrated, unlike logistic regression, which benefited from a lower optimal threshold (0.4). The CV F1 mean increased slightly (from 0.804 to 0.822) due to per-fold threshold optimization during cross-validation, though the final test-set results are unchanged from Run 1 since the selected threshold matches the original default.


---

## Run 3

**Changes:**
- Tested gini vs. entropy criterion to potentially optimize feature selection
- Widened grid search for max_features_list from 2 and 4 to a range from 1-6
- Tested implementation of ccpalpha to further reduce overfitting by reducing tree complexity

**Best Model:** RF_n100_depth5_split10_leaf5_mf1

**CV F1 Mean:** 0.832  
**CV F1 SD:** 0.039  
**Avg Nodes/Tree:** 30.2  
**N Estimators:** 100  
**Depth:** 5  
**Split:** 10  
**Leaf:** 5  
**Max Features:** 1  
**Best Threshold:** 0.4  
**Best Criterion:** gini  
**Best ccp_alpha:** 0  

**Confusion Matrix:**

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 21          | 10          |
| Actual 1     | 2           | 23          |

**Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.91      | 0.68   | 0.78     | 31      |
| 1     | 0.70      | 0.92   | 0.79     | 25      |
| Accuracy | | | 0.79 | 56 |

**ROC-AUC:** 0.935

**Feature Importances:**

| Feature | Importance |
|---------|------------|
| MMSE    | 0.480 |
| nWBV    | 0.162 |
| EDUC    | 0.095 |
| Age     | 0.092 |
| SES     | 0.089 |
| M/F     | 0.081 |

**Conclusion:** First, the ROC-AUC score was slightly higher for the R3 version of the forest than the R2 version (0.935 compared to 0.928, respectively). Although the recall for the AD class (class 1) for this version was higher than that of the recall for the AD class for the previous R2 version (0.92 vs. 0.84, respectively), suggesting that this version of the random forest was less prone to identifying false negatives, the recall for the non-AD class (class 0) for this version was lower than that of the recall for the non-AD class for the previous R2 version (0.68 vs. 0.87, respectively), suggesting that this version of the random forest was more prone to identifying false positives. Also, it is important to note the slight change in threshold, meaning that the slight decrease in threshold means that a lower predicted probability was required for the model to classify a patient as AD-positive. The criterion did not appear to have a large effect, as the best model still utilized gini; similarly, the optimal ccp_alpha value was 0, suggesting that the complexity of the trees had already been sufficiently reduced thanks to the grid search implemented. Finally, it is important to note the reduction of maximum features from 2 to 1 in this version, supporting the creation of less correlated trees and thus increased randomization to reduce overfitting.
