# Logistic Regression — Run Log

## Run 1

**Confusion Matrix:**
[[31  0]
[ 5 20]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.86      | 1.00   | 0.93     | 31      |
| 1            | 1.00      | 0.80   | 0.89     | 25      |
| **accuracy** |           |        | 0.91     | 56      |
| macro avg    | 0.93      | 0.90   | 0.91     | 56      |
| weighted avg | 0.92      | 0.91   | 0.91     | 56      |

**ROC-AUC:** 0.985

**Conclusion:** The model has high precision (1.00) but not very good recall (0.80). Should change the threshold value to reduce the number of false negative cases.

---

## Run 2
*Implemented different threshold values for each model to find the optimal threshold*

`linear_lambda_10 | Train=0.785 | CV=0.839 | Threshold=0.4`

**Best model:** linear_lambda_10
**Best CV accuracy:** 0.839
**Best threshold:** 0.4

**Confusion Matrix:**
[[29  2]
[ 4 21]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.88      | 0.94   | 0.91     | 31      |
| 1            | 0.91      | 0.84   | 0.88     | 25      |
| **accuracy** |           |        | 0.89     | 56      |
| macro avg    | 0.90      | 0.89   | 0.89     | 56      |
| weighted avg | 0.89      | 0.89   | 0.89     | 56      |

**ROC-AUC:** 0.965

**Conclusion:** Decreasing the threshold value from 0.5 to 0.4 improved Alzheimer's detection performance by increasing recall (0.80 → 0.84) while maintaining high precision (0.91). Also, the most optimal algorithm switched from a polynomial to a linear algorithm, highlighting a reduction in the risk of overfitting. However, accuracy is not a sufficient metric for this problem because it does not account for the difference between false positives and false negatives.

---

## Run 3
*Implementation of model comparison based on F1 score, not accuracy*
*Tested lambda values from 20 to 2, in intervals of 2, instead of 100–0.01*

`linear_lambda_20 | Train=0.812 | CV=0.852 | Threshold=0.4`

**Best model:** linear_lambda_20
**Best CV F1 score:** 0.852
**Best threshold:** 0.4

**Confusion Matrix:**
[[28  3]
[ 3 22]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.90      | 0.90   | 0.90     | 31      |
| 1            | 0.88      | 0.88   | 0.88     | 25      |
| **accuracy** |           |        | 0.89     | 56      |
| macro avg    | 0.89      | 0.89   | 0.89     | 56      |
| weighted avg | 0.89      | 0.89   | 0.89     | 56      |

**ROC-AUC:** 0.963

**Conclusion:** F1 score provided a more appropriate metric for comparing algorithms because it considers both precision and recall, which are important for Alzheimer's classification. However, the lambda value was not fully optimized, as the optimal value may exist outside the tested range. Additionally, L2 regularization was used, which reduces overfitting by penalizing large model coefficients. Alternative approaches, such as L1 regularization or class weighting, could be explored to improve feature selection or reduce false negative cases.

---

## Run 4
*Implemented lambda values of 100, 80, 60, 40, and 20*
*Implemented L1 regularization instead of L2 regularization (penalty="l1", solver="liblinear")*

`linear_lambda_20 | Train=0.779 | CV=0.778 | Threshold=0.4`

**Best model:** linear_lambda_20
**Best CV F1 score:** 0.778
**Best threshold:** 0.4

**Confusion Matrix:**
[[26  5]
[ 5 20]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.84      | 0.84   | 0.84     | 31      |
| 1            | 0.80      | 0.80   | 0.80     | 25      |
| **accuracy** |           |        | 0.82     | 56      |
| macro avg    | 0.82      | 0.82   | 0.82     | 56      |
| weighted avg | 0.82      | 0.82   | 0.82     | 56      |

**ROC-AUC:** 0.923

**Conclusion:** L1 regularization worsened precision, recall, and F1 score; although the value for lambda was changed, it could be further optimized.

---

## Run 5
*Changed back to L2 regularization*
*Changed lambda values*

`linear_lambda_43 | Train=0.813 | CV=0.852 | Threshold=0.4`

**Best model:** linear_lambda_43
**Best CV F1 score:** 0.852
**Best threshold:** 0.4

**Confusion Matrix:**
[[25  6]
[ 3 22]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.89      | 0.81   | 0.85     | 31      |
| 1            | 0.79      | 0.88   | 0.83     | 25      |
| **accuracy** |           |        | 0.84     | 56      |
| macro avg    | 0.84      | 0.84   | 0.84     | 56      |
| weighted avg | 0.85      | 0.84   | 0.84     | 56      |

**ROC-AUC:** 0.957

**Conclusion:** The optimized lambda value improved validation performance; however, the resulting test performance was worse compared with the threshold-optimized model from Run 2. This suggests that further optimization of hyperparameters may not necessarily improve generalization performance due to the limited dataset size.

---

## Run 6
*Implemented k-fold CV splitting*

**Best model:** linear_lambda_40
**Best CV F1 score:** 0.812
**Best threshold:** 0.4

**Confusion Matrix:**
[[21 10]
[ 1 24]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.95      | 0.68   | 0.79     | 31      |
| 1            | 0.71      | 0.96   | 0.81     | 25      |
| **accuracy** |           |        | 0.80     | 56      |
| macro avg    | 0.83      | 0.82   | 0.80     | 56      |
| weighted avg | 0.84      | 0.80   | 0.80     | 56      |

**ROC-AUC:** 0.926

**Conclusion:** k-fold CV surfaced a model with substantially improved AD recall (0.96) at the cost of increased false positives and lower No-AD recall (0.68). However, this is the most accurate assessment of the logistic regression model.

---

## Run 7
*Added feature importance to see which features contribute most to the model's predictions*

**Best model:** linear_lambda_40
**Best CV F1 score:** 0.812
**Best threshold:** 0.4

**Confusion Matrix:**
[[21 10]
[ 1 24]]

**Classification Report:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.95      | 0.68   | 0.79     | 31      |
| 1            | 0.71      | 0.96   | 0.81     | 25      |
| **accuracy** |           |        | 0.80     | 56      |
| macro avg    | 0.83      | 0.82   | 0.80     | 56      |
| weighted avg | 0.84      | 0.80   | 0.80     | 56      |

**ROC-AUC:** 0.926

**Feature Coefficients:**

| Feature | Coefficient |
|---------|--------------|
| MMSE    | −0.805       |
| nWBV    | −0.317       |
| M/F     | +0.296       |
| EDUC    | −0.201       |
| Age     | −0.157       |
| SES     | +0.050       |

**Conclusion:** Like the decision tree, the MMSE feature had the largest coefficient magnitude, and was thus the best predictive feature, while SES was the least useful predictive feature for this model. However, unlike the decision tree, nWBV was a more significant indicator against AD (−0.317 compared to 0.077, respectively). Further analysis revealed that MMSE and nWBV displayed only a moderate linear correlation (r = 0.34), suggesting that nWBV carried moderately independent predictive information relative to MMSE. Rather than feature redundancy, this indicates that the importance gap between MMSE and nWBV for the decision tree is driven by MMSE's dominant splits near the root of the tree, which sharply reduced the resulting nodes' impurities — leaving the tree unable to exploit nWBV's independent signal.
