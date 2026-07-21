# Decision Tree Experiments

## Run 1


Best Model: DT_depth4_split2_leaf20
Training F1: 0.796
Validation F1: 0.776
Depth: 4
Split: 2
Leaf: 20
Number of Nodes: 15


### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 29          | 2           |
| Actual 1     | 4           | 21          |

### Classification Report

          precision    recall  f1-score   support

       0       0.88      0.94      0.91        31
       1       0.91      0.84      0.88        25

accuracy                           0.89        56

ROC-AUC: 0.934


### Conclusion

Logistic regression is stronger than a decision tree model as it has a higher CV F1 score (0.852 vs. 0.776) and a higher ROC-AUC score (0.957 vs. 0.934). There could be some underfitting going on, but it is more highly probable that this model is just inherently worse. Also, since the CV dataset was selected randomly once, this could cause variance in the CV F1 score estimation; thus, k-fold cross-validation will be implemented.


---

# Run 2

Implemented CV K splitting


Best Model: DT_depth8_split10_leaf1
CV F1 Mean: 0.779
CV F1 SD: 0.043
Depth: 8
Split: 10
Leaf: 1
Number of Nodes: 53.4


### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 28          | 3           |
| Actual 1     | 6           | 19          |

### Classification Report

          precision    recall  f1-score   support

       0       0.82      0.90      0.86        31
       1       0.86      0.76      0.81        25

accuracy                           0.84        56

ROC-AUC: 0.839


### Conclusion

The model evidently has overfit the data, and this may be because the models with a leaf size of 1 were able to memorize the training set, enabling them to predict the validation fold reasonably well. Additionally, the ROC-AUC score has dropped significantly (from 0.934 to 0.839). This suggests the grid search should thus exclude leaf=1 as an option.


---

# Run 3

Restricted the grid search to leaves of size 10 or 20 (not 1 or 5)

Tested criterion="entropy" as an alternative to the default "gini"


Best Model: DT_depth5_split2_leaf10
CV F1 Mean: 0.763
CV F1 SD: 0.046
Depth: 5
Split: 2
Leaf: 10
Number of Nodes: 24.2


### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 27          | 4           |
| Actual 1     | 7           | 18          |

### Classification Report

          precision    recall  f1-score   support

       0       0.79      0.87      0.83        31
       1       0.82      0.72      0.77        25

accuracy                           0.80        56

ROC-AUC: 0.905


### Conclusion

A lower CV variance was found (i.e., more consistent across folds), but there was a worse test F1 score for the AD class specifically (0.77 vs 0.83) and slightly higher ROC-AUC (0.905 vs 0.890). Because recall for the AD class should be prioritized (0.72 vs 0.80), the gini model should be kept instead.


---

# Run 4

Added feature importance to see which features contribute most to the model’s predictions


Best Model: DT_depth5_split2_leaf10
CV F1 Mean: 0.763
CV F1 SD: 0.046
Depth: 5
Split: 2
Leaf: 10
Number of Nodes: 24.2


### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
|--------------|-------------|-------------|
| Actual 0     | 27          | 4           |
| Actual 1     | 7           | 18          |

### Classification Report

          precision    recall  f1-score   support

       0       0.79      0.87      0.83        31
       1       0.82      0.72      0.77        25

accuracy                           0.80        56

ROC-AUC: 0.905


### Feature Importances


MMSE 0.673
EDUC 0.094
M/F 0.089
nWBV 0.077
Age 0.067
SES 0.000


### Conclusion

As supported by the tree diagram, MMSE leads to the highest information reduction. This is consistent with the fact that MMSE is a clinical cognitive-assessment score closely tied to how the AD/CDR label itself is determined. On the other hand, SES showed no predictive value as a feature for this model.
