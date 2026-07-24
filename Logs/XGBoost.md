# XGBoost Model Optimization Results

## Run 1 — Baseline XGBoost Optimization

### Hyperparameters

| Parameter | Value |
|---|---:|
| Best Model | XGB_Model_n200_depth5_learning_rate0.1_subsample0.8_colsample1.0 |
| CV F1 Mean | 0.855 |
| CV F1 SD | 0.031 |
| N Estimators | 200 |
| Max Depth | 5 |
| Learning Rate | 0.1 |
| Subsample | 0.8 |
| Colsample | 1.0 |
| Best Threshold | 0.3 |

### Confusion Matrix

| | Predicted No AD | Predicted AD |
|-|-:|-:|
| **Actual No AD** | 24 | 7 |
| **Actual AD** | 3 | 22 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-|-:|-:|-:|-:|
| No AD (0) | 0.89 | 0.77 | 0.83 | 31 |
| AD (1) | 0.76 | 0.88 | 0.81 | 25 |
| Accuracy | | | 0.82 | 56 |

**ROC-AUC:** 0.907  
**Test F1:** 0.815

### Feature Importances

| Feature | Importance |
|-|-:|
| MMSE | 0.456 |
| M/F | 0.188 |
| EDUC | 0.091 |
| SES | 0.090 |
| Age | 0.088 |
| nWBV | 0.086 |

### Conclusion

As expected, the most optimal XGBoost model maximized the number of trees (n=200). However, the model also selected the highest learning rate available in the search grid, suggesting that larger updates allowed the model to achieve better performance within the tested parameter range.

Additionally, the CV F1 mean score was the highest among the other evaluated models (logistic regression: 0.812, decision tree: 0.779, random forest: 0.832). The test F1 score (0.815) was also slightly higher than the other models (logistic regression: 0.814, decision tree: 0.766, random forest: 0.793).

Unlike the other tree-based models, nWBV had substantially lower feature importance in XGBoost, ranking as the least influential feature (0.086). This suggests that although nWBV contains predictive information, XGBoost was able to achieve similar predictive performance through alternative feature interactions, particularly MMSE and M/F.

---

# Run 2 — Expanded Hyperparameter Grid

### Changes

The hyperparameter grid was expanded:
- `n_estimators`: 200 → 400
- `learning_rate`: 0.1 → 0.2

This was done because Run 1 selected the upper boundaries of both parameters, suggesting the optimal values may have existed beyond the original search range.

### Hyperparameters

| Parameter | Value |
|-|-:|
| Best Model | XGB_Model_n300_depth4_learning_rate0.2_subsample0.7_colsample0.8 |
| CV F1 Mean | 0.860 |
| CV F1 SD | 0.037 |
| N Estimators | 300 |
| Max Depth | 4 |
| Learning Rate | 0.2 |
| Subsample | 0.7 |
| Colsample | 0.8 |
| Best Threshold | 0.3 |

### Confusion Matrix

| | Predicted No AD | Predicted AD |
|-|-:|-:|
| **Actual No AD** | 24 | 7 |
| **Actual AD** | 4 | 21 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-|-:|-:|-:|-:|
| No AD (0) | 0.86 | 0.77 | 0.81 | 31 |
| AD (1) | 0.75 | 0.84 | 0.79 | 25 |
| Accuracy | | | 0.80 | 56 |

**ROC-AUC:** 0.871  
**Test F1:** 0.792

### Feature Importances

| Feature | Importance |
|-|-:|
| MMSE | 0.425 |
| M/F | 0.162 |
| nWBV | 0.110 |
| EDUC | 0.109 |
| Age | 0.102 |
| SES | 0.092 |

### Conclusion

Expanding the `n_estimators` and `learning_rate` grid produced a model with a marginally higher CV F1 score (0.860 vs. 0.855), but significantly worse test performance (F1: 0.792 vs. 0.815; ROC-AUC: 0.871 vs. 0.907).

This suggests that the increased model complexity caused greater overfitting to the cross-validation folds. Although Run 2 achieved stronger validation performance, Run 1's more conservative hyperparameters generalized better to unseen test data.

These results motivated the implementation of additional regularization methods, including gamma and reg_lambda.

---

# Run 3 — Regularization and Early Stopping

### Changes

Two regularization techniques were added:

- **Gamma:** Limits unnecessary tree splits by requiring a minimum improvement before creating additional splits.
- **Reg Lambda:** Penalizes leaf weights to reduce model complexity.

Additionally, fixed `n_estimators` tuning was replaced with early stopping. Within each validation fold, training stopped when validation performance stopped improving, preventing unnecessary tree growth.

### Hyperparameters

| Parameter | Value |
|-|-:|
| Best Model | XGB_depth4_lr0.15_sub0.7_col0.8_gamma0.5_lambda1 |
| CV F1 Mean | 0.864 |
| CV F1 SD | 0.049 |
| Max Depth | 4 |
| Avg Fold Best Round | 65.8 |
| Learning Rate | 0.15 |
| Subsample | 0.7 |
| Colsample | 0.8 |
| Gamma | 0.5 |
| Reg Lambda | 1 |
| Best Threshold | 0.4 |

**Final model stopped at round:** 34

### Confusion Matrix

| | Predicted No AD | Predicted AD |
|-|-:|-:|
| **Actual No AD** | 25 | 6 |
| **Actual AD** | 3 | 22 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-|-:|-:|-:|-:|
| No AD (0) | 0.89 | 0.81 | 0.85 | 31 |
| AD (1) | 0.79 | 0.88 | 0.83 | 25 |
| Accuracy | | | 0.84 | 56 |

**ROC-AUC:** 0.945  
**Test F1:** 0.830

### Feature Importances

| Feature | Importance |
|-|-:|
| MMSE | 0.460 |
| M/F | 0.167 |
| nWBV | 0.096 |
| EDUC | 0.094 |
| Age | 0.092 |
| SES | 0.091 |

### Conclusion

Compared to the other evaluated models, XGBoost achieved the highest test F1 score:

| Model | Test F1 |
|-|-:|
| XGBoost | **0.830** |
| Logistic Regression | 0.814 |
| Random Forest | 0.793 |
| Decision Tree | 0.766 |

However, the difference between CV F1 mean and test F1 score (0.034) was larger than logistic regression (0.002), indicating that some performance variation occurred between validation and unseen test data.

Despite this, Run 3 achieved the best overall performance while using fewer boosting rounds. The combination of early stopping, gamma regularization, and lambda regularization likely controlled model complexity by limiting unnecessary tree growth and reducing the influence of individual trees.

Additionally, the final XGBoost model continued to identify MMSE as the strongest predictive feature while distributing importance more evenly among the remaining features.
