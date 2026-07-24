# Final Model Performance Comparison

Final comparison across all four models, evaluated on the same held-out 
test set (56 samples, `random_state=42`) using stratified 5-fold 
cross-validation for model selection.

| Model | Run | Test F1 | CV F1 Mean | CV F1 SD | Accuracy | Test ROC-AUC | Threshold | Precision (AD) | Recall (AD) | F1 (AD) | False Positives | False Negatives | Best Parameters |
|-------|-----|:-------:|:----------:|:--------:|:--------:|:-------:|:---------:|:--------------:|:-----------:|:-------:|:---------------:|:---------------:|-----------------|
| Logistic Regression | Run 7 | **0.814** | 0.812 | 0.051 | 0.80 | 0.926 | 0.4 | 0.71 | **0.96** | 0.81 | 10 | **1** | L2, λ = 40 |
| Decision Tree | Run 3 | 0.766 | 0.763 | **0.046** | 0.80 | 0.905 | 0.5 (default) | **0.82** | 0.72 | 0.77 | **4** | 7 | Depth = 5, Split = 2, Leaf = 10, Mean Nodes = 24.2 |
| Random Forest | Run 3 | 0.793 | 0.832 | **0.039** | 0.79 | 0.935 | 0.4 | 0.70 | 0.92 | 0.79 | 10 | 2 | n = 100, Depth = 5, Split = 10, Leaf = 5, Max Features = 1, Gini |
| XGBoost | Run 3 | **0.830** | **0.864** | 0.049 | **0.84** | **0.945** | 0.4 | 0.79 | 0.88 | **0.83** | 6 | 3 | Early stopping (34 rounds), Depth = 4, LR = 0.15, Subsample = 0.7, Colsample = 0.8, Gamma = 0.5, Reg Lambda = 1 |
