import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "XGBoost"]

# ---- Chart 1: CV F1 & ROC-AUC ----
test_f1 = [0.814, 0.766, 0.793, 0.830]
cv_f1 = [0.812, 0.763, 0.832, 0.864]
cv_f1_std = [0.051, 0.046, 0.039, 0.049]   
roc_auc = [0.926, 0.905, 0.935, 0.945]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(models, cv_f1, yerr=cv_f1_std, capsize=5, color="steelblue")
axes[0].set_title("CV F1 Score")
axes[0].set_ylim(0, 1)
axes[1].bar(models, test_f1, color="indianred")
axes[1].set_title("Test F1 Score")
axes[1].set_ylim(0, 1)
axes[2].bar(models, roc_auc, color="indianred")
axes[2].set_title("Test ROC-AUC")
axes[2].set_ylim(0, 1)
plt.tight_layout()
plt.savefig("chart1_f1_auc.png", dpi=150)
plt.close()

# ---- Chart 2: Recall breakdown ----
ad_recall = [0.96, 0.72, 0.92, 0.88]
noad_recall = [0.68, 0.87, 0.68, 0.81]

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, ad_recall, width, label="AD Recall", color="darkorange")
ax.bar(x + width/2, noad_recall, width, label="No-AD Recall", color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Recall by Class")
plt.tight_layout()
plt.savefig("chart2_recall.png", dpi=150)
plt.close()

# ---- Chart 3: Feature importance / coefficient magnitude ----
features = ["MMSE", "nWBV", "M/F", "EDUC", "Age", "SES"]
dt_importance = [0.673, 0.077, 0.089, 0.094, 0.067, 0.000]
rf_importance = [0.480, 0.162, 0.081, 0.095, 0.092, 0.089]
lr_coef_abs = [0.805, 0.317, 0.296, 0.201, 0.157, 0.050]
xg_importance = [0.460, 0.096, 0.167, 0.094, 0.092, 0.091]

y = np.arange(len(features))
height = 0.2   # narrower, since we now have 4 bars per feature instead of 3

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(y - 1.5*height, dt_importance, height, label="Decision Tree")
ax.barh(y - 0.5*height, rf_importance, height, label="Random Forest")
ax.barh(y + 0.5*height, lr_coef_abs, height, label="LR |Coefficient|")
ax.barh(y + 1.5*height, xg_importance, height, label="XGBoost")
ax.set_yticks(y)
ax.set_yticklabels(features)
ax.legend()
ax.set_title("Feature Importance / Coefficient Magnitude")
plt.tight_layout()
plt.savefig("chart3_features.png", dpi=150)
plt.close()

print("Saved chart1_f1_auc.png, chart2_recall.png, chart3_features.png")