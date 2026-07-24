import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier   
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score  
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("alzheimer1.csv")


# Create binary target
# CDR = 0 --> no dementia
# CDR > 0 --> dementia
df["AD"] = (df["CDR"] > 0).astype(int)


# Select features
features = [
    "M/F",
    "Age",
    "EDUC",
    "SES",
    "MMSE",
    "nWBV"
]

# Label encode M/F into binary values
df["M/F"] = df["M/F"].map({
    "F": 0,
    "M": 1
})


# Define X and Y series
X = df[features]
y = df["AD"]


# Split between training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

y_train = y_train.to_numpy()   
y_test = y_test.to_numpy()     


# Fill in missing data
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# Build XGBoosters
# Stratified K Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grid 
n_estimators_list = [200, 300, 400]
depths = [2, 3, 4, 5]
learning_rates = [0.1, 0.15, 0.2]
subsamples = [0.7, 0.8, 1.0]
colsamples = [0.5, 0.8, 1.0]
thresholds = [0.3,0.4,0.5,0.6,0.7]                   

xgb_models = []
best_params = None
best_model_dict = None
best_threshold = 0
best_f1 = 0.0
best_f1_std = 0.0

# Balance the positive and negative cases
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

# Create XGB model dictionaries 
for learning_rate in learning_rates:
    for subsample in subsamples:
        for colsample in colsamples:
            for n_estimators in n_estimators_list:
                for depth in depths:
                    xgb_models.append({
                        "model_name": f"XGB_Model_n{n_estimators}_depth{depth}_learning_rate{learning_rate}_subsample{subsample}_colsample{colsample}",
                        "params": (n_estimators, 
                                depth, 
                                learning_rate,
                                subsample, 
                                colsample)
                    })

total_models = len(xgb_models)
current_model = 0

# CV K-fold validation
for model in xgb_models:
    fold_best_f1s = []
    fold_best_thresholds = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        fold_model = XGBClassifier(
            n_estimators=model["params"][0],
            max_depth=model["params"][1],
            learning_rate=model["params"][2],
            subsample=model["params"][3],
            colsample_bytree=model["params"][4],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
        fold_model.fit(X_fold_train, y_fold_train)

        val_prob = fold_model.predict_proba(X_fold_val)[:, 1]

        # Calculate best threshold for this fold
        best_fold_f1 = 0
        best_fold_threshold = 0.5
        for threshold in thresholds:
            val_pred = (val_prob >= threshold).astype(int)
            f1 = f1_score(y_fold_val, val_pred)
            if f1 > best_fold_f1:
                best_fold_f1 = f1
                best_fold_threshold = threshold

        fold_best_f1s.append(best_fold_f1)
        fold_best_thresholds.append(best_fold_threshold)

    # Calculate and update XGBooster dictionary with these statistics
    mean_cv_f1 = sum(fold_best_f1s) / len(fold_best_f1s)
    mean_best_threshold = max(set(fold_best_thresholds), key=fold_best_thresholds.count)
    cv_f1_std = (sum((f1 - mean_cv_f1) ** 2 for f1 in fold_best_f1s) / len(fold_best_f1s)) ** 0.5

    model["mean_cv_f1"] = mean_cv_f1
    model["mean_best_threshold"] = mean_best_threshold
    model["cv_f1_std"] = cv_f1_std

    # Save best XGBooster info
    if mean_cv_f1 > best_f1:
        best_model_dict = model
        best_params = model["params"]
        best_f1 = mean_cv_f1
        best_f1_std = cv_f1_std
        best_threshold = mean_best_threshold

    # Printing out progress
    current_model += 1
    n_est, d, lr, sub, col = model["params"]
    print(f"[{current_model}/{total_models}]: n_estimators={n_est}, depth={d}, learning rate={lr}, subsamples={sub}, colsamples={col}")

# Print out the best XGBooster statistics
print(
    f"\nBest Model: {best_model_dict['model_name']}\n"
    f"CV F1 Mean: {best_f1:.3f}\n"
    f"CV F1 SD: {best_f1_std:.3f}\n"
    f"N Estimators: {best_params[0]}\n"
    f"Max Depth: {best_params[1]}\n"
    f"Learning Rate: {best_params[2]}\n"
    f"Subsample: {best_params[3]}\n"
    f"Colsample: {best_params[4]}\n"
    f"Best Threshold: {best_threshold}\n"
)

# Refit best model on full training set
best_model = XGBClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    learning_rate=best_params[2],
    subsample=best_params[3],
    colsample_bytree=best_params[4],
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# Final evaluation on test set
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.3f}")
test_f1 = f1_score(y_test, test_pred)
print(f"Test F1: {test_f1:.3f}")

# Feature importances (averaged across all trees in the model)
print("\nFeature Importances:")
for name, importance in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<10} {importance:.3f}")

