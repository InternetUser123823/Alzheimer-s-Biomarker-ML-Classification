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

# n_estimators is no longer grid-searched — early stopping determines 
# the optimal number of trees per fold automatically. MAX_ROUNDS is just 
# a generous upper ceiling (never expected to be reached), and 
# EARLY_STOP_ROUNDS controls how many rounds without improvement are 
# tolerated before training halts.
MAX_ROUNDS = 500
EARLY_STOP_ROUNDS = 20

# Hyperparameter grid 
depths = [3, 4, 5]
learning_rates = [0.1, 0.15, 0.2]
subsamples = [0.7, 0.8, 1.0]
colsamples = [0.8, 1.0]
gammas = [0, 0.5, 1]
reg_lambdas = [1, 5, 10]
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
            for gamma in gammas:
                for depth in depths:
                        for reg_lambda in reg_lambdas:
                            xgb_models.append({
                                "model_name": f"XGB_depth{depth}_lr{learning_rate}_sub{subsample}_col{colsample}_gamma{gamma}_lambda{reg_lambda}",
                                "params": (gamma, 
                                        depth, 
                                        learning_rate,
                                        subsample, 
                                        colsample,
                                        reg_lambda)
                            })

total_models = len(xgb_models)
current_model = 0

# CV K-fold validation
for model in xgb_models:
    fold_best_f1s = []
    fold_best_thresholds = []
    fold_best_rounds = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        gamma, d, lr, sub, col, reg_lambda = model["params"]

        fold_model = XGBClassifier(
            n_estimators=MAX_ROUNDS,
            max_depth=d,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            gamma=gamma,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            n_jobs=-1
        )

        fold_model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

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
        fold_best_rounds.append(fold_model.best_iteration)

    # Calculate and update XGBooster dictionary with these statistics
    mean_cv_f1 = sum(fold_best_f1s) / len(fold_best_f1s)
    mean_best_threshold = max(set(fold_best_thresholds), key=fold_best_thresholds.count)
    cv_f1_std = (sum((f1 - mean_cv_f1) ** 2 for f1 in fold_best_f1s) / len(fold_best_f1s)) ** 0.5
    mean_best_round = sum(fold_best_rounds) / len(fold_best_rounds)

    model["mean_best_round"] = mean_best_round
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
    gamma, d, lr, sub, col, reg_lambda = model["params"]
    print(f"[{current_model}/{total_models}]: depth={d}, learning_rate={lr}, subsample={sub}, colsample={col}, gamma={gamma}, reg_lambda={reg_lambda}") 

# Print out the best XGBooster statistics
print(
    f"\nBest Model: {best_model_dict['model_name']}\n"
    f"CV F1 Mean: {best_f1:.3f}\n"
    f"CV F1 SD: {best_f1_std:.3f}\n"
    f"Depth: {best_params[1]}\n"
    f"Avg Fold Best Round: {best_model_dict['mean_best_round']:.1f}\n"
    f"Learning Rate: {best_params[2]}\n"
    f"Subsample: {best_params[3]}\n"
    f"Colsample: {best_params[4]}\n"
    f"Gamma: {best_params[0]}\n"
    f"Reg Lambda: {best_params[5]}\n"
    f"Best Threshold: {best_threshold}\n"
)

# Refit best model on full training set
gamma, d, lr, sub, col, reg_lambda = best_params
best_model = XGBClassifier(
    n_estimators=MAX_ROUNDS,
    max_depth=d,
    learning_rate=lr,
    subsample=sub,
    colsample_bytree=col,
    gamma=gamma,
    reg_lambda=reg_lambda,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=EARLY_STOP_ROUNDS,
    n_jobs=-1
)

# Refit on full training set; carved out a small internal validation set for model training
X_fit, X_val_final, y_fit, y_val_final = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

best_model.fit(
    X_fit, y_fit,
    eval_set=[(X_val_final, y_val_final)],
    verbose=False
)

print(f"\nFinal model stopped at round: {best_model.best_iteration}")


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

