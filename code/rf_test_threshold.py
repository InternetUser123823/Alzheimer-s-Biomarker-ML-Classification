import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier   
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

# One-hot encode M/F into binary values
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


# Build random forest
# Stratified K Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter grid (grid reduced to reduce CPU load)
n_estimators_list = [100]             
depths = [3, 4, 5]                   
splits = [10, 20]                       
leaves = [3, 5, 10]                    
max_features_list = [1, 2, 3, 4, 5, 6]              
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]   
criterion_list = ["gini", "entropy"]
ccp_alphas = [0, 0.001, 0.005, 0.01, 0.02]

forests = []
best_params = None
best_forest_dict = None
best_threshold = 0
best_f1 = 0.0
best_f1_std = 0.0

# Create forest dictionaries 
for ccp_alpha in ccp_alphas:
    for criterion in criterion_list:
        for max_feat in max_features_list:
            for n_estimators in n_estimators_list:
                for depth in depths:
                    for split in splits:
                        for leaf in leaves:
                            forests.append({
                                "model_name": f"RF_n{n_estimators}_depth{depth}_split{split}_leaf{leaf}_mf{max_feat}",
                                "params": (n_estimators, 
                                        depth, 
                                        split, 
                                        leaf, 
                                        max_feat, 
                                        criterion, 
                                        ccp_alpha)
                            })

total_models = len(forests)
current_model = 0

# CV K-fold validation
for forest in forests:
    fold_best_f1s = []
    fold_best_thresholds = []
    fold_avg_nodes = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        fold_model = RandomForestClassifier(
            n_estimators=forest["params"][0],
            max_depth=forest["params"][1],
            min_samples_split=forest["params"][2],
            min_samples_leaf=forest["params"][3],
            max_features=forest["params"][4],
            criterion=forest["params"][5],
            ccp_alpha=forest["params"][6],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        fold_model.fit(X_fold_train, y_fold_train)

        # Calculate average amount of nodes for this fold
        avg_nodes_this_fold = sum(
            t.tree_.node_count for t in fold_model.estimators_
        ) / len(fold_model.estimators_)
        fold_avg_nodes.append(avg_nodes_this_fold)

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

    # Calculate and update forest dictionary with these statistics
    mean_cv_f1 = sum(fold_best_f1s) / len(fold_best_f1s)
    mean_best_threshold = max(set(fold_best_thresholds), key=fold_best_thresholds.count)
    cv_f1_std = (sum((f1 - mean_cv_f1) ** 2 for f1 in fold_best_f1s) / len(fold_best_f1s)) ** 0.5
    avg_nodes = sum(fold_avg_nodes) / len(fold_avg_nodes)

    forest["mean_cv_f1"] = mean_cv_f1
    forest["mean_best_threshold"] = mean_best_threshold
    forest["cv_f1_std"] = cv_f1_std
    forest["avg_nodes"] = avg_nodes

    # Save best forest info
    if mean_cv_f1 > best_f1:
        best_forest_dict = forest
        best_params = forest["params"]
        best_f1 = mean_cv_f1
        best_f1_std = cv_f1_std
        best_threshold = mean_best_threshold

    # Printing out progress
    current_model += 1
    n_est, d, s, l, mf, crit, ccpa = forest["params"]
    print(f"[{current_model}/{total_models}] max_features={mf}, n_estimators={n_est}, depth={d}, split={s}, leaf={l}")

# Print out the best forest statistics
print(
    f"\nBest Model: {best_forest_dict['model_name']}\n"
    f"CV F1 Mean: {best_f1:.3f}\n"
    f"CV F1 SD: {best_f1_std:.3f}\n"
    f"Avg Nodes/Tree: {best_forest_dict['avg_nodes']:.1f}\n"
    f"N Estimators: {best_params[0]}\n"
    f"Depth: {best_params[1]}\n"
    f"Split: {best_params[2]}\n"
    f"Leaf: {best_params[3]}\n"
    f"Max Features: {best_params[4]}\n"
    f"Best Threshold: {best_threshold}\n"
    f"Best criterion: {best_params[5]}\n"
    f"Best ccpalpha: {best_params[6]}"
)

# Refit best model on full training set
best_model = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    min_samples_split=best_params[2],
    min_samples_leaf=best_params[3],
    max_features=best_params[4],
    criterion=best_params[5],
    ccp_alpha=best_params[6],
    class_weight="balanced",
    random_state=42,
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

# Feature importances (averaged across all trees in the forest)
print("\nFeature Importances:")
for name, importance in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<10} {importance:.3f}")

