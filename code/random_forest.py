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

# Hyperparameter grid
n_estimators_list = [50, 100, 200]
depths = [2, 3, 4, 5, 6, 8, 10]
splits = [2, 5, 10, 20]
leaves = [3, 5, 10, 20]   # leaf=1 excluded, same reasoning as the decision tree
max_features = [2, 3, 4, 6]

forests = []
best_score = 0
best_forests_index = None
best_params = None

# Printing out progress
total_models = (
    len(max_features)
    * len(n_estimators_list)
    * len(depths)
    * len(splits)
    * len(leaves)
)
current_model = 0

for max_feat in max_features:
    for n_estimators in n_estimators_list:
        for depth in depths:
            for split in splits:
                for leaf in leaves:
                    forest = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        max_features=max_feat,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1   
                    )

                    current_model += 1
                    print(
                        f"[{current_model}/{total_models}] "
                        f"max_features={max_feat}, "
                        f"n_estimators={n_estimators}, "
                        f"depth={depth}, "
                        f"split={split}, "
                        f"leaf={leaf}"
                    )

                    cv_results = cross_validate(
                        forest, X_train, y_train,
                        cv=skf, scoring="f1",
                        return_estimator=True
                    )
                    cv_scores = cv_results["test_score"]
                    score = cv_scores.mean()

                    fold_avg_nodes = [
                        sum(t.tree_.node_count for t in est.estimators_) / len(est.estimators_)
                        for est in cv_results["estimator"]
                    ]
                    avg_nodes = sum(fold_avg_nodes) / len(fold_avg_nodes)

                    forests.append({
                        "model_name": f"RF_n{n_estimators}_depth{depth}_split{split}_leaf{leaf}",
                        "cv_f1_mean": score,
                        "cv_f1_std": cv_scores.std(),
                        "avg_nodes_per_tree": avg_nodes,
                        "params": (n_estimators, depth, split, leaf, max_feat)
                    })

                    if score > best_score:
                        best_score = score
                        best_forests_index = len(forests) - 1
                        best_params = (n_estimators, depth, split, leaf, max_feat)

print(
    f"Best Model: {forests[best_forests_index]['model_name']}\n"
    f"CV F1 Mean: {forests[best_forests_index]['cv_f1_mean']:.3f}\n"
    f"CV F1 SD: {forests[best_forests_index]['cv_f1_std']:.3f}\n"
    f"Avg Nodes/Tree: {forests[best_forests_index]['avg_nodes_per_tree']:.1f}\n"
    f"N Estimators: {best_params[0]}\n"
    f"Depth: {best_params[1]}\n"
    f"Split: {best_params[2]}\n"
    f"Leaf: {best_params[3]}\n"
    f"Max Features: {best_params[4]}"
)

# Refit best model on full training set
best_model = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    min_samples_split=best_params[2],
    min_samples_leaf=best_params[3],
    max_features = best_params[4],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# Final evaluation on test set
test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.3f}")

# Feature importances (averaged across all trees in the forest)
print("\nFeature Importances:")
for name, importance in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<10} {importance:.3f}")
