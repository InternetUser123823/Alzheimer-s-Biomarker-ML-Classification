import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
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


# Fill in missing data
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# Finds best model
# Stratified K Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set depth, splits, and leaves
depths = [2, 3, 4, 5, 6, 8, 10]
splits = [2, 5, 10, 20]
leaves = [10, 20]

trees = []
best_score = 0
best_model = None
best_params = None

# Make and train decision tree models
for depth in depths:
    for split in splits:
        for leaf in leaves:
            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )

            cv_results = cross_validate(
                tree, X_train, y_train,
                cv=skf, scoring="f1",
                return_estimator=True
            )
            cv_scores = cv_results["test_score"]
            score = cv_scores.mean()

            fold_node_counts = [est.tree_.node_count for est in cv_results["estimator"]]
            avg_nodes = sum(fold_node_counts) / len(fold_node_counts)

            trees.append({
                "model_name": f"DT_depth{depth}_split{split}_leaf{leaf}",
                "cv_f1_mean": score,
                "cv_f1_std": cv_scores.std(),
                "nodes": avg_nodes
            })

            if score > best_score:
                best_score = score
                best_tree_index = len(trees) - 1
                best_model = tree
                best_params = (depth, split, leaf)


# Evaluation
# Print out results of all the models
# for tree in trees:
#     print(
#         f"{tree['model_name']:<40} "
#         f"Train F1: {tree['train_f1']:.3f} | "
#         f"CV F1: {tree['cv_f1']:.3f} | "
#         f"Nodes: {tree['nodes']}"
#     )

# Print out the results of the best model:
print(
    f"Best Model: {trees[best_tree_index]["model_name"]}\n"
    f"CV F1 Mean: {trees[best_tree_index]["cv_f1_mean"]:.3f}\n"
    f"CV F1 SD: {trees[best_tree_index]["cv_f1_std"]:.3f}\n"
    f"Depth: {best_params[0]}\n"
    f"Split: {best_params[1]}\n"
    f"Leaf: {best_params[2]}\n"
    f"Number of Nodes: {trees[best_tree_index]["nodes"]}"
)

# Final evaluation on test set
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.3f}")
test_f1 = f1_score(y_test, test_pred)
print(f"Test F1: {test_f1:.3f}")

# Feature importances 
print("\nFeature Importances:")
for name, importance in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<10} {importance:.3f}")
    

# Plot best model
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=features,
    class_names=["No AD", "AD"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()
