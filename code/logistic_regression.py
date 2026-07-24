import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score

# Silence L2 warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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


# Feature scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_cv = scaler.transform(X_cv)
X_test = scaler.transform(X_test)


# Fill in missing data
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
# X_cv = imputer.transform(X_cv)
X_test = imputer.transform(X_test)

# Create a new dataset for polynomial model training
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
# X_cv_poly = poly.transform(X_cv)
X_test_poly = poly.transform(X_test)

# Statistics on the dataset
# print("Linear training set:", X_train.shape)
# print("Linear CV set:", X_cv.shape)
# print("Linear test set:", X_test.shape)

# print("Poly training set:", X_train_poly.shape)
# print("Poly CV set:", X_cv_poly.shape)
# print("Poly test set:", X_test_poly.shape)


# Create and train linear models
lambda_values = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 20]
models = []

# Linear models
for lam in lambda_values:
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        C=1/lam,
        max_iter=1000
    )

    models.append({
        "name": f"linear_lambda_{lam}",
        "type": "linear",
        "lambda": lam,
        "model": model,
        "X_train": X_train,
        "X_test": X_test
    })

# Polynomial models
for lam in lambda_values:
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        C=1/lam,
        max_iter=1000
    )

    models.append({
        "name": f"poly_lambda_{lam}",
        "type": "polynomial",
        "lambda": lam,
        "model": model,
        "X_train": X_train_poly,
        "X_test": X_test_poly
    })


# Evaluation of models' F1 values based on threshold
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

best_f1 = 0.0
best_model = None
best_name = ""
best_threshold = 0.5

# Stratified K Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_dict in models:
    X_data = model_dict["X_train"]

    fold_best_f1s = []       # each fold's best F1 (at that fold's best threshold)
    fold_best_thresholds = [] # each fold's best threshold

    for train_idx, val_idx in skf.split(X_data, y_train):
        X_fold_train, X_fold_val = X_data[train_idx], X_data[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        fold_model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            C=1/model_dict["lambda"],
            max_iter=1000
        )
        fold_model.fit(X_fold_train, y_fold_train)

        val_prob = fold_model.predict_proba(X_fold_val)[:, 1]

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

    mean_cv_f1 = sum(fold_best_f1s) / len(fold_best_f1s)
    # use the most common best threshold across folds
    mean_best_threshold = max(set(fold_best_thresholds), key=fold_best_thresholds.count)

    # print(f"{model_dict['name']} | CV F1={mean_cv_f1:.3f} | Threshold={mean_best_threshold}")

    if mean_cv_f1 > best_f1:
        best_model = model_dict["model"]
        best_f1 = mean_cv_f1
        best_name = model_dict["name"]
        best_X_test_set = model_dict["X_test"]
        best_threshold = mean_best_threshold
        best_lambda = model_dict["lambda"]
        best_type = model_dict["type"]

print(f"\nBest model: {best_name}")
print(f"Best CV F1 score: {best_f1:.3f}")
print(f"Best threshold: {best_threshold}")

# Final evaluation on test set
best_model.fit(X_train, y_train)
test_prob = best_model.predict_proba(best_X_test_set)[:,1]
test_pred = (test_prob >= best_threshold).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.3f}")
test_f1 = f1_score(y_test, test_pred)
print(f"Test F1: {test_f1:.3f}")

# Feature coefficients (log-odds weights)
print("\nFeature Coefficients:")
for name, coef in sorted(zip(features, best_model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {name:<10} {coef:+.3f}")

# Further analysis (see README)
print(f"Correlation: {df[["MMSE", "nWBV"]].corr()}")
