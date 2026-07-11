from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Get the current project folder
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    BASE_DIR / "SOURCE(1).csv"
)

print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Remove ID because it is only an identifier
df = df.drop(
    columns=["ID"],
    errors="ignore"
)

# Fill missing numerical values with median
df["Work_Experience"] = df["Work_Experience"].fillna(
    df["Work_Experience"].median()
)

df["Family_Size"] = df["Family_Size"].fillna(
    df["Family_Size"].median()
)

# Fill missing categorical values with mode
categorical_columns = [
    "Ever_Married",
    "Graduated",
    "Profession",
    "Var_1"
]

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# ============================================================
# 3. HANDLE OUTLIERS
# ============================================================

numerical_columns = [
    "Age",
    "Work_Experience",
    "Family_Size"
]

outlier_limits = {}

for column in numerical_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    outlier_limits[column] = {
        "lower": lower_limit,
        "upper": upper_limit
    }

    df[column] = df[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Spending_Score"]
)

y = df["Spending_Score"]

# Convert categorical features into numbers
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

# Encode target variable
le = LabelEncoder()

y = le.fit_transform(y)

print(
    "Class mapping:",
    dict(
        zip(
            le.classes_,
            le.transform(le.classes_)
        )
    )
)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 6. CREATE MODEL PIPELINES
# ============================================================

pipe_lr = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lr",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


pipe_rf = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "rf",
        RandomForestClassifier(
            random_state=42
        )
    )
])


pipe_svc = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svc",
        SVC(
            probability=True,
            random_state=42
        )
    )
])


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

grid_lr = {
    "lr__C": [
        0.01,
        0.1,
        1,
        10
    ]
}


grid_rf = {
    "rf__n_estimators": [
        50,
        100
    ],

    "rf__max_depth": [
        3,
        5,
        10
    ]
}


grid_svc = {
    "svc__C": [
        0.1,
        1,
        10
    ],

    "svc__kernel": [
        "linear",
        "rbf"
    ]
}


cv_lr = GridSearchCV(
    pipe_lr,
    param_grid=grid_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)


cv_rf = GridSearchCV(
    pipe_rf,
    param_grid=grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)


cv_svc = GridSearchCV(
    pipe_svc,
    param_grid=grid_svc,
    cv=5,
    scoring="accuracy",
    n_jobs=1
)


# ============================================================
# 8. TRAIN MODELS
# ============================================================

print("\nTraining Logistic Regression...")
cv_lr.fit(X_train, y_train)

print("Training Random Forest...")
cv_rf.fit(X_train, y_train)

print("Training Support Vector Machine...")
cv_svc.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 9. EVALUATE MODELS
# ============================================================

models = {
    "Logistic Regression": cv_lr,
    "Random Forest": cv_rf,
    "Support Vector Machine": cv_svc
}

results = []

for model_name, model in models.items():

    prediction = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
        multi_class="ovr",
        average="weighted"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "CV Accuracy": model.best_score_,
        "Best Parameters": str(
            model.best_params_
        )
    })


results_df = pd.DataFrame(results)

print("\nModel results:")
print(
    results_df.round(4)
)


# ============================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = (
    cv_rf
    .best_estimator_
    .named_steps["rf"]
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 11. SAVE MODEL FILES
# ============================================================

joblib.dump(
    cv_rf.best_estimator_,
    BASE_DIR / "random_forest_model.joblib"
)

joblib.dump(
    X.columns.tolist(),
    BASE_DIR / "feature_columns.joblib"
)

joblib.dump(
    le,
    BASE_DIR / "label_encoder.joblib"
)

joblib.dump(
    outlier_limits,
    BASE_DIR / "outlier_limits.joblib"
)


# Save evaluation results
results_df.to_csv(
    BASE_DIR / "model_results.csv",
    index=False
)

feature_importance.to_csv(
    BASE_DIR / "feature_importance.csv",
    index=False
)


print("\nFiles generated successfully:")

print("1. random_forest_model.joblib")
print("2. feature_columns.joblib")
print("3. label_encoder.joblib")
print("4. outlier_limits.joblib")
print("5. model_results.csv")
print("6. feature_importance.csv")