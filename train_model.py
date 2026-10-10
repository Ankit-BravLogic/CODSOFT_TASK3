
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# 2. Remove identifier columns that should not be model features
X = df.drop(
    columns=["RowNumber", "CustomerId", "Surname", "Exited"]
)
y = df["Exited"]

# 3. Identify numerical and categorical columns
categorical_features = ["Geography", "Gender"]
numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 4. Build preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Split the data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")

# 6. Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}

# 7. Train and evaluate every model
results = []
trained_models = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    results.append(metrics)
    trained_models[name] = pipeline

# 8. Compare the results
results_df = pd.DataFrame(results).sort_values(
    by="F1 Score", ascending=False
)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4).to_string(index=False))

# 9. Select the best model by F1-score
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest model by F1-score:", best_name)
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, best_predictions))

# 10. Save the trained pipeline and test information
Path("models").mkdir(exist_ok=True)

joblib.dump(best_model, "models/churn_model.pkl")

results_df.to_csv("models/model_comparison.csv", index=False)

print("\nSaved model: models/churn_model.pkl")
print("Saved comparison: models/model_comparison.csv")
