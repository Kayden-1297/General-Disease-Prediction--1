"""
train.py
Trains several classifiers on the disease-symptom dataset,
compares their accuracy, and saves the best model + label encoder
+ feature column order to the models/ folder.

Run from the project root:
    python src/train.py
"""

import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import load_and_clean, split_features_target, encode_target


def build_models():
    """Return a dict of model_name -> untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }


def main():
    print("Loading and cleaning data...")
    train_df, test_df = load_and_clean(
        "data/train_disease.csv", "data/test_disease.csv"
    )

    X_train, y_train_raw = split_features_target(train_df)
    X_test, y_test_raw = split_features_target(test_df)

    # Make sure test columns are in the exact same order as train columns
    X_test = X_test[X_train.columns]

    y_train, y_test, label_encoder = encode_target(y_train_raw, y_test_raw)

    print(f"Training rows: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Number of disease classes: {len(label_encoder.classes_)}\n")

    models = build_models()
    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 5-fold cross-validation on the training set for a more robust estimate
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        results.append({
            "model": name,
            "test_accuracy": acc,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_time_sec": round(train_time, 3),
        })
        print(f"{name:22s} | Test Acc: {acc:.4f} | CV Acc: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f}) | Time: {train_time:.3f}s")

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print("\n=== Model comparison (sorted by test accuracy) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    # Detailed report for the best model
    best_preds = best_model.predict(X_test)
    print("\nClassification report (best model):")
    print(classification_report(
        y_test, best_preds, target_names=label_encoder.classes_, zero_division=0
    ))

    # Save artifacts needed for prediction later
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    joblib.dump(list(X_train.columns), "models/feature_columns.joblib")
    results_df.to_csv("models/model_comparison.csv", index=False)

    print("\nSaved: models/best_model.joblib, models/label_encoder.joblib, "
          "models/feature_columns.joblib, models/model_comparison.csv")


if __name__ == "__main__":
    main()
