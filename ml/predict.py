"""
predict.py
Loads the saved model and lets you predict a disease from a list
of symptom names you type in.

Run from the project root:
    python src/predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"
COLUMNS_PATH = "models/feature_columns.joblib"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, label_encoder, feature_columns


def predict_disease(symptom_list, model, label_encoder, feature_columns, top_n=3):
    """
    symptom_list: list of strings, e.g. ["itching", "skin_rash", "nodal_skin_eruptions"]
    Returns the top_n predicted diseases with probabilities (if supported).
    """
    # Build a single-row input vector matching the training feature order
    input_row = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)

    unknown = []
    for symptom in symptom_list:
        symptom = symptom.strip()
        if symptom in input_row.columns:
            input_row.at[0, symptom] = 1
        else:
            unknown.append(symptom)

    if unknown:
        print(f"Warning: these symptoms were not recognized and were ignored: {unknown}")
        print("Tip: run with --list-symptoms to see all valid symptom names.\n")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_row)[0]
        top_idx = probs.argsort()[::-1][:top_n]
        results = [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_idx]
    else:
        pred = model.predict(input_row)[0]
        results = [(label_encoder.inverse_transform([pred])[0], None)]

    return results


if __name__ == "__main__":
    import sys

    model, label_encoder, feature_columns = load_artifacts()

    if "--list-symptoms" in sys.argv:
        print("Valid symptom names (use exactly as shown, comma-separated):\n")
        for col in feature_columns:
            print(" -", col)
        sys.exit(0)

    print("=== Disease Prediction Tool ===")
    print("Type comma-separated symptoms from the dataset (e.g. itching,skin_rash,nodal_skin_eruptions)")
    print("Run 'python src/predict.py --list-symptoms' to see all valid symptom names.\n")

    user_input = input("Enter symptoms: ")
    symptoms = [s.strip() for s in user_input.split(",") if s.strip()]

    if not symptoms:
        print("No symptoms entered. Exiting.")
        sys.exit(0)

    results = predict_disease(symptoms, model, label_encoder, feature_columns)

    print("\nTop predictions:")
    for disease, prob in results:
        if prob is not None:
            print(f"  {disease:35s}  {prob*100:.1f}%")
        else:
            print(f"  {disease}")
