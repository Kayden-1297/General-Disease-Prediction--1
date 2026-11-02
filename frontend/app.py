import threading
import webbrowser
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import load_artifacts, predict_disease

app = Flask(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.joblib"

model, label_encoder, feature_columns = load_artifacts()
comparison_path = PROJECT_ROOT / "models" / "model_comparison.csv"
comparison = pd.read_csv(comparison_path).fillna("").to_dict(orient="records")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Disease Prediction</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    background: #eef2ff;
    color: #172033;
}
header {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    color: white;
    padding: 60px 20px;
    text-align: center;
}
header h1 {
    margin: 0 0 12px;
    font-size: 42px;
}
header p {
    margin: 0;
    opacity: 0.9;
    font-size: 17px;
}
main {
    max-width: 900px;
    margin: 35px auto;
    padding: 0 18px;
}
.card {
    background: white;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 15px 35px rgba(30, 41, 59, 0.12);
}
h2 { margin-top: 0; }
textarea, input {
    width: 100%;
    padding: 13px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    font-size: 16px;
}
button {
    width: 100%;
    margin-top: 18px;
    padding: 15px;
    border: 0;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s;
}
button:hover {
    background: #1d4ed8;
    transform: translateY(-1px);
}
#results { margin-top: 20px; }
.result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 17px;
    margin: 12px 0;
    background: #f8fafc;
    border: 1px solid #dbeafe;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
}
.symptoms {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-height: 230px;
    overflow-y: auto;
}
.symptom {
    padding: 7px 10px;
    background: #e2e8f0;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
}
.symptom.selected { background: #2563eb; color: white; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }
th { background: #f8fafc; }
@media (max-width: 600px) {
    .result { flex-direction: column; gap: 6px; }
    table { font-size: 12px; }
}
</style>
</head>
<body>
<header>
    <h1>Disease Prediction System</h1>
    <p>Select symptoms to receive the top disease predictions</p>
</header>

<main>
    <section class="card">
        <h2>Enter Symptoms</h2>
        <p>Click symptoms below or type comma-separated values.</p>
        <input id="search" placeholder="Search symptoms...">
        <div id="symptoms" class="symptoms"></div>
        <textarea id="selected" rows="3"
            placeholder="Example: itching, skin_rash"></textarea>
        <button onclick="predict()">Predict Disease</button>
        <div id="results"></div>
    </section>
</main>

<script>
let allSymptoms = [];
let selectedSymptoms = new Set();

async function loadData() {
    const response = await fetch("/api/data");
    const data = await response.json();

    allSymptoms = data.symptoms;
    renderSymptoms(allSymptoms);

}

function renderSymptoms(symptoms) {
    document.getElementById("symptoms").innerHTML = symptoms.map(symptom => `
        <span class="symptom ${selectedSymptoms.has(symptom) ? "selected" : ""}"
              onclick="toggleSymptom('${symptom}')">${symptom}</span>
    `).join("");
}

function toggleSymptom(symptom) {
    if (selectedSymptoms.has(symptom)) {
        selectedSymptoms.delete(symptom);
    } else {
        selectedSymptoms.add(symptom);
    }

    document.getElementById("selected").value =
        Array.from(selectedSymptoms).join(", ");
    renderSymptoms(allSymptoms);
}

document.getElementById("search").addEventListener("input", event => {
    const value = event.target.value.toLowerCase();
    renderSymptoms(allSymptoms.filter(s => s.toLowerCase().includes(value)));
});

async function predict() {
    const symptoms = document.getElementById("selected").value
        .split(",")
        .map(s => s.trim())
        .filter(Boolean);

    if (!symptoms.length) {
        alert("Please select or enter at least one symptom.");
        return;
    }

    const response = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({symptoms})
    });

    const data = await response.json();

    document.getElementById("results").innerHTML = `
        <h2>Top Predictions</h2>
        ${data.results.map(result => `
            <div class="result">
                <strong>${result.disease}</strong>
                <span>${result.probability === null
                    ? "Prediction available"
                    : (result.probability * 100).toFixed(1) + "%"}</span>
            </div>
        `).join("")}
        ${data.unknown.length
            ? `<p>Unrecognized symptoms: ${data.unknown.join(", ")}</p>`
            : ""}
    `;
}

loadData();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/data")
def data():
    return jsonify({
        "symptoms": feature_columns
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    symptoms = request.json.get("symptoms", [])

    known = [s.strip() for s in symptoms if s.strip() in feature_columns]
    unknown = [s.strip() for s in symptoms if s.strip() not in feature_columns]

    results = predict_disease(
        known,
        model,
        label_encoder,
        feature_columns,
        top_n=3
    )

    return jsonify({
        "results": [
            {
                "disease": disease,
                "probability": float(probability)
                if probability is not None else None
            }
            for disease, probability in results
        ],
        "unknown": unknown
    })

if __name__ == "__main__":
    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.2, open_browser).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )