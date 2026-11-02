# Disease Prediction from Symptoms — ML Project

Predicts one of 41 diseases from a set of 132 binary symptom features,
using classic ML models (Logistic Regression, Decision Tree, Random Forest,
SVM, KNN, Naive Bayes) trained and compared automatically.

## Project structure

```
Model-2/
├── data/
│   ├── train_disease.csv
│   └── test_disease.csv
├── ml/
│   ├── preprocess.py      # loads & cleans the data
│   ├── train.py           # trains + compares all models, saves the best one
│   └── predict.py         # loads saved model, predicts from symptoms you type
├── models/                # saved model + encoder
├── frontend/
│   └── app.py              # Flask web interface
├── notebooks/
│   └── disease_prediction_colab.ipynb
├── requirements.txt
└── README.md
```

---

## PART 1 — Build & run it in VS Code (from scratch)

### Step 1: Install prerequisites
- Install **Python 3.9+** from https://python.org (check "Add to PATH" during install on Windows).
- Install **VS Code** from https://code.visualstudio.com.
- In VS Code, install the **Python extension** (Ms-Python) from the Extensions panel (Ctrl+Shift+X).

### Step 2: Open the project folder
Open this folder in VS Code: `File > Open Folder...` → select `Model-2`.

### Step 3: Add the dataset
Copy `train_disease.csv` and `test_disease.csv` into the `data/` folder.

### Step 4: Create a virtual environment
This keeps your project's packages isolated from the rest of your system.
```bash
python -m venv venv
```
Activate it:
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **Mac/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal prompt will show `(venv)` at the start.
In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the one inside `venv`.

### Step 5: Install the required libraries
```bash
pip install -r requirements.txt
```
(The requirements also include Flask for the web interface.)

### Step 6: Run the existing source files
The source files are already in the `ml/` folder.

### Step 7: Run preprocessing sanity check
```bash
python ml/preprocess.py
```
Expected output: train/test shapes, 41 classes, no nulls. This confirms your data loaded correctly.

### Step 8: Train and compare models
```bash
python ml/train.py
```
This will:
- Train 6 different models
- Print each model's test accuracy, 5-fold cross-validation accuracy, and training time
- Print a full classification report for the best model
- Save the best model to `models/best_model.joblib`, the label encoder to `models/label_encoder.joblib`, the feature column order to `models/feature_columns.joblib`, and a comparison table to `models/model_comparison.csv`

On this dataset you should see accuracy at or near **100%** across most models — that's expected, the symptom patterns per disease are cleanly distinguishable in this dataset. If you ever retrain on a *messier* real-world dataset, don't expect this — perfect accuracy there is usually a red flag for a data leak.

### Step 9: Test the model with your own symptoms
```bash
python ml/predict.py
```
When prompted, type comma-separated symptom names, e.g.:
```
itching,skin_rash,nodal_skin_eruptions
```
The script prints the top 3 predicted diseases with confidence percentages.

To see the exact list of valid symptom names (spelling matters):
```bash
python ml/predict.py --list-symptoms
```

### Step 10 (optional): Re-run and iterate
- Try tweaking `RandomForestClassifier(n_estimators=...)` or other hyperparameters in `train.py`.
- Add a `GridSearchCV` if you want to tune hyperparameters instead of using defaults.
- To use a different model than the auto-picked "best" one, just change the `best_name` logic in `train.py`.

---

## PART 2 — Run the same project in Google Colab

Colab is useful if you don't want to install anything locally, or want free GPU/CPU compute.

### Step 1: Open Colab
Go to https://colab.research.google.com and sign in with your Google account.

### Step 2: Upload the notebook
`File > Upload notebook` → select `disease_prediction_colab.ipynb` (provided).

Or start a blank notebook and copy in the code cells manually from `notebooks/disease_prediction_colab.ipynb`.

### Step 3: Upload your data
Run the first code cell — it will prompt a file picker. Upload **both**
`train_disease.csv` and `test_disease.csv` when asked.

(Alternative: mount Google Drive instead with
```python
from google.colab import drive
drive.mount('/content/drive')
```
and point `pd.read_csv(...)` at the file path in your Drive.)

### Step 4: Run all cells
`Runtime > Run all`. The notebook will:
1. Load and clean the data
2. Show a class-balance chart
3. Train and compare 6 models
4. Show a classification report + confusion matrix for the best model
5. Save and **download** the trained model files to your computer
6. Let you test predictions with your own symptom list in the last cells

### Step 5: Download your trained model
The notebook automatically triggers browser downloads for:
- `best_model.joblib`
- `label_encoder.joblib`
- `feature_columns.joblib`

You can drop these into your local `models/` folder and use them with the same
`ml/predict.py` script from Part 1 — no need to retrain locally.

### Step 11 (optional): Run the web interface
From the project root, run:
```bash
python frontend/app.py
```
Then open http://127.0.0.1:5000 in your browser. Keep the terminal open while
using the application.

---

## Notes on this dataset
- 132 symptom columns (binary: 0 = absent, 1 = present), 1 target column `prognosis` (41 disease names).
- The raw `train_disease.csv` has a stray trailing comma in its header row that creates a junk "Unnamed" column — the preprocessing code drops it automatically.
- The training data is synthetically generated: each disease has 120 rows, but many of those rows are literal duplicates of the same symptom pattern. This is normal for this dataset and doesn't need to be "fixed."
- Because the symptom-to-disease mapping is deterministic and non-overlapping in this dataset, near-perfect accuracy is expected and is not a bug.
