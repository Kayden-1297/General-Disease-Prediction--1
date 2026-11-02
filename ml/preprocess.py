"""
preprocess.py
Loads the raw train/test CSVs, cleans them, and returns
feature matrix (X) and encoded labels (y) ready for modeling.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean(train_path: str, test_path: str):
    """Load train/test CSVs and drop junk columns."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # The raw train CSV has a trailing empty column caused by a stray
    # trailing comma in the header row. Drop any "Unnamed" columns.
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    # Strip whitespace from column names (some have trailing spaces)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # NOTE: This dataset is synthetically generated, so many rows repeat
    # the exact same symptom pattern per disease (120 rows/class, only a
    # handful of unique patterns). We deliberately do NOT drop duplicates
    # here because they carry real signal about how often each pattern
    # occurs, and dropping them makes classes imbalanced (5-10 rows each).

    return train_df, test_df


def split_features_target(df: pd.DataFrame, target_col: str = "prognosis"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def encode_target(y_train: pd.Series, y_test: pd.Series):
    """Fit a LabelEncoder on train labels, apply to both."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


if __name__ == "__main__":
    train_df, test_df = load_and_clean(
        "data/train_disease.csv", "data/test_disease.csv"
    )
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Classes:", train_df["prognosis"].nunique())
    print("Any nulls in train?", train_df.isnull().values.any())
    print("Any nulls in test?", test_df.isnull().values.any())
