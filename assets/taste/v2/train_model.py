import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_CSV = Path("results.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def parse_feature_map(raw):
    """
    Convert DynamoDB-style feature_map to flat numeric dict
    Example input:
    {
        "savory_depth__gap": {"N": "0.123"},
        "heat__pos_mean": {"N": "0.22"}
    }
    """
    if isinstance(raw, str):
        raw = json.loads(raw)

    parsed = {}

    for k, v in raw.items():

        if isinstance(v, dict) and "N" in v:
            parsed[k] = float(v["N"])

        elif isinstance(v, (int, float)):
            parsed[k] = float(v)

        else:
            # fallback safety
            try:
                parsed[k] = float(v)
            except Exception:
                parsed[k] = 0.0

    return parsed


def load_feature_maps(df):

    feature_maps = []

    for raw in df["feature_map"]:
        fm = parse_feature_map(raw)
        feature_maps.append(fm)

    return feature_maps


def collect_feature_names(feature_maps):

    names = set()

    for fm in feature_maps:
        names.update(fm.keys())

    return sorted(names)


def build_matrix(feature_maps, feature_names):

    X = []

    for fm in feature_maps:

        row = [fm.get(name, 0.0) for name in feature_names]

        X.append(row)

    return np.array(X)


def main():

    if not INPUT_CSV.exists():
        raise RuntimeError("results.csv not found")

    df = pd.read_csv(INPUT_CSV)

    df = df[df["actual_score_1_to_5"].notna()]

    print("Rows loaded:", len(df))

    feature_maps = load_feature_maps(df)

    feature_names = collect_feature_names(feature_maps)

    print("Feature count:", len(feature_names))

    X = build_matrix(feature_maps, feature_names)

    y = df["actual_score_1_to_5"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=20,
        min_samples_leaf=2,
        l2_regularization=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_pred = np.clip(model.predict(X_train), 1, 5)
    test_pred = np.clip(model.predict(X_test), 1, 5)

    print("\nTraining metrics\n")

    print("Train MAE:", mean_absolute_error(y_train, train_pred))
    print("Test MAE:", mean_absolute_error(y_test, test_pred))

    print("Train RMSE:", rmse(y_train, train_pred))
    print("Test RMSE:", rmse(y_test, test_pred))

    MODEL_DIR.mkdir(exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCHEMA_PATH, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    print("\nSaved model ->", MODEL_PATH)
    print("Saved schema ->", SCHEMA_PATH)


if __name__ == "__main__":
    main()
