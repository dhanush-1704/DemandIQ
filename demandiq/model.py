"""Train and evaluate a one-step-ahead demand model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .data import FEATURES, make_features, validate_sales, next_day_features


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / max(np.abs(actual).sum(), 1e-9)),
        "bias": float(error.mean()),
        "n": int(len(actual)),
    }


def fit_and_evaluate(raw: pd.DataFrame, holdout_days: int = 28):
    df = make_features(validate_sales(raw)).dropna(subset=FEATURES)
    if df.empty:
        raise ValueError("Not enough history: at least 29 daily records per series")
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days - 1)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Insufficient train/test history")
    model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06,
                                          max_leaf_nodes=15, random_state=42)
    model.fit(train[FEATURES], train["units_sold"])
    pred = np.maximum(model.predict(test[FEATURES]), 0)
    baseline = test["lag_7"].to_numpy()
    report = {
        "evaluation": "chronological one-step-ahead with observed prior demand",
        "cutoff": str(cutoff.date()),
        "model": metrics(test["units_sold"], pred),
        "seasonal_naive_7": metrics(test["units_sold"], baseline),
    }
    final = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.06,
                                          max_leaf_nodes=15, random_state=42)
    final.fit(df[FEATURES], df["units_sold"])
    return final, report


def save_model(model, report, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    (path / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def forecast_next_day(raw: pd.DataFrame, model):
    df = validate_sales(raw)
    future = next_day_features(df).dropna(subset=FEATURES)
    if future.empty:
        raise ValueError("Not enough history for next-day forecasting")
    future = future.copy()
    future["forecast_units"] = np.maximum(model.predict(future[FEATURES]), 0)
    return future[["date", "store_id", "product_id", "forecast_units"]].reset_index(drop=True)
