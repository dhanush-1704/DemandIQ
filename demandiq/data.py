"""Input validation and leakage-safe feature generation."""
import numpy as np
import pandas as pd

KEYS = ["store_id", "product_id"]
REQUIRED = ["date", *KEYS, "units_sold", "price", "promotion"]
LAGS = (1, 7, 14, 28)
FEATURES = ["price", "promotion", "day_of_week", "month", "day_of_month",
            "lag_1", "lag_7", "lag_14", "lag_28", "mean_7", "mean_28"]


def validate_sales(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = frame[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for column in ["units_sold", "price", "promotion"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Required columns cannot contain nulls")
    if (df["units_sold"] < 0).any() or (df["price"] < 0).any():
        raise ValueError("Sales and prices must be nonnegative")
    if not df["promotion"].isin([0, 1]).all():
        raise ValueError("promotion must be 0 or 1")
    if df.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store-product-date rows")
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    for _, group in df.groupby(KEYS, sort=False):
        expected = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        if len(expected) != len(group):
            raise ValueError("Missing calendar dates for a store-product series")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features at date t only use demand from dates < t."""
    out = df.copy().sort_values(KEYS + ["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_month"] = out["date"].dt.day
    groups = out.groupby(KEYS, sort=False)["units_sold"]
    for lag in LAGS:
        out[f"lag_{lag}"] = groups.shift(lag)
    for window in (7, 28):
        out[f"mean_{window}"] = groups.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        )
    return out


def next_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one-day-ahead features with future price/promotion held constant."""
    latest = df.sort_values("date").groupby(KEYS, as_index=False).tail(1).copy()
    latest["date"] = latest["date"] + pd.Timedelta(days=1)
    latest["units_sold"] = np.nan
    combined = pd.concat([df, latest], ignore_index=True)
    features = make_features(combined)
    return features[features["units_sold"].isna()].copy()
