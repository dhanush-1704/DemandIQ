import pandas as pd
import pytest
from demandiq.data import validate_sales, make_features
from demandiq.inventory import InventoryRequest, recommend_inventory
from demandiq.model import fit_and_evaluate, forecast_next_day


def sample(n=110):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n),
        "store_id": ["S1"] * n,
        "product_id": ["P1"] * n,
        "units_sold": [10 + (i % 7) for i in range(n)],
        "price": [12.0] * n,
        "promotion": [0] * n,
    })


def test_lags_use_only_past():
    features = make_features(validate_sales(sample()))
    assert pd.isna(features.iloc[0]["lag_1"])
    assert features.iloc[1]["lag_1"] == 10
    assert features.iloc[7]["lag_7"] == 10
    assert features.iloc[7]["mean_7"] == pytest.approx(13)


def test_duplicate_rejected():
    data = sample()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales(pd.concat([data, data.iloc[[0]]]))


def test_gap_rejected():
    with pytest.raises(ValueError, match="Missing calendar"):
        validate_sales(sample().drop(index=10))


def test_train_and_forecast():
    model, report = fit_and_evaluate(sample())
    assert report["model"]["n"] == 28
    assert report["seasonal_naive_7"]["mae"] == 0
    forecast = forecast_next_day(sample(), model)
    assert len(forecast) == 1
    assert forecast.iloc[0]["forecast_units"] >= 0


def test_inventory_order():
    req = InventoryRequest(forecast_daily_units=10, on_hand=5, on_order=0,
                           lead_time_days=3, review_period_days=1,
                           daily_demand_std=0)
    assert recommend_inventory(req)["recommended_order_units"] == 35
