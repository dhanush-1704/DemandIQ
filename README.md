# DemandIQ — Demand Forecasting & Inventory Optimization

An end-to-end, reproducible data science project for retail demand prediction and inventory decision support.

## Features
- Ingest daily sales data with product, store, date, units, price, promotion and optional stock.
- Build **leakage-safe** lag and rolling demand features.
- Compare a seasonal-naive benchmark against a histogram gradient boosting regressor.
- Evaluate with chronological holdout, MAE, RMSE, WAPE and bias.
- Generate next-day demand forecasts and actionable reorder suggestions.
- Serve results through a FastAPI API.
- Automated unit tests and synthetic sample data generator.

> **Project status:** Foundation implemented. Real-dataset benchmarking, calibrated forecast intervals, SHAP, MLflow, PostgreSQL, frontend and deployment are planned. No model performance claims are made before evaluation.

## Tech stack
Python 3.11+, Pandas, NumPy, scikit-learn, FastAPI, Pydantic, pytest. The current model is scikit-learn's HistGradientBoostingRegressor; LightGBM and XGBoost are planned comparisons.

## Quick start

```bash
git clone https://github.com/dhanush-1704/DemandIQ.git
cd DemandIQ
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_sample_data.py
python -m demandiq.cli train --input data/sample_sales.csv --output artifacts
python -m demandiq.cli forecast --input data/sample_sales.csv --model artifacts/model.joblib
uvicorn demandiq.api:app --reload
```

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## Input schema
CSV columns: `date` (YYYY-MM-DD), `store_id`, `product_id`, `units_sold`, `price`, `promotion` (0/1). One row per store-product-date. Daily continuous observations are required for lag-based features. Missing dates should be investigated and explicitly imputed, not silently treated as zero sales.

## Modeling methodology
1. Validate schema, duplicates, nonnegative units and prices.
2. Construct calendar, demand lag (1, 7, 14, 28 days) and trailing means (7, 28 days), always using **prior days only**.
3. Use the final 28 days as a chronological holdout.
4. Fit the model on earlier observations only. Compare against a 7-day seasonal-naive baseline.
5. Report metrics on eligible holdout rows, including error bias. **This is a one-step-ahead, observed-history evaluation**: actual prior-day demand is available for each holdout prediction. Multi-day recursive forecasts require separate validation.
6. Train the final model on all eligible rows for one-day-ahead predictions.

## Inventory assumptions
The reorder calculator uses average daily demand, lead time, target coverage and a normal-approximation safety-stock formula. This is an illustrative heuristic, **not** a calibrated service-level guarantee. Real deployment requires lead-time variability, demand uncertainty, supplier constraints, inventory positions and cost optimization.

## API
- `GET /health`
- `POST /inventory/recommend` — JSON with `forecast_daily_units`, `on_hand`, `on_order`, `lead_time_days`, `review_period_days`, `daily_demand_std`, `service_z`.

## Tests
```bash
pytest -q
```

## Roadmap
- Public retail dataset with documented licensing and source
- Rolling-origin backtesting, multiple forecast horizons and probabilistic calibration
- LightGBM/XGBoost/StatsForecast comparisons, SHAP, MLflow
- PostgreSQL pipeline and scheduled refresh
- Next.js analytics dashboard and Docker deployment

## Limitations
Synthetic sample data is for demonstration, not evidence of generalization. Lagged features assume complete daily records; cold-start products need a separate approach. Inventory formulas are approximate and must be validated against business costs.

## License
MIT (see LICENSE).
