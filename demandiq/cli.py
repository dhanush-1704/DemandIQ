"""Command-line model training and forecasting."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from .model import fit_and_evaluate, forecast_next_day, save_model


def main():
    parser = argparse.ArgumentParser(description="DemandIQ CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="artifacts")
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--input", required=True)
    forecast.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if args.command == "train":
        model, report = fit_and_evaluate(df)
        save_model(model, report, args.output)
        print(json.dumps(report, indent=2))
    else:
        model = joblib.load(Path(args.model))
        result = forecast_next_day(df, model)
        print(result.to_csv(index=False))


if __name__ == "__main__":
    main()
