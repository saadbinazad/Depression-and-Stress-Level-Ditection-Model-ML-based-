"""
Predict stress levels using the saved best model artifact.

Usage:
  python -m src.pipeline.predict --file data/raw/stress_data.csv --head 5
  python -m src.pipeline.predict --file path/to/new_samples.csv --output reports/results/predictions.csv

Notes:
- Expects the saved artifact created by train_pipeline.py in models/best_model.joblib
- Automatically aligns columns (adds missing columns as 0, drops unknown extra columns)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Train first.")
    return joblib.load(path)


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    # One-hot for categorical features to approximate training pipeline
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    # Add missing columns with zeros
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    # Keep only known columns in the right order
    df = df[feature_columns]
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict stress levels using the saved model")
    parser.add_argument("--file", type=str, required=True, help="CSV file with samples to predict")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save predictions CSV")
    parser.add_argument("--head", type=int, default=0, help="Print first N predictions to console")
    args = parser.parse_args()

    artifact_path = config.get_model_save_path("best")
    artifact = load_artifact(artifact_path)

    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_columns = artifact["feature_columns"]
    target_encoder = artifact.get("target_encoder")

    df = pd.read_csv(args.file)
    if artifact.get("target_column") in df.columns:
        df = df.drop(columns=[artifact["target_column"]])

    X = align_features(df, feature_columns)
    if scaler is not None:
        known = getattr(scaler, "feature_names_in_", None)
        if known is not None:
            # Ensure all known columns exist, add missing as zeros
            for col in known:
                if col not in X.columns:
                    X[col] = 0
            # Transform only known columns and keep others untouched
            X[list(known)] = scaler.transform(X[list(known)])
        else:
            # Fallback: transform all feature columns
            X[feature_columns] = scaler.transform(X[feature_columns])

    preds = model.predict(X)
    if target_encoder is not None:
        # inverse_transform expects integers; if model outputs non-int, cast
        try:
            preds = target_encoder.inverse_transform(preds.astype(int))
        except Exception:
            pass

    result_df = df.copy()
    result_df["predicted_stress_level"] = preds

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        logger.info(f"Predictions saved to {out_path}")

    if args.head and args.head > 0:
        print(result_df.head(args.head))


if __name__ == "__main__":
    main()
