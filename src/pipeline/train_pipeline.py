"""
End-to-end training pipeline for Stress Level Prediction.
Loads a CSV from data/raw, preprocesses, trains multiple models, evaluates,
saves the best model and artifacts.

Usage:
  python -m src.pipeline.train_pipeline --file stress_data.csv --target stress_level

If --file is omitted, the first CSV in data/raw is used.
If --target is omitted, the default from config.TARGET_COLUMN is used.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import config
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_selector import FeatureSelector
from src.models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_default_csv(raw_dir: Path) -> Optional[Path]:
    csvs = sorted(raw_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def encode_categoricals_one_hot(df: pd.DataFrame, cat_cols: Optional[list[str]] = None) -> pd.DataFrame:
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, drop_first=False)


def save_artifacts(artifact: Dict[str, Any], model_name: str = "best") -> Path:
    save_path = config.get_model_save_path(model_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, save_path)
    logger.info(f"Saved artifact to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train stress level prediction models")
    parser.add_argument("--file", type=str, default=None, help="CSV filename located under data/raw/")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE, help="Test split size")
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE, help="Random seed")
    parser.add_argument("--corr-threshold", type=float, default=config.CORRELATION_THRESHOLD, help="Correlation threshold")
    parser.add_argument("--var-threshold", type=float, default=config.VARIANCE_THRESHOLD, help="Low variance threshold")
    args = parser.parse_args()

    config.create_directories()

    raw_dir = config.RAW_DATA_DIR
    filename = args.file
    if filename is None:
        default_csv = find_default_csv(raw_dir)
        if default_csv is None:
            raise FileNotFoundError(f"No CSV found in {raw_dir}. Please place your file there or pass --file.")
        filename = default_csv.name
        logger.info(f"Using detected CSV: {filename}")

    target_col = args.target or config.TARGET_COLUMN

    # Load
    loader = DataLoader(str(raw_dir))
    df = loader.load_csv(filename)
    loader.validate_target_column(df, target_col)

    # Standardize common missing tokens and trim whitespace in string columns
    missing_tokens = {"", "NA", "N/A", "na", "n/a", "-", "--", "None", "none", "NULL", "null"}
    df = df.replace(list(missing_tokens), np.nan)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Basic cleaning
    pre = DataPreprocessor()
    df = pre.remove_duplicates(df)
    df = pre.handle_missing_values(df, strategy="mean")

    # Ensure target has no missing values (drop remaining null targets if any)
    if df[target_col].isna().any():
        n_drop = int(df[target_col].isna().sum())
        logger.info(f"Dropping {n_drop} rows with missing target '{target_col}' after imputation")
        df = df[df[target_col].notna()].copy()

    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Label encode target if categorical
    target_encoder: Optional[LabelEncoder] = None
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        logger.info(f"Encoded target '{target_col}' with classes: {list(target_encoder.classes_)}")

    # One-hot encode ALL non-numeric features for robust inference
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = encode_categoricals_one_hot(X, cat_cols)

    # Remove highly correlated features (optional but helpful)
    if X.shape[1] > 1:
        fs = FeatureSelector()
        temp = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
        temp = fs.correlation_analysis(temp, target_col=target_col, threshold=args.corr_threshold)
        y = temp[target_col].values
        X = temp.drop(columns=[target_col])

    # Remove low variance features
    if X.shape[1] > 1:
        variances = X.var()
        keep_cols = variances[variances >= args.var_threshold].index.tolist()
        dropped = [c for c in X.columns if c not in keep_cols]
        if dropped:
            logger.info(f"Dropping {len(dropped)} low variance features")
        X = X[keep_cols]

    feature_columns = X.columns.tolist()

    # Final guard: drop any rows with NaN in y (and align X accordingly)
    y_series = pd.Series(y, name=target_col)
    if y_series.isna().any():
        n = int(y_series.isna().sum())
        logger.info(f"Final guard: dropping {n} rows with NaN in target before split")
        keep_idx = y_series.notna()
        X = X.loc[keep_idx].reset_index(drop=True)
        y_series = y_series.loc[keep_idx].reset_index(drop=True)
        y = y_series.values

    # Extra guard: ensure X has no NaNs
    if X.isna().any().any():
        # Conservative approach: drop rows with any NaN remaining
        before = len(X)
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = np.asarray(y)[mask.values]
        logger.info(f"Dropped {before - len(X)} rows with NaNs in features after preprocessing")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Scale
    X_train_scaled, X_test_scaled = pre.scale_features(X_train, X_test)

    # Train and evaluate
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_all_models(X_train_scaled, y_train)
    trainer.evaluate_all_models(X_test_scaled, y_test)
    best_name, best_model, best_scores = trainer.get_best_model()

    # Save artifacts
    artifact = {
        "model_name": best_name,
        "model": best_model,
        "scaler": pre.scalers.get("features"),
        "feature_columns": feature_columns,
        "target_column": target_col,
        "target_encoder": target_encoder,
        "metrics": best_scores,
        "training_file": filename,
    }
    model_path = save_artifacts(artifact, model_name="best")

    # Save metrics text summary
    results_path = config.get_results_save_path("training_metrics")
    with open(results_path, "w") as f:
        f.write(f"Best model: {best_name}\n")
        for k, v in best_scores.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"Model saved at: {model_path}\n")
    logger.info(f"Metrics saved to {results_path}")

    logger.info("Training complete.")
    logger.info(f"Best model: {best_name} | Acc: {best_scores['accuracy']:.4f} | F1: {best_scores['f1_score']:.4f}")


if __name__ == "__main__":
    main()
