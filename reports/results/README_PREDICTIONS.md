# Stress Level Prediction - Run Summary

This document summarizes the exact steps performed to train the model and generate predictions from your provided dataset.

## 1) Data Source
- Input file: `data/raw/stress_data.csv`
- Detected target column: `Stress Label`
- Rows loaded: ~1977 (3 duplicates removed)

## 2) Preprocessing
- Standardized missing tokens: "", NA, N/A, na, n/a, -, --, None, none, NULL, null → treated as missing
- Trimmed whitespace in text columns
- Missing values imputed:
  - Numeric: mean imputation
  - Categorical: most_frequent (via safeguards)
- Dropped any rows with missing target after cleaning
- Categorical features one-hot encoded
- Correlation pruning: removed highly correlated features (threshold: 0.8)
- Low-variance feature removal (threshold: 0.01)
- Standardized numerical features (StandardScaler)

## 3) Modeling
- Models trained:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - SVM (probability=True)
- Split: train/test with test_size=0.2, random_state=42; stratified when possible
- Best model selected by test accuracy

## 4) Results
- Best model: `gradient_boosting`
- Metrics (from `reports/results/training_metrics.txt`):
  - Accuracy: 0.8532
  - Precision: 0.8522
  - Recall: 0.8532
  - F1-score: 0.8513
- Saved artifact: `models/best_model.joblib`
  - Includes: trained model, fitted scaler, feature column list, optional target encoder

## 5) Prediction
- Command executed:
  ```bash
  python -m src.pipeline.predict \
    --file data/raw/stress_data.csv \
    --output reports/results/predictions.csv \
    --head 5
  ```
- Feature alignment during prediction:
  - One-hot encoded categorical columns
  - Added any missing training-time columns with zeros
  - Scaled only columns known to the fitted scaler
- Output file: `reports/results/predictions.csv`
  - Added column: `predicted_stress_level`

## 6) How to Reproduce
- Train (auto-detect CSV in `data/raw`, pass target to override):
  ```bash
  python -m src.pipeline.train_pipeline --file stress_data.csv --target "Stress Label"
  ```
- Predict on any CSV with the same schema:
  ```bash
  python -m src.pipeline.predict \
    --file path/to/your_new_file.csv \
    --output reports/results/your_new_predictions.csv \
    --head 10
  ```

## 7) Notes & Assumptions
- Target classes detected: ['High Perceived Stress', 'Low Stress', 'Moderate Stress']
- If your target column name changes, pass `--target` or update `src/utils/config.py` (TARGET_COLUMN)
- Profiling package is optional in this environment; manual EDA is supported in notebooks

## 8) File Manifest
- `models/best_model.joblib` — Saved best model and preprocessing artifacts
- `reports/results/training_metrics.txt` — Metrics summary for best model
- `reports/results/predictions.csv` — Predictions with `predicted_stress_level`

---
If you want, I can export a confusion matrix and per-class metrics into `reports/figures/` as well.
