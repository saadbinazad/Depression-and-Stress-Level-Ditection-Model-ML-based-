#!/usr/bin/env python3
"""
CSE445 Dataset - Stress Level Prediction Analysis
Automated script version of the CSE445_stress_analysis.ipynb notebook.
"""

import os, json, time, datetime as dt, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                            classification_report, confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import joblib

def main():
    print("="*60)
    print("CSE445 DATASET - STRESS LEVEL PREDICTION ANALYSIS")
    print("="*60)
    
    # 1. Configuration
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    
    DATA_PATH = Path('data/raw/CSE445_Dataset.csv')
    TARGET = 'Stress_Level'
    MODEL_DIR = Path('models')
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR = Path('reports/results')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print('✓ Configuration ready.')
    print(f'Target variable: {TARGET}')
    print(f'Data path: {DATA_PATH}')
    
    # 2. Data Loading
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f'✓ Dataset loaded successfully')
    print(f'Shape: {df.shape}')
    print(f'Columns: {len(df.columns)}')
    
    # Check target variable
    if TARGET not in df.columns:
        raise KeyError(f'Target {TARGET} not found. Available columns: {list(df.columns)[:10]}')
    
    print(f'✓ Target variable "{TARGET}" found')
    print(f'Target classes: {df[TARGET].unique()}')
    print(f'Target distribution:')
    print(df[TARGET].value_counts())
    
    # 3. Data Cleaning
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().replace({
            'nan': np.nan, 
            'None': np.nan, 
            '': np.nan
        })
    
    print('✓ String columns cleaned')
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f'Missing values per column:')
        print(missing_counts[missing_counts > 0])
    else:
        print('No missing values found')
    
    # 4. Feature and Target Preparation
    def split_num_cat(data, target):
        nums, cats = [], []
        for col in data.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(data[col]):
                nums.append(col)
            else:
                cats.append(col)
        return nums, cats
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')
    
    # Handle missing targets
    if y.isnull().sum() > 0:
        print(f'Warning: {y.isnull().sum()} missing values in target variable')
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        print(f'Removed missing targets. New shapes: X{X.shape}, y{y.shape}')
    
    # 5. Preprocessing Pipeline
    def create_preprocessor(X):
        nums, cats = [], []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                nums.append(col)
            else:
                cats.append(col)
        
        # Numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        transformers = []
        if nums:
            transformers.append(('num', num_pipeline, nums))
        if cats:
            transformers.append(('cat', cat_pipeline, cats))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor, nums, cats
    
    preprocessor, num_cols, cat_cols = create_preprocessor(X)
    print(f'✓ Preprocessor created')
    print(f'Numerical features: {len(num_cols)}')
    print(f'Categorical features: {len(cat_cols)}')
    
    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f'Training set: {X_train.shape}')
    print(f'Test set: {X_test.shape}')
    
    # 7. Model Training
    models = {
        'Dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(random_state=RANDOM_STATE, probability=True)
    }
    
    print(f'Models to train: {list(models.keys())}')
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f'\\nTraining {name}...')
        start_time = time.time()
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'training_time': training_time
        }
        
        trained_models[name] = pipeline
        
        print(f'✓ {name} completed in {training_time:.2f}s')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  F1-Score: {f1:.4f}')
    
    print('\\n✓ All models trained successfully!')
    
    # 8. Results Analysis
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print('\\n=== MODEL PERFORMANCE COMPARISON ===')
    print(results_df)
    
    # Find best model
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    print(f'\\n🏆 Best Model: {best_model_name}')
    print(f'Best F1-Score: {results_df.loc[best_model_name, "f1_score"]:.4f}')
    
    # 9. Detailed Evaluation
    print(f'\\n=== DETAILED EVALUATION: {best_model_name} ===')
    
    y_pred_best = best_model.predict(X_test)
    
    print('\\nClassification Report:')
    print(classification_report(y_test, y_pred_best))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / 'CSE445_stress_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Confusion matrix saved to {cm_path}')
    
    # 10. Save Results
    # Save the best model
    model_filename = f'best_CSE445_stress_model.joblib'
    model_path = MODEL_DIR / model_filename
    joblib.dump(best_model, model_path)
    print(f'✓ Best model saved: {model_path}')
    
    # Save results to CSV
    results_filename = 'CSE445_stress_model_metrics.csv'
    results_path = RESULTS_DIR / results_filename
    results_df.to_csv(results_path)
    print(f'✓ Results saved: {results_path}')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred_best,
        'correct': y_test.values == y_pred_best
    })
    predictions_path = RESULTS_DIR / 'CSE445_stress_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f'✓ Predictions saved: {predictions_path}')
    
    # 11. Performance Comparison Plot
    plt.figure(figsize=(12, 8))
    
    # Performance metrics comparison
    plt.subplot(2, 2, 1)
    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
    x_pos = np.arange(len(results_df))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x_pos + i*width, results_df[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos + width*1.5, results_df.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training time comparison
    plt.subplot(2, 2, 2)
    plt.bar(results_df.index, results_df['training_time'])
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_plot_path = RESULTS_DIR / 'CSE445_model_performance_comparison.png'
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Performance comparison plot saved: {performance_plot_path}')
    
    # 12. Final Summary
    print('\\n' + '='*60)
    print('CSE445 STRESS LEVEL PREDICTION - FINAL SUMMARY')
    print('='*60)
    print(f'Dataset: {DATA_PATH.name}')
    print(f'Samples: {len(df):,}')
    print(f'Features: {X.shape[1]}')
    print(f'Target: {TARGET}')
    print(f'Classes: {len(y.unique())}')
    print(f'\\nBest Model: {best_model_name}')
    print(f'Test Accuracy: {results_df.loc[best_model_name, "accuracy"]:.4f}')
    print(f'Test F1-Score: {results_df.loc[best_model_name, "f1_score"]:.4f}')
    print(f'\\nFiles Generated:')
    print(f'- Model: {model_path}')
    print(f'- Results: {results_path}')
    print(f'- Predictions: {predictions_path}')
    print(f'- Confusion Matrix: {cm_path}')
    print(f'- Performance Plot: {performance_plot_path}')
    print('\\n✅ Analysis completed successfully!')
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'results': results_df,
        'accuracy': results_df.loc[best_model_name, "accuracy"],
        'f1_score': results_df.loc[best_model_name, "f1_score"]
    }

if __name__ == "__main__":
    results = main()
