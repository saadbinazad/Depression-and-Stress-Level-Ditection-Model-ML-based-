"""
Configuration settings for the stress prediction project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data settings
TARGET_COLUMN = "stress_level"  # Update this with your actual target column name
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model settings
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'lbfgs']
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
}

# Feature engineering settings
FEATURE_SELECTION_METHODS = [
    'correlation_analysis',
    'univariate_selection',
    'recursive_feature_elimination',
    'feature_importance_selection'
]

# Visualization settings
FIGURE_SIZE = (10, 8)
DPI = 300
SAVE_FORMAT = 'png'

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data preprocessing settings
MISSING_VALUE_STRATEGY = 'mean'  # Options: 'mean', 'median', 'mode', 'drop'
CORRELATION_THRESHOLD = 0.8
VARIANCE_THRESHOLD = 0.01

# File extensions
ALLOWED_DATA_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.json']

def create_directories():
    """Create project directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RESULTS_DIR, NOTEBOOKS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
def get_model_save_path(model_name: str) -> Path:
    """Get the path to save a model."""
    return MODELS_DIR / f"{model_name}_model.joblib"
    
def get_figure_save_path(figure_name: str) -> Path:
    """Get the path to save a figure."""
    return FIGURES_DIR / f"{figure_name}.{SAVE_FORMAT}"
    
def get_results_save_path(results_name: str) -> Path:
    """Get the path to save results."""
    return RESULTS_DIR / f"{results_name}.txt"
