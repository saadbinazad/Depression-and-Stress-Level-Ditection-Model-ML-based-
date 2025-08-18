"""
Model training utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize default models with basic hyperparameters.
        
        Returns:
            Dict[str, Any]: Dictionary of initialized models
        """
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                random_state=42,
                probability=True
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self.models
        
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Trained model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        logger.info(f"Training {model_name} model...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        logger.info(f"Successfully trained {model_name} model")
        return model
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
            
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
            
        logger.info(f"Trained {len(self.trained_models)} models")
        return self.trained_models
        
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        model = self.trained_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Store metrics
        self.model_scores[model_name] = metrics
        
        logger.info(f"Evaluated {model_name} model:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return metrics
        
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of all model metrics
        """
        for model_name in self.trained_models.keys():
            self.evaluate_model(model_name, X_test, y_test)
            
        return self.model_scores
        
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation for {model_name}:")
        logger.info(f"  Mean CV Score: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score'] * 2:.4f})")
        
        return cv_results
        
    def hyperparameter_tuning(self, model_name: str, param_grid: Dict, X_train: pd.DataFrame, 
                            y_train: pd.Series, cv: int = 5) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model
            param_grid (Dict): Parameter grid for tuning
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv (int): Number of cross-validation folds
            
        Returns:
            Best estimator from grid search
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
        
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
            
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Saved {model_name} model to {filepath}")
        
    def load_model(self, filepath: str, model_name: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to load the model from
            model_name (str): Name to assign to the loaded model
        """
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        logger.info(f"Loaded model from {filepath} as {model_name}")
        
    def get_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """
        Get the best performing model based on accuracy.
        
        Returns:
            Tuple[str, Any, Dict[str, float]]: Best model name, model object, and scores
        """
        if not self.model_scores:
            raise ValueError("No models have been evaluated yet")
            
        best_model_name = max(self.model_scores.keys(), 
                            key=lambda x: self.model_scores[x]['accuracy'])
        
        best_model = self.trained_models[best_model_name]
        best_scores = self.model_scores[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with accuracy: {best_scores['accuracy']:.4f}")
        
        return best_model_name, best_model, best_scores
