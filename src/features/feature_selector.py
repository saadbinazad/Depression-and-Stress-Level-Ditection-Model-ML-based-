"""
Feature selection utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    """Class for feature selection using various methods."""
    
    def __init__(self):
        """Initialize feature selector."""
        self.selected_features = None
        self.feature_scores = None
        
    def correlation_analysis(self, df: pd.DataFrame, target_col: str, threshold: float = 0.8) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target column name
            threshold (float): Correlation threshold for removal
            
        Returns:
            pd.DataFrame: Dataset with reduced correlation
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        
        # Return dataset without highly correlated features
        X_reduced = X.drop(columns=to_drop)
        return pd.concat([X_reduced, y], axis=1)
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 10, score_func=f_classif) -> list:
        """
        Select k best features using univariate statistical tests.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            k (int): Number of features to select
            score_func: Scoring function to use
            
        Returns:
            list: Selected feature names
        """
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        self.selected_features = selected_features
        self.feature_scores = dict(zip(selected_features, feature_scores))
        
        logger.info(f"Selected {len(selected_features)} features using univariate selection")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
        
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> list:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            n_features (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Perform RFE
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        
        self.selected_features = selected_features
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
        
    def feature_importance_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> list:
        """
        Select features based on Random Forest feature importance.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            threshold (float): Importance threshold for selection
            
        Returns:
            list: Selected feature names
        """
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = dict(zip(X.columns, importances))
        
        # Select features above threshold
        selected_features = [feature for feature, importance in feature_importance.items() if importance >= threshold]
        
        self.selected_features = selected_features
        self.feature_scores = feature_importance
        
        logger.info(f"Selected {len(selected_features)} features using importance threshold {threshold}")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
        
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.feature_scores is None:
            logger.warning("No feature scores available. Run a feature selection method first.")
            return pd.DataFrame()
            
        importance_df = pd.DataFrame(
            list(self.feature_scores.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        return importance_df
        
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> list:
        """
        Select features using mutual information.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            k (int): Number of features to select
            
        Returns:
            list: Selected feature names
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        self.selected_features = selected_features
        self.feature_scores = dict(zip(selected_features, feature_scores))
        
        logger.info(f"Selected {len(selected_features)} features using mutual information")
        logger.info(f"Selected features: {selected_features}")
        
        return selected_features
