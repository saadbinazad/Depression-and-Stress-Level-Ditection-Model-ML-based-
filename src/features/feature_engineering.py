"""
Feature engineering utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for creating new features from existing data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.new_features = []
        
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df (pd.DataFrame): Input dataset
            feature_pairs (list): List of tuples containing feature pairs
            
        Returns:
            pd.DataFrame: Dataset with interaction features
        """
        df_processed = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                interaction_name = f"{feature1}_x_{feature2}"
                df_processed[interaction_name] = df_processed[feature1] * df_processed[feature2]
                self.new_features.append(interaction_name)
                logger.info(f"Created interaction feature: {interaction_name}")
            else:
                logger.warning(f"Features {feature1} or {feature2} not found in dataset")
                
        return df_processed
        
    def create_polynomial_features(self, df: pd.DataFrame, features: list, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df (pd.DataFrame): Input dataset
            features (list): List of feature names to create polynomial features for
            degree (int): Degree of polynomial features
            
        Returns:
            pd.DataFrame: Dataset with polynomial features
        """
        df_processed = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    poly_name = f"{feature}_poly_{d}"
                    df_processed[poly_name] = df_processed[feature] ** d
                    self.new_features.append(poly_name)
                    logger.info(f"Created polynomial feature: {poly_name}")
            else:
                logger.warning(f"Feature {feature} not found in dataset")
                
        return df_processed
        
    def create_binned_features(self, df: pd.DataFrame, feature_bins: dict) -> pd.DataFrame:
        """
        Create binned categorical features from continuous variables.
        
        Args:
            df (pd.DataFrame): Input dataset
            feature_bins (dict): Dictionary with feature names as keys and bin edges as values
            
        Returns:
            pd.DataFrame: Dataset with binned features
        """
        df_processed = df.copy()
        
        for feature, bins in feature_bins.items():
            if feature in df.columns:
                binned_name = f"{feature}_binned"
                df_processed[binned_name] = pd.cut(df_processed[feature], bins=bins, include_lowest=True)
                self.new_features.append(binned_name)
                logger.info(f"Created binned feature: {binned_name}")
            else:
                logger.warning(f"Feature {feature} not found in dataset")
                
        return df_processed
        
    def create_ratio_features(self, df: pd.DataFrame, ratio_pairs: list) -> pd.DataFrame:
        """
        Create ratio features between specified feature pairs.
        
        Args:
            df (pd.DataFrame): Input dataset
            ratio_pairs (list): List of tuples containing feature pairs for ratios
            
        Returns:
            pd.DataFrame: Dataset with ratio features
        """
        df_processed = df.copy()
        
        for numerator, denominator in ratio_pairs:
            if numerator in df.columns and denominator in df.columns:
                ratio_name = f"{numerator}_over_{denominator}"
                # Avoid division by zero
                df_processed[ratio_name] = df_processed[numerator] / (df_processed[denominator] + 1e-8)
                self.new_features.append(ratio_name)
                logger.info(f"Created ratio feature: {ratio_name}")
            else:
                logger.warning(f"Features {numerator} or {denominator} not found in dataset")
                
        return df_processed
        
    def create_aggregate_features(self, df: pd.DataFrame, group_col: str, agg_features: list, agg_funcs: list) -> pd.DataFrame:
        """
        Create aggregate features grouped by a categorical column.
        
        Args:
            df (pd.DataFrame): Input dataset
            group_col (str): Column to group by
            agg_features (list): Features to aggregate
            agg_funcs (list): Aggregation functions to apply
            
        Returns:
            pd.DataFrame: Dataset with aggregate features
        """
        df_processed = df.copy()
        
        if group_col not in df.columns:
            logger.warning(f"Group column {group_col} not found in dataset")
            return df_processed
            
        for feature in agg_features:
            if feature in df.columns:
                for func in agg_funcs:
                    agg_name = f"{feature}_{func}_by_{group_col}"
                    agg_values = df_processed.groupby(group_col)[feature].transform(func)
                    df_processed[agg_name] = agg_values
                    self.new_features.append(agg_name)
                    logger.info(f"Created aggregate feature: {agg_name}")
            else:
                logger.warning(f"Feature {feature} not found in dataset")
                
        return df_processed
        
    def create_datetime_features(self, df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
        """
        Extract datetime features from datetime columns.
        
        Args:
            df (pd.DataFrame): Input dataset
            datetime_cols (list): List of datetime column names
            
        Returns:
            pd.DataFrame: Dataset with datetime features
        """
        df_processed = df.copy()
        
        for col in datetime_cols:
            if col in df.columns:
                # Convert to datetime if not already
                df_processed[col] = pd.to_datetime(df_processed[col])
                
                # Extract features
                df_processed[f"{col}_year"] = df_processed[col].dt.year
                df_processed[f"{col}_month"] = df_processed[col].dt.month
                df_processed[f"{col}_day"] = df_processed[col].dt.day
                df_processed[f"{col}_hour"] = df_processed[col].dt.hour
                df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek
                df_processed[f"{col}_is_weekend"] = (df_processed[col].dt.dayofweek >= 5).astype(int)
                
                new_features = [f"{col}_year", f"{col}_month", f"{col}_day", 
                              f"{col}_hour", f"{col}_dayofweek", f"{col}_is_weekend"]
                self.new_features.extend(new_features)
                
                logger.info(f"Created datetime features for: {col}")
            else:
                logger.warning(f"Datetime column {col} not found in dataset")
                
        return df_processed
        
    def get_new_features(self) -> list:
        """
        Get list of newly created features.
        
        Returns:
            list: List of new feature names
        """
        return self.new_features
        
    def reset_new_features(self):
        """Reset the list of new features."""
        self.new_features = []
