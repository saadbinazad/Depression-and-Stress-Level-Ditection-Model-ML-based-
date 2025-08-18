"""
Data preprocessing utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing data for machine learning models."""
    
    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        df_processed = df.copy()
        
        if strategy == 'drop':
            # Drop rows with any missing values
            df_processed = df_processed.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df_processed.shape}")
        else:
            # Impute missing values
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            categorical_columns = df_processed.select_dtypes(exclude=[np.number]).columns
            
            # Handle numeric columns
            if len(numeric_columns) > 0:
                if strategy in ['mean', 'median']:
                    imputer = SimpleImputer(strategy=strategy)
                    df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
                    self.imputers['numeric'] = imputer
                    
            # Handle categorical columns
            if len(categorical_columns) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_columns] = imputer.fit_transform(df_processed[categorical_columns])
                self.imputers['categorical'] = imputer
                
            logger.info(f"Imputed missing values using {strategy} strategy")
            
        return df_processed
        
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_col (str): Target column name (will be label encoded)
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(exclude=[np.number]).columns
        
        for col in categorical_columns:
            if col == target_col:
                # Label encode target variable
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.encoders[col] = le
                logger.info(f"Label encoded target column: {col}")
            else:
                # One-hot encode other categorical variables
                if df_processed[col].nunique() <= 10:  # Only if not too many categories
                    df_encoded = pd.get_dummies(df_processed[col], prefix=col)
                    df_processed = df_processed.drop(col, axis=1)
                    df_processed = pd.concat([df_processed, df_encoded], axis=1)
                    logger.info(f"One-hot encoded column: {col}")
                else:
                    # Label encode if too many categories
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.encoders[col] = le
                    logger.info(f"Label encoded column: {col} (too many categories for one-hot)")
                    
        return df_processed
        
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Scaled training and test features
        """
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
            self.scalers['features'] = scaler
            
            if X_test is not None:
                X_test_scaled = X_test.copy()
                X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
                return X_train_scaled, X_test_scaled
            else:
                return X_train_scaled, None
        else:
            logger.warning("No numerical columns found for scaling")
            return X_train, X_test
            
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset without duplicates
        """
        initial_shape = df.shape
        df_processed = df.drop_duplicates()
        final_shape = df_processed.shape
        
        logger.info(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows")
        return df_processed
        
    def remove_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with low variance.
        
        Args:
            df (pd.DataFrame): Input dataset
            threshold (float): Variance threshold
            
        Returns:
            pd.DataFrame: Dataset without low variance features
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            variances = df[numeric_columns].var()
            low_variance_cols = variances[variances < threshold].index
            
            if len(low_variance_cols) > 0:
                df_processed = df.drop(columns=low_variance_cols)
                logger.info(f"Removed {len(low_variance_cols)} low variance features: {list(low_variance_cols)}")
                return df_processed
            else:
                logger.info("No low variance features found")
                return df
        else:
            logger.warning("No numerical columns found for variance analysis")
            return df
