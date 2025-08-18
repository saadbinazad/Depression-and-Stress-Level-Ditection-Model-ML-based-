"""
Data loading utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and basic validation of datasets."""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize DataLoader with path to data directory.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = Path(data_path)
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file and perform basic validation.
        
        Args:
            filename (str): Name of the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
            
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Dictionary with basic dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return info
        
    def validate_target_column(self, df: pd.DataFrame, target_col: str) -> bool:
        """
        Validate that target column exists and has valid values.
        
        Args:
            df (pd.DataFrame): Dataset
            target_col (str): Name of target column
            
        Returns:
            bool: True if valid, raises exception otherwise
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
            
        if df[target_col].isnull().all():
            raise ValueError(f"Target column '{target_col}' contains only null values")
            
        logger.info(f"Target column '{target_col}' validation passed")
        return True
