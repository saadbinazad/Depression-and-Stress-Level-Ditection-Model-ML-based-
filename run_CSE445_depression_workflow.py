#!/usr/bin/env python3
"""
CSE445 Depression Level Prediction Workflow
============================================

This script performs automated depression level prediction analysis on the CSE445_Dataset.csv
using machine learning techniques similar to the existing stress analysis workflow.

Author: AI Assistant
Date: August 2025
"""

import os
import json
import time
import warnings
import datetime as dt
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class CSE445DepressionAnalyzer:
    """CSE445 Depression Level Prediction Analyzer"""
    
    def __init__(self, data_path='data/raw/CSE445_Dataset.csv', target='Depression_Level'):
        self.data_path = Path(data_path)
        self.target = target
        self.model_dir = Path('models')
        self.results_dir = Path('reports/results')
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize variables
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        
        print("✓ CSE445 Depression Analyzer initialized")
        print(f"Data path: {self.data_path}")
        print(f"Target: {self.target}")
    
    def load_data(self):
        """Load and validate the dataset"""
        print("\n=== LOADING DATA ===")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded successfully")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        
        # Validate target
        if self.target not in self.df.columns:
            available_cols = [col for col in self.df.columns if 'depression' in col.lower()]
            if available_cols:
                print(f"Target '{self.target}' not found. Available depression columns: {available_cols}")
                # Try to find a suitable target
                if 'Depression_Level' in self.df.columns:
                    self.target = 'Depression_Level'
                elif 'depression_level' in self.df.columns:
                    self.target = 'depression_level'
                else:
                    self.target = available_cols[0]
                print(f"Using target: {self.target}")
            else:
                raise KeyError(f"No depression-related columns found in {list(self.df.columns)}")
        
        print(f"✓ Target variable '{self.target}' found")
        print(f"Target classes: {self.df[self.target].unique()}")
        print(f"Target distribution:")
        print(self.df[self.target].value_counts())
        
        return self
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n=== PREPROCESSING DATA ===")
        
        # Clean string columns
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype(str).str.strip().replace({
                'nan': np.nan, 
                'None': np.nan, 
                '': np.nan
            })
        
        print("✓ String columns cleaned")
        
        # Check missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values per column:")
            print(missing_counts[missing_counts > 0])
        else:
            print("No missing values found")
        
        # Prepare features and target
        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]
        
        # Remove rows with missing target
        if self.y.isnull().sum() > 0:
            print(f"Removing {self.y.isnull().sum()} rows with missing target")
            valid_idx = ~self.y.isnull()
            self.X = self.X[valid_idx]
            self.y = self.y[valid_idx]
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        # Split features by type
        self.num_cols = []
        self.cat_cols = []
        
        for col in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[col]):
                self.num_cols.append(col)
            else:
                self.cat_cols.append(col)
        
        print(f"Numerical columns: {len(self.num_cols)}")
        print(f"Categorical columns: {len(self.cat_cols)}")
        
        return self
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        print("\n=== CREATING PREPROCESSOR ===")
        
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
        if self.num_cols:
            transformers.append(('num', num_pipeline, self.num_cols))
        if self.cat_cols:
            transformers.append(('cat', cat_pipeline, self.cat_cols))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        print("✓ Preprocessor created")
        return self
    
    def split_data(self):
        """Split data into train and test sets"""
        print("\n=== SPLITTING DATA ===")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_STATE, stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training target distribution:")
        print(self.y_train.value_counts(normalize=True))
        
        return self
    
    def train_models(self):
        """Train multiple models"""
        print("\n=== TRAINING MODELS ===")
        
        # Define models
        models = {
            'Dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'SVM': SVC(random_state=RANDOM_STATE, probability=True)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'training_time': training_time
            }
            
            self.trained_models[name] = pipeline
            
            print(f"✓ {name} completed in {training_time:.2f}s")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # Find best model
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        self.best_model_name = results_df.index[0]
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"Best F1-Score: {results_df.loc[self.best_model_name, 'f1_score']:.4f}")
        
        return self
    
    def evaluate_models(self):
        """Evaluate and visualize model performance"""
        print("\n=== EVALUATING MODELS ===")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("DEPRESSION MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        print(results_df)
        
        # Detailed evaluation of best model
        print(f"\n=== DETAILED EVALUATION: {self.best_model_name} ===")
        
        y_pred_best = self.best_model.predict(self.X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_best))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_best)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=np.unique(self.y), yticklabels=np.unique(self.y))
        plt.title(f'Confusion Matrix - {self.best_model_name} (Depression)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        confusion_matrix_path = self.results_dir / 'CSE445_depression_confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved: {confusion_matrix_path}")
        
        return self
    
    def cross_validate(self):
        """Perform cross-validation analysis"""
        print("\n=== CROSS-VALIDATION ANALYSIS ===")
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        top_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
        
        cv_results = {}
        
        for model_name in top_models:
            if model_name in self.trained_models:
                print(f"Cross-validating {model_name}...")
                model = self.trained_models[model_name]
                
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
                
                cv_results[model_name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def save_results(self):
        """Save models and results"""
        print("\n=== SAVING RESULTS ===")
        
        # Save best model
        model_filename = f'best_CSE445_depression_model.joblib'
        model_path = self.model_dir / model_filename
        joblib.dump(self.best_model, model_path)
        print(f"✓ Best model saved: {model_path}")
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_path = self.results_dir / 'CSE445_depression_model_metrics.csv'
        results_df.to_csv(results_path)
        print(f"✓ Results saved: {results_path}")
        
        # Save predictions
        y_pred_best = self.best_model.predict(self.X_test)
        predictions_df = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted': y_pred_best,
            'correct': self.y_test.values == y_pred_best
        })
        predictions_path = self.results_dir / 'CSE445_depression_predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved: {predictions_path}")
        
        return self
    
    def generate_summary(self):
        """Generate final summary"""
        print('\n' + '='*60)
        print('CSE445 DEPRESSION LEVEL PREDICTION - FINAL SUMMARY')
        print('='*60)
        print(f'Dataset: {self.data_path.name}')
        print(f'Samples: {len(self.df):,}')
        print(f'Features: {self.X.shape[1]}')
        print(f'Target: {self.target}')
        print(f'Classes: {len(self.y.unique())} - {list(self.y.unique())}')
        
        print(f'\nClass Distribution:')
        for class_name, count in self.y.value_counts().items():
            percentage = (count / len(self.y)) * 100
            print(f'  {class_name}: {count} ({percentage:.1f}%)')
        
        print(f'\nBest Model: {self.best_model_name}')
        print(f'Test Accuracy: {self.results[self.best_model_name]["accuracy"]:.4f}')
        print(f'Test F1-Score: {self.results[self.best_model_name]["f1_score"]:.4f}')
        
        print('\n✅ Depression analysis completed successfully!')
        print('='*60)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting CSE445 Depression Level Prediction Analysis...")
        print(f"Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Run the complete pipeline
            (self.load_data()
                 .preprocess_data()
                 .create_preprocessor()
                 .split_data()
                 .train_models()
                 .evaluate_models()
                 .cross_validate()
                 .save_results()
                 .generate_summary())
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Total analysis time: {total_time:.2f} seconds")
            print("🎉 Analysis completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            raise e
        
        return self


def main():
    """Main execution function"""
    print("=" * 80)
    print("CSE445 DEPRESSION LEVEL PREDICTION ANALYSIS")
    print("=" * 80)
    
    try:
        # Initialize and run analyzer
        analyzer = CSE445DepressionAnalyzer()
        analyzer.run_full_analysis()
        
        return analyzer
        
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Set up matplotlib for non-interactive backend
    plt.style.use('default')
    
    # Run the analysis
    analyzer = main()
    
    if analyzer:
        print("\n🔍 Analysis object available for further exploration")
        print("Available attributes:", [attr for attr in dir(analyzer) if not attr.startswith('_')])
