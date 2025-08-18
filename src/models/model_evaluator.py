"""
Model evaluation utilities for the stress prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import learning_curve
import logging
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for comprehensive model evaluation and visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize model evaluator.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots
        """
        self.figsize = figsize
        self.evaluation_results = {}
        
    def generate_classification_report(self, y_true: pd.Series, y_pred: pd.Series, 
                                     target_names: List[str] = None) -> Dict:
        """
        Generate detailed classification report.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (pd.Series): Predicted labels
            target_names (List[str]): Target class names
            
        Returns:
            Dict: Classification report as dictionary
        """
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_true, y_pred, target_names=target_names))
        
        return report
        
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series, 
                            labels: List[str] = None, normalize: bool = False,
                            save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (pd.Series): Predicted labels
            labels (List[str]): Class labels
            normalize (bool): Whether to normalize the confusion matrix
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', xticklabels=labels, yticklabels=labels)
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            
        return plt.gcf()
        
    def plot_roc_curves(self, models_proba: Dict[str, np.ndarray], y_true: pd.Series,
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_proba (Dict[str, np.ndarray]): Dictionary of model probabilities
            y_true (pd.Series): True binary labels
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: ROC curves plot
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, y_proba in models_proba.items():
            # For multiclass, use positive class probability
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
                
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
            
        return plt.gcf()
        
    def plot_precision_recall_curves(self, models_proba: Dict[str, np.ndarray], y_true: pd.Series,
                                    save_path: str = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models_proba (Dict[str, np.ndarray]): Dictionary of model probabilities
            y_true (pd.Series): True binary labels
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Precision-Recall curves plot
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, y_proba in models_proba.items():
            # For multiclass, use positive class probability
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
                
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
            
        return plt.gcf()
        
    def plot_learning_curves(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           model_name: str = "Model", cv: int = 5,
                           save_path: str = None) -> plt.Figure:
        """
        Plot learning curves for a model.
        
        Args:
            model: Trained model object
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model_name (str): Name of the model
            cv (int): Number of cross-validation folds
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Learning curves plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=self.figsize)
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
            
        return plt.gcf()
        
    def compare_models_performance(self, model_scores: Dict[str, Dict[str, float]],
                                 save_path: str = None) -> plt.Figure:
        """
        Create a bar plot comparing model performances.
        
        Args:
            model_scores (Dict[str, Dict[str, float]]): Model performance scores
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Model comparison plot
        """
        # Convert to DataFrame for easier plotting
        scores_df = pd.DataFrame(model_scores).T
        
        # Create subplots for each metric
        metrics = scores_df.columns
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 6))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            axes[i].bar(scores_df.index, scores_df[metric])
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].set_xticklabels(scores_df.index, rotation=45)
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
            
        return fig
        
    def generate_evaluation_summary(self, model_scores: Dict[str, Dict[str, float]],
                                  best_model_name: str) -> Dict:
        """
        Generate a comprehensive evaluation summary.
        
        Args:
            model_scores (Dict[str, Dict[str, float]]): Model performance scores
            best_model_name (str): Name of the best performing model
            
        Returns:
            Dict: Evaluation summary
        """
        summary = {
            'best_model': best_model_name,
            'best_model_scores': model_scores[best_model_name],
            'all_model_scores': model_scores,
            'model_ranking': sorted(model_scores.keys(), 
                                  key=lambda x: model_scores[x]['accuracy'], 
                                  reverse=True)
        }
        
        logger.info("Evaluation Summary:")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best Model Accuracy: {model_scores[best_model_name]['accuracy']:.4f}")
        logger.info(f"Model Ranking: {summary['model_ranking']}")
        
        self.evaluation_results = summary
        return summary
        
    def save_evaluation_report(self, filepath: str, summary: Dict = None):
        """
        Save evaluation report to a text file.
        
        Args:
            filepath (str): Path to save the report
            summary (Dict): Evaluation summary (uses stored if None)
        """
        if summary is None:
            summary = self.evaluation_results
            
        if not summary:
            logger.warning("No evaluation results to save")
            return
            
        with open(filepath, 'w') as f:
            f.write("STRESS LEVEL PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Best Model: {summary['best_model']}\n")
            f.write(f"Best Model Performance:\n")
            for metric, value in summary['best_model_scores'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("All Model Performances:\n")
            f.write("-" * 30 + "\n")
            for model_name in summary['model_ranking']:
                f.write(f"\n{model_name}:\n")
                for metric, value in summary['all_model_scores'][model_name].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                    
        logger.info(f"Evaluation report saved to {filepath}")
