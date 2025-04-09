#!/usr/bin/env python3
"""
This script creates visualizations to help me understand my model's predictions.
I use it to generate plots that show feature importance and SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import shap
import joblib

# I set up logging to track what's happening with my visualizations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self):
        """I set up where to find my data and where to save my plots."""
        self.data_dir = Path('data/processed')
        self.model_dir = Path('models')
        self.figures_dir = Path('figures')
        self.model = None
        self.shap_values = None
        
    def load_model_and_data(self):
        """I load my trained model and the data I need for visualization."""
        try:
            # I load the model I trained earlier
            self.model = joblib.load(self.model_dir / 'model.joblib')
            
            # I get the SHAP values I calculated
            self.shap_values = np.load(self.model_dir / 'shap_values.npy')
            
            # I load my processed data
            self.data = pd.read_csv(self.data_dir / 'processed_data.csv')
            self.X = self.data.drop('target', axis=1)
            
            logger.info("I loaded my model and data successfully")
            
        except Exception as e:
            logger.error(f"I had trouble loading my model and data: {e}")
            raise
            
    def plot_feature_importance(self):
        """I create a plot showing which features are most important in my model."""
        try:
            # I get the importance scores from my model
            importance = self.model.feature_importances_
            feature_names = self.X.columns
            
            # I create a DataFrame to make plotting easier
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # I create a bar plot of the top 10 features
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='importance',
                y='feature',
                data=importance_df.head(10)
            )
            plt.title('Top 10 Most Important Features in My Model')
            plt.tight_layout()
            
            # I save the plot
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.figures_dir / 'feature_importance.png')
            plt.close()
            
            logger.info("I created and saved the feature importance plot")
            
        except Exception as e:
            logger.error(f"I had trouble creating the feature importance plot: {e}")
            raise
            
    def plot_shap_summary(self):
        """I create a SHAP summary plot to understand feature importance."""
        try:
            # I create a summary plot using SHAP
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.shap_values,
                self.X,
                plot_type="bar",
                show=False
            )
            plt.title('SHAP Feature Importance in My Model')
            plt.tight_layout()
            
            # I save the plot
            plt.savefig(self.figures_dir / 'shap_summary.png')
            plt.close()
            
            logger.info("I created and saved the SHAP summary plot")
            
        except Exception as e:
            logger.error(f"I had trouble creating the SHAP summary plot: {e}")
            raise
            
    def plot_shap_dependence(self, feature: str):
        """I create a dependence plot to see how a specific feature affects predictions."""
        try:
            # I create a dependence plot for the specified feature
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                self.shap_values,
                self.X,
                show=False
            )
            plt.title(f'How {feature} Affects Predictions in My Model')
            plt.tight_layout()
            
            # I save the plot
            plt.savefig(self.figures_dir / f'shap_dependence_{feature}.png')
            plt.close()
            
            logger.info(f"I created and saved the SHAP dependence plot for {feature}")
            
        except Exception as e:
            logger.error(f"I had trouble creating the SHAP dependence plot: {e}")
            raise

def main():
    """This is where I create all my visualizations."""
    try:
        # I start the visualization process
        visualizer = Visualizer()
        
        # I load my model and data
        visualizer.load_model_and_data()
        
        # I create the plots
        visualizer.plot_feature_importance()
        visualizer.plot_shap_summary()
        
        # I create dependence plots for the top 3 features
        top_features = visualizer.X.columns[:3]
        for feature in top_features:
            visualizer.plot_shap_dependence(feature)
            
    except Exception as e:
        logger.error(f"Something went wrong while creating visualizations: {e}")
        raise

if __name__ == "__main__":
    main() 