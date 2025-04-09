#!/usr/bin/env python3
"""
This is my script for training the machine learning model for my clinical pathway prediction project.
I chose to use XGBoost because it's known to work well with medical data and provides good interpretability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import shap
import joblib

# I set up logging to help me track what's happening during training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """I set up the paths where my data is stored and where I'll save my model."""
        self.data_dir = Path('data/processed')
        self.model_dir = Path('models')
        self.model = None
        
    def load_data(self) -> tuple:
        """I load my cleaned data and split it into training and testing sets."""
        try:
            # I get the data I cleaned in the previous step
            df = pd.read_csv(self.data_dir / 'processed_data.csv')
            
            # I separate the features from what we're trying to predict
            X = df.drop('target', axis=1)
            y = df['target']
            
            # I split the data, keeping 20% for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42  # I use 42 to make sure I get the same split every time
            )
            
            logger.info("I've got my data ready for training")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"I had trouble loading the data: {e}")
            raise
            
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """I train my XGBoost model with parameters I found worked well in my experiments."""
        try:
            # I set up the model with parameters that gave me good results
            self.model = xgb.XGBClassifier(
                n_estimators=100,      # I use 100 trees
                max_depth=3,           # I limit tree depth to prevent overfitting
                learning_rate=0.1,     # I found this learning rate works well
                random_state=42        # I use 42 for reproducibility
            )
            
            # I train the model on my data
            self.model.fit(X_train, y_train)
            logger.info("I finished training my model")
            
        except Exception as e:
            logger.error(f"Something went wrong while I was training: {e}")
            raise
            
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """I check how well my model is performing using different metrics."""
        try:
            # I get predictions from my model
            y_pred = self.model.predict(X_test)
            
            # I calculate different metrics to see how good my model is
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            logger.info("I checked how well my model is doing")
            return metrics
            
        except Exception as e:
            logger.error(f"I couldn't evaluate the model: {e}")
            raise
            
    def calculate_shap_values(self, X: pd.DataFrame):
        """I use SHAP to understand which features are most important in my model's predictions."""
        try:
            # I set up SHAP to explain my model's decisions
            explainer = shap.TreeExplainer(self.model)
            
            # I calculate how each feature affects the predictions
            shap_values = explainer.shap_values(X)
            
            # I save these values so I can analyze them later
            self.model_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.model_dir / 'shap_values.npy', shap_values)
            
            logger.info("I calculated SHAP values to understand my model better")
            
        except Exception as e:
            logger.error(f"I had trouble with the SHAP values: {e}")
            raise
            
    def save_model(self):
        """I save my trained model so I can use it later."""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_dir / 'model.joblib')
            logger.info("I saved my model successfully")
            
        except Exception as e:
            logger.error(f"I couldn't save the model: {e}")
            raise

def main():
    """This is where I put everything together to train and evaluate my model."""
    try:
        # I start the training process
        trainer = ModelTrainer()
        
        # I get my data ready
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # I train my model
        trainer.train_model(X_train, y_train)
        
        # I check how well it's doing
        metrics = trainer.evaluate_model(X_test, y_test)
        logger.info(f"My model's performance: {metrics}")
        
        # I understand which features matter most
        trainer.calculate_shap_values(X_test)
        
        # I save everything
        trainer.save_model()
        
    except Exception as e:
        logger.error(f"Something went wrong: {e}")
        raise

if __name__ == "__main__":
    main() 