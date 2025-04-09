#!/usr/bin/env python3
"""
This script handles the preprocessing of MIMIC-IV data for my clinical pathway prediction project.
I wrote it to clean and prepare the data before training the machine learning model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the paths for my MIMIC-IV data files."""
        self.data_dir = Path('data')
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
    def load_data(self) -> pd.DataFrame:
        """Load and combine MIMIC-IV data files."""
        try:
            # Load patient data from admissions table
            admissions = pd.read_csv(self.raw_dir / 'admissions.csv')
            
            # Load patient demographics
            patients = pd.read_csv(self.raw_dir / 'patients.csv')
            
            # Load lab results
            labevents = pd.read_csv(self.raw_dir / 'labevents.csv')
            
            # Load diagnoses
            diagnoses = pd.read_csv(self.raw_dir / 'diagnoses_icd.csv')
            
            # Merge data starting with admissions
            df = admissions.merge(
                patients,
                on='subject_id',
                how='left'
            ).merge(
                diagnoses,
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            
            # Aggregate lab results by patient and admission
            lab_agg = labevents.groupby(['subject_id', 'hadm_id']).agg({
                'valuenum': ['mean', 'std', 'count']
            }).reset_index()
            
            # Merge with lab results
            df = df.merge(
                lab_agg,
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            
            logger.info(f"I loaded {len(df)} patient admissions")
            return df
            
        except Exception as e:
            logger.error(f"Oops! Something went wrong loading the MIMIC-IV data: {e}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up the MIMIC-IV data by handling missing values and categorical variables."""
        try:
            # Handle missing values
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            # Convert categorical variables to numeric
            df = pd.get_dummies(df, columns=categorical_cols)
            
            # Create target variable (e.g., readmission within 30 days)
            df['target'] = (df['readmission'] == 1).astype(int)
            
            # Remove any remaining non-numeric columns
            df = df.select_dtypes(include=[np.number])
            
            logger.info("Finished cleaning up the MIMIC-IV data")
            return df
            
        except Exception as e:
            logger.error(f"Had trouble cleaning the data: {e}")
            raise
            
    def save_data(self, df: pd.DataFrame):
        """Save the cleaned data for later use."""
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.processed_dir / 'processed_data.csv', index=False)
            logger.info("Saved the cleaned data successfully")
            
        except Exception as e:
            logger.error(f"Couldn't save the data: {e}")
            raise

def main():
    """This is where everything comes together."""
    try:
        # Start the data preprocessing
        preprocessor = DataPreprocessor()
        
        # Load and clean the MIMIC-IV data
        df = preprocessor.load_data()
        df_clean = preprocessor.clean_data(df)
        
        # Save the results
        preprocessor.save_data(df_clean)
        
    except Exception as e:
        logger.error(f"Something went wrong: {e}")
        raise

if __name__ == "__main__":
    main() 