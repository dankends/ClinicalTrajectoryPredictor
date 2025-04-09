# Clinical Pathway Prediction Project

This project was developed as part of my coursework in Machine Learning for Healthcare. It uses MIMIC-IV data to predict optimal clinical pathways for patients using machine learning techniques.

## Project Description

I created this project to explore how machine learning can be applied to healthcare data as a project for my course in university. The main goals were to:
- Learn how to preprocess and analyze medical data from MIMIC-IV
- Implement different machine learning models
- Understand model interpretability using SHAP values
- Create visualizations to explain model predictions

## Data Source

This project uses MIMIC-IV (Medical Information Mart for Intensive Care IV), a publicly available database developed by MIT and Beth Israel Deaconess Medical Center. MIMIC-IV contains de-identified health data associated with patients who stayed in critical care units.

To use this project, you'll need to:
1. Complete the required training for MIMIC-IV data access
2. Apply for access at https://mimic.mit.edu/iv/
3. Download the necessary tables and place them in the `data/raw` directory

## Project Structure

I organized the project into the following structure:
```
clinical-pathway-prediction/
├── data/
│   ├── raw/              # MIMIC-IV data files
│   └── processed/        # Cleaned and processed data
├── notebooks/
│   ├── 1_data_exploration.ipynb        # Data analysis notebook
│   ├── 2_feature_engineering.ipynb     # Feature creation notebook
│   └── 3_model_training.ipynb          # Model development notebook
├── src/
│   ├── data_preprocessing.py           # Script for cleaning data
│   ├── model_training.py               # Script for training models
│   └── visualization.py                # Script for creating plots
└── requirements.txt                    # Python package requirements
```

## How to Run the Project

1. First, set up a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the scripts in order:
```bash
# Clean and prepare the data
python src/data_preprocessing.py

# Train the machine learning model
python src/model_training.py

# Generate visualizations
python src/visualization.py
```

## Course Project Outcomes

Through this course project, I gained hands-on experience with:
- Working with MIMIC-IV healthcare data
- Data preprocessing and feature engineering
- Building and evaluating machine learning models
- Using SHAP values to explain model predictions
- Creating informative visualizations

## Technical Requirements

The project uses:
- Python 3.8+
- pandas for data manipulation
- scikit-learn for machine learning
- xgboost for the prediction model
- shap for model interpretability
- matplotlib and seaborn for visualization