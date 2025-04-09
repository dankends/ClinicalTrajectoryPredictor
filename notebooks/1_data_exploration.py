# %% [markdown]
# # MIMIC-IV Data Exploration
# 
# This notebook explores the MIMIC-IV dataset to understand its structure and contents.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
data_dir = Path('../data')
raw_dir = data_dir / 'raw'
processed_dir = data_dir / 'processed'

# %% [markdown]
# ## 1. Load and Explore Admissions Data

# %%
# Load admissions data
admissions = pd.read_csv(raw_dir / 'admissions.csv')

# Display basic information
print(f"Number of admissions: {len(admissions)}")
print("\nFirst few rows:")
admissions.head()

# %% [markdown]
# ## 2. Explore Patient Demographics

# %%
# Load patient data
patients = pd.read_csv(raw_dir / 'patients.csv')

# Display basic information
print(f"Number of patients: {len(patients)}")
print("\nFirst few rows:")
patients.head()

# %% [markdown]
# ## 3. Explore Lab Results

# %%
# Load lab events
labevents = pd.read_csv(raw_dir / 'labevents.csv')

# Display basic information
print(f"Number of lab events: {len(labevents)}")
print("\nFirst few rows:")
labevents.head()

# %% [markdown]
# ## 4. Explore Diagnoses

# %%
# Load diagnoses
diagnoses = pd.read_csv(raw_dir / 'diagnoses_icd.csv')

# Display basic information
print(f"Number of diagnoses: {len(diagnoses)}")
print("\nFirst few rows:")
diagnoses.head()

# %% [markdown]
# ## 5. Data Quality Checks

# %%
# Check for missing values
print("Missing values in admissions:")
admissions.isnull().sum()

print("\nMissing values in patients:")
patients.isnull().sum()

print("\nMissing values in lab events:")
labevents.isnull().sum()

# %% [markdown]
# ## 6. Initial Insights

# %%
# Based on this exploratory analysis, we can see:
# 
# 1. The size and scope of our dataset
# 2. The distribution of patient demographics
# 3. Common lab tests and diagnoses
# 4. Areas where data quality might need attention
# 
# Next steps:
# 1. Clean and preprocess the data
# 2. Create features for the prediction model
# 3. Train and evaluate the model 