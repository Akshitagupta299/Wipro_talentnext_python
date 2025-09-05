# 2.  Perform Data Preprocessing on melb data.csv dataset with statistical perspective. The dataset M can be downloaded from https://www.kaggle.com/datasets/gunjanpathak/melb-data?resource=download

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
data = pd.read_csv("melb_data.csv")
print("Initial Shape of dataset:", data.shape)

# Preview first 5 rows
print("\n--- First 5 rows ---")
print(data.head())

# Step 3: Statistical Overview
print("\n--- Dataset Info ---")
print(data.info())

print("\n--- Statistical Summary (Numerical) ---")
print(data.describe())

print("\n--- Statistical Summary (Categorical) ---")
print(data.describe(include=['object']))

# Check missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Step 4: Handle Inappropriate Data
# Drop duplicates
data = data.drop_duplicates()

# Drop irrelevant columns (not useful for prediction)
drop_cols = ['Address', 'Date', 'SellerG', 'CouncilArea', 'Regionname']
data = data.drop(columns=drop_cols, errors='ignore')

# Ensure target variable (Price) is valid
data = data[data['Price'] > 0]

# Step 5: Handle Missing Values
num_cols = data.select_dtypes(include=[np.number]).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Numerical → Median
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Categorical → Mode
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Step 6: Outlier Detection & Treatment
for col in ['Price', 'Landsize', 'BuildingArea']:
    if col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 7: Encode Categorical Data
label_enc = LabelEncoder()
for col in cat_cols:
    if col in data.columns:
        data[col] = label_enc.fit_transform(data[col])

# Step 8: Correlation Analysis
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Step 9: Distribution Check (Statistics Perspective)
for col in ['Price', 'Rooms', 'Distance', 'Landsize']:
    if col in data.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

# Step 10: Final Dataset
print("\nFinal Shape of dataset:", data.shape)
print(data.head())
