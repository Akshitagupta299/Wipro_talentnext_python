# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Step 2: Load Dataset
data = pd.read_csv("melb_data.csv")
print("Shape of dataset:", data.shape)

# Preview dataset
print(data.head())

# Step 3: Dataset Overview
print("\n--- Dataset Info ---")
print(data.info())

print("\n--- Statistical Summary ---")
print(data.describe(include='all'))   # includes categorical

# Step 4: Handle Inappropriate Data
# Drop duplicates
data = data.drop_duplicates()

# Remove irrelevant columns (not useful for prediction)
drop_cols = ['Address', 'Date', 'SellerG', 'CouncilArea', 'Regionname']
data = data.drop(columns=drop_cols, errors='ignore')

# Ensure target variable (Price) is valid
data = data[data['Price'] > 0]

# Step 5: Handle Missing Values
missing_values = data.isnull().sum()
print("\n--- Missing Values ---")
print(missing_values[missing_values > 0])

# Numerical features → Median
num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

# Categorical features → Mode
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Step 6: Outlier Treatment (Statistical Perspective)
# Using IQR Method
for col in ['Price', 'Landsize', 'BuildingArea']:
    if col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 7: Handle Categorical Data
label_enc = LabelEncoder()
for col in cat_cols:
    data[col] = label_enc.fit_transform(data[col])

# Step 8: Correlation Analysis
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Step 9: Final Dataset
print("\nFinal Shape:", data.shape)
print(data.head())
