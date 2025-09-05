import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset
df = pd.read_csv("datasetExample.csv")
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 2. Visual Detection of Outliers
# Boxplot for each numeric column
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.show()

# Pairplot to see multivariate outliers
sns.pairplot(df[numeric_cols])
plt.show()

# 3. Statistical Detection of Outliers
# Method 1: Z-Score
from scipy import stats

z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers_z = np.where(z_scores > 3)
print("\nZ-Score Method:")
print(f"Outliers found at positions: {outliers_z}")

# Method 2: IQR (Interquartile Range)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers_iqr = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\nOutliers in {col} using IQR:")
    print(outliers_iqr)

# 4. Handling Outliers 
# df_clean = df[(z_scores < 3).all(axis=1)]
# print("\nDataset after removing outliers:")
# print(df_clean.shape)
