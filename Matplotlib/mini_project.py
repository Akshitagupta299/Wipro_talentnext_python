# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset
data = pd.read_csv("Diabetes.csv")
print("Shape of dataset:", data.shape)
print(data.head())

# Step 3: Data Pre-processing
print("\n--- Dataset Info ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Statistical Summary ---")
print(data.describe())

# Replace zeros with NaN in medical features (where zero is biologically impossible)
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# Fill missing values with median
for col in cols_with_zero:
    data[col].fillna(data[col].median(), inplace=True)

# Step 4: Handle Categorical Data
# Outcome is categorical (0 = Non-diabetic, 1 = Diabetic)
print("\n--- Value Counts of Outcome ---")
print(data['Outcome'].value_counts())

sns.countplot(x="Outcome", data=data, palette="Set2")
plt.title("Distribution of Diabetes Outcome")
plt.show()

# Step 5: Univariate Analysis
num_cols = data.columns.drop("Outcome")

# Histograms for numerical features
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots for outlier detection
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col], color="lightblue")
    plt.title(f"Boxplot of {col}")
    plt.show()

# Step 6: Bi-variate Analysis
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplots
sns.pairplot(data, hue="Outcome", diag_kind="kde")
plt.show()

# Relationship of each feature with Outcome
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Outcome", y=col, data=data, palette="Set3")
    plt.title(f"{col} vs Outcome")
    plt.show()
