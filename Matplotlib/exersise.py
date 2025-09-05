# 1. Perform Exploratory Data Analysis for the dataset Mall_Customers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
mall = pd.read_csv("Mall_Customers.csv")
print("Mall_Customers Dataset Info:")
print(mall.info())
print(mall.describe())

# Univariate Analysis
plt.figure(figsize=(6,4))
sns.histplot(mall["Age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(mall["Annual Income (k$)"], bins=20, kde=True, color="green")
plt.title("Annual Income Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(mall["Spending Score (1-100)"], bins=20, kde=True, color="purple")
plt.title("Spending Score Distribution")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(6,4))
sns.boxplot(x="Gender", y="Age", data=mall, palette="Set2")
plt.title("Age vs Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", data=mall)
plt.title("Income vs Spending Score by Gender")
plt.show()

# 2. Perform Exploratory Data Analysis for the dataset salary_data
salary = pd.read_csv("Salary_Data.csv")
print("\nSalary Data Info:")
print(salary.info())
print(salary.describe())

# Univariate Analysis
plt.figure(figsize=(6,4))
sns.histplot(salary["YearsExperience"], bins=10, kde=True, color="orange")
plt.title("Years of Experience Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(salary["Salary"], bins=10, kde=True, color="blue")
plt.title("Salary Distribution")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(6,4))
sns.scatterplot(x="YearsExperience", y="Salary", data=salary, color="red")
plt.title("Salary vs Years of Experience")
plt.show()

sns.regplot(x="YearsExperience", y="Salary", data=salary, color="green")
plt.title("Regression Line: Salary vs Experience")
plt.show()

# 3. Perform Exploratory Data Analysis for the dataset Social_Network_Ads
social = pd.read_csv("Social_Network_Ads.csv")
print("\nSocial Network Ads Info:")
print(social.info())
print(social.describe())

# Univariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=social, palette="Set2")
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(social["Age"], bins=10, kde=True, color="blue")
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(social["EstimatedSalary"], bins=10, kde=True, color="red")
plt.title("Estimated Salary Distribution")
plt.show()

# Bivariate Analysis
plt.figure(figsize=(6,4))
sns.boxplot(x="Purchased", y="Age", data=social, palette="coolwarm")
plt.title("Age vs Purchased")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Purchased", y="EstimatedSalary", data=social, palette="coolwarm")
plt.title("Salary vs Purchased")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="Age", y="EstimatedSalary", hue="Purchased", data=social, palette="Set1")
plt.title("Age vs Salary (Purchased)")
plt.show()
