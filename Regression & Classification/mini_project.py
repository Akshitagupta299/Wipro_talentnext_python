### 1. Sales Prediction Use Case
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Load the data
df = pd.read_csv("Advertising.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())

# 2. Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicates if any
df = df.drop_duplicates()

# 3. Handle Categorical Data (if any)
# In Advertising dataset, usually all features are numeric.
# But if dataset has categorical columns, encode them:
# df = pd.get_dummies(df, drop_first=True)

# 4. Exploratory Data Analysis (EDA)

# Pairplot
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Relationship with Sales
for col in ['TV', 'Radio', 'Newspaper']:
    plt.figure(figsize=(5,4))
    sns.scatterplot(x=df[col], y=df['Sales'])
    plt.title(f"Sales vs {col}")
    plt.show()

# 5. Build the Model

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 6. Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Compare Actual vs Predicted
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nComparison of Actual vs Predicted:\n", results.head())

### 2. Diabetes Prediction
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("diabetes.csv")  # after downloading from Kaggle
print("Shape of dataset:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())

# 2. Data Preprocessing
print("\nMissing values:\n", df.isnull().sum())

# Replace zero values in certain columns with NaN
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

### 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x='Outcome', data=df, palette="Set2")
plt.title("Diabetes Outcome Distribution")
plt.show()

sns.pairplot(df, hue="Outcome", diag_kind="kde")
plt.show()

# 4. Train-Test Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# 6. K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

print("\n--- KNN Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# 7. Compare Results
print("\nModel Comparison:")
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
