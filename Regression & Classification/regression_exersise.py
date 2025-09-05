# 1. Predict the price of the car based on its features. Use appropriate evaluation metrics. cars.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("cars.csv")
print(df.head())
print(df.info())

# Encode categorical variables if present
df = pd.get_dummies(df, drop_first=True)

# Features & Target (assuming 'price' column exists)
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nCar Price Prediction Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# 2. Create a model that can predict the profit based on its features. Use appropriate evaluation metrics. The Dataset can be downloaded from kaggle.com Dataset: 50 startups.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("50_startups.csv")
print(df.head())
print(df.info())

# Encode categorical variable (State)
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop("Profit", axis=1)
y = df["Profit"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nStartup Profit Prediction Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# 3. Create a model that can predict the profit based on its features Use appropriate evaluation metrics. The Dataset can be downloaded from kaggle.com Dataset: Salary Data
# Load dataset
df = pd.read_csv("Salary_Data.csv")
print(df.head())
print(df.info())

# Features & Target
X = df[['YearsExperience']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nSalary Prediction Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Visualization
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Predicted Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.show()
