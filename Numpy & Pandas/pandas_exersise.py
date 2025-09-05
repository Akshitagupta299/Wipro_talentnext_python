# 3. Pandas-DataFrame
# Download the data set and rename to cars.csv
# Link: Dataset: https://www.kaggle.com/uciml/autompg-dataset/data?select=auto-mpg.csv or https://archive.ics.uci.edu/ml/datasets/Auto+MPG
# a. Import Pandas
# b. Import the Cars Dataset and store the Pandas DataFrame in the variable cars
# c. Inspect the first 10 Rows of the DataFrame cars
# d. Inspect the DataFrame cars by "printing" cars
# e. Inspect the last 5 Rows
# f. Get some meta information on our DataFrame!
# Importing and Inspecting your own Dataset
import pandas as pd

# a. Import Pandas (already done above)

# b. Import Cars Dataset (rename file to cars.csv after downloading)
cars = pd.read_csv("cars.csv")

# c. Inspect the first 10 rows
print("\nFirst 10 Rows of Cars Dataset:\n", cars.head(10))

# d. Inspect full DataFrame (printing structure)
print("\nDataFrame (Cars):\n", cars)

# e. Inspect last 5 rows
print("\nLast 5 Rows:\n", cars.tail(5))

# f. Get meta information
print("\nMeta Information:\n")
print(cars.info())

# Summary statistics
print("\nStatistical Summary:\n", cars.describe())

# 4. Download 50_startups dataset Link :https://www.kaggle.com/datasets/farhanmd29/50-startups
# a. Create DataFrame using Pandas
# b. Read the data from 50_startups.csv file and load the data into dataframe.
# c. Check the statistical summary.
# d. Check for corelation coefficient between dependent and independent variables.
import pandas as pd

# a & b. Create DataFrame from 50_startups.csv
startups = pd.read_csv("50_startups.csv")

# Inspect first few rows
print("\nFirst 5 Rows of 50_Startups Dataset:\n", startups.head())

# c. Statistical summary
print("\nStatistical Summary:\n", startups.describe())

# d. Correlation coefficient between dependent & independent variables
print("\nCorrelation Matrix:\n", startups.corr(numeric_only=True))
