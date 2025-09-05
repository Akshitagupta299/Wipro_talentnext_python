import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
data = pd.read_csv("melb_data.csv")
print("Shape before cleaning:", data.shape)
print(data.head())

# 2. Handle inappropriate data
# Remove duplicates
data = data.drop_duplicates()

# Drop irrelevant columns (you can adjust based on use-case)
drop_cols = ['Address', 'Date', 'SellerG', 'CouncilArea', 'Regionname']
data = data.drop(columns=drop_cols, errors='ignore')

# Remove rows with Price <= 0
data = data[data['Price'] > 0]

# 3. Handle missing data
for col in data.columns:
    if data[col].dtype == "object":   # categorical
        data[col] = data[col].fillna(data[col].mode()[0])
    else:                             # numerical
        data[col] = data[col].fillna(data[col].median())

# 4. Handle categorical data
label_enc = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_enc.fit_transform(data[col])

print("Shape after cleaning:", data.shape)
print(data.info())
print(data.head())
