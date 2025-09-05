# 1. Perform Text Preprocessing on SMSSpamCollection Dataset. The dataset can be downloaded from https://www.kaggle.com/datasets
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download stopwords if not already present
nltk.download('stopwords')

# Step 2: Load Dataset
data = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
print("Dataset shape:", data.shape)
print(data.head())

# Step 3: Convert Labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 4: Text Cleaning Function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = text.strip()
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords + apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply preprocessing
data['clean_message'] = data['message'].apply(preprocess_text)

# Step 5: Train-Test Split
X = data['clean_message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Vectorization (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nPreprocessing Complete ✅")
print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_test_tfidf shape:", X_test_tfidf.shape)
