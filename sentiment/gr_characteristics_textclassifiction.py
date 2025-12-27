# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\MSI\Downloads\sentiment\gr.csv")  

print(dataset.head())
print("Total Rows:", len(dataset))
print(dataset.isnull().sum())

# ----------------------------
# Cleaning the texts
# ----------------------------

import re
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

corpus = []

# Replace 'Characteristic' with the correct column name from your CSV
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Characteristic'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print("Sample cleaned text:", corpus[0])

# ----------------------------
# Bag of Words Model
# ----------------------------

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

# ❗ Since your CSV has **no labels**, we create dummy labels (all 0)
y = np.zeros(len(dataset))

# ----------------------------
# Train-Test Split
# ----------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ----------------------------
# Classifier (Logistic Regression)
# ----------------------------

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
