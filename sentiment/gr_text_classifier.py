# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK stopwords (first run only)
nltk.download("stopwords")

# -----------------------------
# MODEL FUNCTIONS
# -----------------------------
def clean_text(text):
    """Cleans text: removes non-alphabet characters, lowercases, removes stopwords, stems words."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words("english"))]
    return ' '.join(text)

def train_model(csv_path):
    """Loads CSV, preprocesses text, and trains Logistic Regression model."""
    global classifier, cv
    dataset = pd.read_csv(csv_path)

    # Preprocess text
    corpus = [clean_text(text) for text in dataset["Characteristic"]]

    # Bag of Words
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()

    # Dummy labels (since GR CSV has no labels)
    y = np.zeros(len(dataset))

    # Train classifier
    classifier = LogisticRegression()
    classifier.fit(X, y)

    return "Model trained successfully!"

def predict_text(input_text):
    """Predicts output for user input text."""
    cleaned = clean_text(input_text)
    vector = cv.transform([cleaned]).toarray()
    prediction = classifier.predict(vector)
    return prediction[0]

# -----------------------------
# GUI APPLICATION
# -----------------------------
def upload_file():
    """Handles CSV file upload and model training."""
    global file_path
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if file_path:
        file_label.config(text=f"Selected: {file_path}")
        result = train_model(file_path)
        messagebox.showinfo("Success", result)

def classify_text():
    """Handles user text input and shows prediction."""
    user_text = input_box.get("1.0", "end-1c").strip()
    if not user_text:
        messagebox.showwarning("Warning", "Please enter text.")
        return
    result = predict_text(user_text)
    messagebox.showinfo("Prediction", f"Model Output: {result}")

# -----------------------------
# GUI SETUP
# -----------------------------
root = tk.Tk()
root.title("GR Text Classifier")
root.geometry("500x400")
root.config(bg="#e6f0ff")

# Title
title = tk.Label(root, text="Girls Representative Text Classifier",
                 font=("Arial", 16, "bold"), bg="#e6f0ff")
title.pack(pady=10)

# File selection
file_label = tk.Label(root, text="No file selected", bg="#e6f0ff", font=("Arial", 10))
file_label.pack(pady=5)

upload_btn = tk.Button(root, text="Upload CSV", command=upload_file,
                       bg="#4a90e2", fg="white", width=20)
upload_btn.pack(pady=10)

# Text input
input_label = tk.Label(root, text="Enter text to classify:", bg="#e6f0ff", font=("Arial", 12))
input_label.pack()

input_box = tk.Text(root, height=5, width=50)
input_box.pack(pady=5)

# Predict button
predict_btn = tk.Button(root, text="Classify Text", command=classify_text,
                        bg="#4a90e2", fg="white", width=20)
predict_btn.pack(pady=10)

root.mainloop()
