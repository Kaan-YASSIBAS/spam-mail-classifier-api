from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "spam_dataset.csv"
MODEL_PATH = BASE_DIR / "app" / "model" / "spam_classifier_model.pkl"

df = pd.read_csv(DATA_PATH)

print("=== DATASET ===")
print(df)

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label="spam")
recall = recall_score(y_test, predictions, pos_label="spam")
f1 = f1_score(y_test, predictions, pos_label="spam")
cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

print("\n=== PREDICTIONS ===")
print("Predictions:", predictions)
print("Real values:", y_test.values)

print("\n=== EVALUATION ===")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\n=== CONFUSION MATRIX ===")
print(cm)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")