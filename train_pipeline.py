import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ✅ Import your clean TextPreprocessor
from preprocessing import TextPreprocessor

# Label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.fit(['ham', 'spam'])

# Build pipeline
pipeline = Pipeline([
    ('preprocessing', TextPreprocessor()),
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Load data
import kagglehub
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
file_path = os.path.join(path, "spam.csv")

df = pd.read_csv(file_path, encoding='latin1')[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df['label_encoded'] = label_encoder.transform(df['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label_encoded'],
    test_size=0.2,
    stratify=df['label_encoded'],
    random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save pipeline
joblib.dump(pipeline, "spam_detection_pipeline.pkl")
print("✅ Pipeline saved successfully.")
