import pandas as pd
import joblib
from sklearn.metrics import classification_report
from preprocessing import clean_text

df = pd.read_csv("data/raw/resumes.csv")
df['cleaned_text'] = df['text'].apply(clean_text)

embedder = joblib.load("models/embedder.pkl")
model = joblib.load("models/classifier.pkl")

X = embedder.encode(df['cleaned_text'].tolist())
y_true = df['label']

y_pred = model.predict(X)
print(classification_report(y_true, y_pred))