import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import clean_text
from embeddings import ResumeEmbedder

df = pd.read_csv("data/raw/resumes.csv")
df['cleaned_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

embedder = ResumeEmbedder()
X_train_vec = embedder.encode(X_train.tolist())

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, "models/classifier.pkl")
joblib.dump(embedder, "models/embedder.pkl")

print("Model and embedder saved successfully.")