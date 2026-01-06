import joblib
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text
from src import embeddings

# Register 'embeddings' as an alias for 'src.embeddings' so pickle can find the class
sys.modules['embeddings'] = embeddings

model = joblib.load("models/classifier.pkl")
embedder = joblib.load("models/embedder.pkl")

def predict_resume(text: str) -> str:
    cleaned_text = clean_text(text)
    emb = embedder.encode([cleaned_text])
    prediction = model.predict(emb)[0]
    prob = max(model.predict_proba(emb)[0])
    return prediction, prob