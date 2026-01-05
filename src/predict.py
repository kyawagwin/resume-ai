import joblib
from src.preprocessing import clean_text

model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_resume(text: str) -> str:
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    prob = max(model.predict_proba(vectorized_text)[0])
    return prediction, prob