from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_resume

app = FastAPI(title="Smart Resume Classifier")

class ResumeRequest(BaseModel):
    text: str

@app.post("/predict")
def classify_resume(request: ResumeRequest):
    role, confidence = predict_resume(request.text)
    return {"predicted_role": role, "confidence": round(confidence, 2)}