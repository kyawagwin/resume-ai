from fastapi import FastAPI
from pydantic import BaseModel
from src.search import semantic_search

app = FastAPI(title="Smart Resume Classifier")

class ResumeRequest(BaseModel):
    text: str

@app.post("/predict")
def classify_resume(request: ResumeRequest, top_k: int = 5):
    results = semantic_search(request.text, top_k=top_k)
    return {"results": results}