import faiss
import joblib
from src.embeddings import ResumeEmbedder
from src.preprocessing import clean_text

INDEX_PATH = "models/resume.index"
META_PATH = "models/resume_meta.pkl"

index = faiss.read_index(INDEX_PATH)
meta = joblib.load(META_PATH)
embedder = ResumeEmbedder()

def semantic_search(query: str, top_k=5):
    query = clean_text(query)
    q_emb = embedder.encode([query])

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "resume_text": meta.iloc[idx]["text"],
            "label": meta.iloc[idx]["label"],
            "score": float(score)
        })

    return results
