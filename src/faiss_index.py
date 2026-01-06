import faiss
import joblib
import numpy as np
import pandas as pd
from preprocessing import clean_text
from embeddings import ResumeEmbedder

INDEX_PATH = "models/resume.index"
META_PATH = "models/resume_meta.pkl"

def build_faiss_index(csv_path="data/raw/resumes.csv"):
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["text"].apply(clean_text)

    embedder = ResumeEmbedder()
    embeddings = embedder.encode(df["clean_text"].tolist())

    # Convert to float32 numpy array
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    joblib.dump(df[["text", "label"]], META_PATH)

    print(f"✅ FAISS index built with {index.ntotal} resumes")

if __name__ == "__main__":
    build_faiss_index()
