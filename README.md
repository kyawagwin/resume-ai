## Resume AI Classifier

Smart resume intelligence with two modes:

- **Semantic search** over resumes using FAISS and Sentence Transformers.
- **Logistic Regression classifier** trained on the same embeddings for label prediction.

The FastAPI endpoint returns top-k similar resumes with labels and similarity scores, making it useful for both retrieval and classification.

## Features

- Sentence Transformer embeddings (`all-MiniLM-L6-v2`) with normalized vectors
- FAISS index for fast semantic search over stored resumes
- Logistic Regression baseline classifier (job labels)
- Text preprocessing with stopword removal and email/number cleanup
- FastAPI endpoint with configurable `top_k` results
- Dockerfile for containerized deployment

## Project Structure

```
resume-ai/
├── api/
│   └── main.py            # FastAPI app exposing /predict
├── data/
│   └── raw/
│       └── resumes.csv    # Source dataset (text, label)
├── models/
│   ├── classifier.pkl     # Trained Logistic Regression model
│   ├── embedder.pkl       # Saved SentenceTransformer wrapper
│   ├── resume.index       # FAISS index (built)
│   └── resume_meta.pkl    # Resume metadata aligned to the index
├── src/
│   ├── embeddings.py      # SentenceTransformer wrapper
│   ├── evaluate.py        # Classification report
│   ├── faiss_index.py     # Build FAISS index from CSV
│   ├── features.py        # (reserved for feature engineering)
│   ├── predict.py         # Local prediction helper
│   ├── preprocessing.py   # Text cleaning utilities
│   ├── search.py          # Semantic search over FAISS
│   └── train.py           # Train classifier + save artifacts
├── dockerfile             # Container build
├── environment.yml        # Conda environment (includes faiss-cpu)
└── requirements.txt       # Pip dependencies
```

## Setup

### Prerequisites

- Python 3.10 (matching `environment.yml`)
- Conda/Miniconda recommended

### Install

Conda (preferred; installs `faiss-cpu`):

```bash
conda env create -f environment.yml
conda activate resume-ai
```

Pip (ensure you also install FAISS):

```bash
pip install -r requirements.txt
pip install faiss-cpu
```

If you hit NumPy ABI issues, pin to a 1.x release:

```bash
pip install "numpy<2"
```

NLTK stopwords are downloaded at runtime, but you can preload them:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Data

Place your data in `data/raw/resumes.csv` with at least two columns:

- `text`: resume body
- `label`: target class / role

## Train

Train the classifier and save artifacts:

```bash
python src/train.py
```

Outputs: `models/classifier.pkl` and `models/embedder.pkl`.

## Build the FAISS Index

Create the search index and metadata store:

```bash
python src/faiss_index.py
```

Outputs: `models/resume.index` and `models/resume_meta.pkl`.

## Run the API

Start the FastAPI server (uses the FAISS index + embedder):

```bash
uvicorn api.main:app --reload
```

Docs: `http://127.0.0.1:8000/docs`

## API: Semantic Search / Predict

- **Endpoint**: `POST /predict?top_k=5`
- **Body**:

```json
{
  "text": "Experienced software engineer with 5 years in Python development..."
}
```

- **Response**: list of top-k matches with labels and similarity scores

```json
{
  "results": [
    {
      "resume_text": "...",
      "label": "Software Engineer",
      "score": 0.82
    }
  ]
}
```

## Evaluate

Run a classification report on the dataset:

```bash
python src/evaluate.py
```

## Docker

Build and run:

```bash
docker build -t resume-ai .
docker run -p 8000:8000 resume-ai
```

Ensure the container has access to your data/model artifacts or bake them into the image.

## Contributing

Issues and PRs are welcome.

## License

MIT
