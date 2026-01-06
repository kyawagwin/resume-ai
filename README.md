# Resume AI Classifier

An intelligent resume classification system that uses machine learning to categorize resumes into different job roles. Built with FastAPI, scikit-learn, and Sentence Transformers, this project provides a REST API for automated resume screening with semantic embeddings.

## Features

- **Semantic Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2) for high-quality text representations
- **Text Preprocessing**: Advanced text cleaning and normalization for resume data
- **ML Classification**: Logistic Regression model trained on resume embeddings
- **REST API**: FastAPI-based API for real-time predictions
- **Confidence Scores**: Returns prediction confidence for each classification
- **Model Persistence**: Trained models saved for quick inference
- **Dockerized**: Ready for containerized deployment

## Project Structure

```
resume-ai/
├── api/
│   └── main.py              # FastAPI application
├── data/
│   └── raw/
│       └── resumes.csv      # Training dataset
├── models/
│   ├── classifier.pkl       # Trained model
│   └── embedder.pkl         # Sentence transformer embedder
├── src/
│   ├── train.py            # Model training script
│   ├── predict.py          # Prediction functions
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── embeddings.py       # Sentence transformer embeddings
│   ├── features.py         # Feature engineering
│   └── evaluate.py         # Model evaluation
├── dockerfile              # Docker configuration
├── environment.yml         # Conda environment
└── requirements.txt        # Python dependencies
```

## Setup

### Prerequisites

- Conda or Miniconda
- Python 3.8+

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd resume-ai
```

2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate resume-ai
```

Or install dependencies with pip:

```bash
pip install -r requirements.txt
```

### Important: NumPy Compatibility

If you encounter NumPy-related errors, downgrade to a compatible version:

```bash
pip install "numpy<2"
```

3. Download NLTK data (if required):

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Usage

### Training the Model

Train the classifier with your resume dataset:

```bash
python src/train.py
```

This will:

- Load and preprocess resume data
- Generate semantic embeddings using Sentence Transformers
- Train a Logistic Regression model
- Save the model and embedder to `models/`

### Running the API

Start the FastAPI server:

```bash
conda activate resume-ai
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Making Predictions

**Endpoint**: `POST /predict`

**Request Body**:

```json
{
  "text": "Experienced software engineer with 5 years in Python development..."
}
```

**Response**:

```json
{
  "predicted_role": "Software Engineer",
  "confidence": 0.92
}
```

**Example with cURL**:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your resume text here..."}'
```

## Docker Deployment

Build and run with Docker:

```bash
docker build -t resume-ai .
docker run -p 8000:8000 resume-ai
```

## Model Evaluation

Evaluate model performance:

```bash
python src/evaluate.py
```

## Technologies

- **FastAPI**: Modern, fast web framework for building APIs
- **scikit-learn**: Machine learning library for classification
- **Sentence Transformers**: State-of-the-art semantic text embeddings
- **pandas**: Data manipulation and analysis
- **NLTK**: Natural language processing toolkit
- **joblib**: Model serialization
- **uvicorn**: ASGI server for FastAPI

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.
