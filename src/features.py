from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(corpus):
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
