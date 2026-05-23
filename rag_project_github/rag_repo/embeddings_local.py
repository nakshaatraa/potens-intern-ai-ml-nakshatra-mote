"""
Local TF-IDF based embeddings that work without any internet access.
Drop-in replacement for HuggingFaceEmbeddings.
"""
import numpy as np
import pickle, os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db", "tfidf_vocab.pkl")

class LocalTFIDFEmbeddings:
    """Sklearn TF-IDF vectorizer wrapped as a LangChain-compatible embedding class."""
    def __init__(self):
        self.vectorizer = None

    def _load_or_fit(self, texts=None):
        if self.vectorizer is not None:
            return
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
        elif texts:
            self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            self.vectorizer.fit(texts)
            os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
            with open(VOCAB_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
        else:
            raise RuntimeError("No vocabulary found. Fit the vectorizer first.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load_or_fit(texts)
        matrix = self.vectorizer.transform(texts).toarray()
        return matrix.tolist()

    def embed_query(self, text: str) -> list[float]:
        self._load_or_fit()
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec.tolist()
