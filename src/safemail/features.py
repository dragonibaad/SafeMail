from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features)


def train_word2vec(
    texts: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    seed: int = 42,
) -> Word2Vec:
    tokenized = [text.split() for text in texts]
    return Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )


def average_word2vec_vectors(texts: list[str], w2v_model: Word2Vec) -> np.ndarray:
    vectors: list[np.ndarray] = []
    vector_size = w2v_model.vector_size
    for text in texts:
        tokens = [token for token in text.split() if token in w2v_model.wv]
        if not tokens:
            vectors.append(np.zeros(vector_size))
            continue
        token_vectors = np.array([w2v_model.wv[token] for token in tokens])
        vectors.append(token_vectors.mean(axis=0))
    return np.array(vectors)
