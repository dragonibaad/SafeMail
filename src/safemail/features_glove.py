from __future__ import annotations

from pathlib import Path

import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors


def load_or_download_glove(models_dir: Path, model_name: str = "glove-wiki-gigaword-100") -> KeyedVectors:
    """Load cached GloVe vectors or download and cache them locally."""
    glove_path = models_dir / "glove_vectors.kv"
    if glove_path.exists():
        return KeyedVectors.load(str(glove_path), mmap="r")

    glove_vectors = api.load(model_name)
    glove_vectors.save(str(glove_path))
    return glove_vectors


def average_glove_vectors(texts: list[str], glove_vectors: KeyedVectors) -> np.ndarray:
    vectors: list[np.ndarray] = []
    vector_size = glove_vectors.vector_size
    for text in texts:
        tokens = [token for token in text.split() if token in glove_vectors]
        if not tokens:
            vectors.append(np.zeros(vector_size))
            continue
        token_vectors = np.array([glove_vectors[token] for token in tokens])
        vectors.append(token_vectors.mean(axis=0))
    return np.array(vectors)
