import numpy as np
from gensim.models import KeyedVectors

from safemail.features_glove import average_glove_vectors


def test_average_glove_vectors_shape_and_defaults():
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["hello", "world"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    vectors = average_glove_vectors(["hello world", "unknown"], kv)
    assert vectors.shape == (2, 3)
    assert np.allclose(vectors[0], np.array([0.5, 0.5, 0.0]))
    assert np.allclose(vectors[1], np.zeros(3))
