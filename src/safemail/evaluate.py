from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from .config import AppConfig
from .data import build_dataset_bundle
from .features import average_word2vec_vectors
from .features_glove import average_glove_vectors


@dataclass
class EvalResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def _scores(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, name: str) -> EvalResult:
    return EvalResult(
        model=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def evaluate_and_export(config: AppConfig | None = None) -> pd.DataFrame:
    cfg = config or AppConfig.from_env()
    data = build_dataset_bundle(cfg.dataset_path, cfg.test_size, cfg.random_state)
    y_true = data.y_test.to_numpy()
    X_test = data.X_test.tolist()

    tfidf = joblib.load(cfg.models_dir / "tfidf_vectorizer.pkl")
    lr = joblib.load(cfg.models_dir / "lr_model.pkl")
    nb = joblib.load(cfg.models_dir / "nb_model.pkl")
    rf = joblib.load(cfg.models_dir / "rf_model.pkl")
    rf_glove = None
    rf_glove_path = cfg.models_dir / "rf_glove_model.pkl"
    glove_path = cfg.models_dir / "glove_vectors.kv"
    if rf_glove_path.exists() and glove_path.exists():
        rf_glove = joblib.load(rf_glove_path)
    from gensim.models import Word2Vec

    w2v = Word2Vec.load(str(cfg.models_dir / "w2v_model.model"))
    glove_vectors = None
    if glove_path.exists():
        from gensim.models import KeyedVectors

        glove_vectors = KeyedVectors.load(str(glove_path), mmap="r")

    X_test_tfidf = tfidf.transform(X_test)
    X_test_w2v = average_word2vec_vectors(X_test, w2v)
    X_test_glove = average_glove_vectors(X_test, glove_vectors) if glove_vectors is not None else None

    lr_prob = lr.predict_proba(X_test_tfidf)[:, 1]
    nb_prob = nb.predict_proba(X_test_tfidf)[:, 1]
    rf_prob = rf.predict_proba(X_test_w2v)[:, 1]
    rf_glove_prob = rf_glove.predict_proba(X_test_glove)[:, 1] if rf_glove is not None and X_test_glove is not None else None

    tokenizer = AutoTokenizer.from_pretrained(cfg.models_dir / "distilbert_model")
    db_model = TFAutoModelForSequenceClassification.from_pretrained(cfg.models_dir / "distilbert_model")
    encodings = tokenizer(X_test, truncation=True, padding=True, max_length=256, return_tensors="tf")
    logits = db_model(encodings).logits
    db_prob = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]

    results = [
        _scores(y_true, (lr_prob >= 0.5).astype(int), lr_prob, "Logistic Regression (TF-IDF)"),
        _scores(y_true, (nb_prob >= 0.5).astype(int), nb_prob, "Naive Bayes (TF-IDF)"),
        _scores(y_true, (rf_prob >= 0.5).astype(int), rf_prob, "Random Forest (Word2Vec)"),
        _scores(y_true, (db_prob >= 0.5).astype(int), db_prob, "DistilBERT"),
    ]
    if rf_glove_prob is not None:
        results.append(_scores(y_true, (rf_glove_prob >= 0.5).astype(int), rf_glove_prob, "Random Forest (GloVe)"))

    table = pd.DataFrame([r.__dict__ for r in results])
    output_path = Path("reports")
    output_path.mkdir(exist_ok=True)
    table.to_csv(output_path / "model_comparison.csv", index=False)
    return table


if __name__ == "__main__":
    print(evaluate_and_export())
