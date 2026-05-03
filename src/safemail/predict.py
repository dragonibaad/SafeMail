from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from .config import AppConfig
from .data import clean_text
from .features import average_word2vec_vectors
from .features_glove import average_glove_vectors
from .interpretability import extract_suspicious_indicators, highlight_text


@dataclass
class ModelOutputs:
    label: str
    confidence: float


class SafeMailPredictor:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.cfg = config or AppConfig.from_env()
        self.models_dir = Path(self.cfg.models_dir)
        self.tfidf = joblib.load(self.models_dir / "tfidf_vectorizer.pkl")
        self.lr = joblib.load(self.models_dir / "lr_model.pkl")
        self.nb = joblib.load(self.models_dir / "nb_model.pkl")
        self.rf = joblib.load(self.models_dir / "rf_model.pkl")
        self.rf_glove = None
        self.w2v = Word2Vec.load(str(self.models_dir / "w2v_model.model"))
        self.glove_vectors = None
        rf_glove_path = self.models_dir / "rf_glove_model.pkl"
        glove_path = self.models_dir / "glove_vectors.kv"
        if rf_glove_path.exists() and glove_path.exists():
            self.rf_glove = joblib.load(rf_glove_path)
            self.glove_vectors = KeyedVectors.load(str(glove_path), mmap="r")
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_dir / "distilbert_model")
        self.distilbert = TFAutoModelForSequenceClassification.from_pretrained(self.models_dir / "distilbert_model")

    @staticmethod
    def _label(probability: float) -> str:
        return "Phishing" if probability >= 0.5 else "Safe"

    def predict_single(self, email_text: str) -> dict[str, Any]:
        cleaned = clean_text(email_text)
        tfidf_features = self.tfidf.transform([cleaned])
        w2v_features = average_word2vec_vectors([cleaned], self.w2v)
        glove_features = None
        if self.rf_glove is not None and self.glove_vectors is not None:
            glove_features = average_glove_vectors([cleaned], self.glove_vectors)

        lr_prob = float(self.lr.predict_proba(tfidf_features)[0][1])
        nb_prob = float(self.nb.predict_proba(tfidf_features)[0][1])
        rf_prob = float(self.rf.predict_proba(w2v_features)[0][1])
        rf_glove_prob = None
        if self.rf_glove is not None and glove_features is not None:
            rf_glove_prob = float(self.rf_glove.predict_proba(glove_features)[0][1])

        enc = self.tokenizer(cleaned, truncation=True, padding=True, max_length=256, return_tensors="tf")
        logits = self.distilbert(enc).logits
        db_prob = float(tf.nn.softmax(logits, axis=-1).numpy()[0][1])

        probs = [lr_prob, nb_prob, rf_prob, db_prob]
        if rf_glove_prob is not None:
            probs.append(rf_glove_prob)
        votes = [1 if p >= 0.5 else 0 for p in probs]
        final_vote = 1 if sum(votes) >= 2 else 0
        ensemble_conf = float(np.mean(probs))

        indicators = extract_suspicious_indicators(email_text)
        model_outputs = {
            "logistic_regression": {"label": self._label(lr_prob), "confidence": lr_prob},
            "naive_bayes": {"label": self._label(nb_prob), "confidence": nb_prob},
            "random_forest_w2v": {"label": self._label(rf_prob), "confidence": rf_prob},
            "distilbert": {"label": self._label(db_prob), "confidence": db_prob},
        }
        if rf_glove_prob is not None:
            model_outputs["random_forest_glove"] = {
                "label": self._label(rf_glove_prob),
                "confidence": rf_glove_prob,
            }

        return {
            "final_label": "Phishing" if final_vote == 1 else "Safe",
            "final_confidence": ensemble_conf if final_vote == 1 else 1 - ensemble_conf,
            "models": model_outputs,
            "indicators": indicators,
            "highlighted_email_html": highlight_text(email_text),
        }

    def predict_batch(self, email_texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict_single(text) for text in email_texts if str(text).strip()]
