from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from .config import AppConfig
from .data import build_dataset_bundle
from .features import average_word2vec_vectors, build_tfidf_vectorizer, train_word2vec
from .features_glove import average_glove_vectors, load_or_download_glove


def train_and_save_classical_models(config: AppConfig | None = None) -> None:
    cfg = config or AppConfig.from_env()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataset_bundle(cfg.dataset_path, cfg.test_size, cfg.random_state)
    X_train = data.X_train.tolist()
    y_train = data.y_train

    tfidf_vectorizer = build_tfidf_vectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=cfg.random_state)
    lr_model.fit(X_train_tfidf, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    w2v_model = train_word2vec(X_train, seed=cfg.random_state)
    X_train_w2v = average_word2vec_vectors(X_train, w2v_model)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train_w2v, y_train)

    glove_vectors = load_or_download_glove(cfg.models_dir)
    X_train_glove = average_glove_vectors(X_train, glove_vectors)
    rf_glove_model = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rf_glove_model.fit(X_train_glove, y_train)

    joblib.dump(tfidf_vectorizer, cfg.models_dir / "tfidf_vectorizer.pkl")
    joblib.dump(lr_model, cfg.models_dir / "lr_model.pkl")
    joblib.dump(nb_model, cfg.models_dir / "nb_model.pkl")
    joblib.dump(rf_model, cfg.models_dir / "rf_model.pkl")
    joblib.dump(rf_glove_model, cfg.models_dir / "rf_glove_model.pkl")
    w2v_model.save(str(cfg.models_dir / "w2v_model.model"))


if __name__ == "__main__":
    train_and_save_classical_models()
