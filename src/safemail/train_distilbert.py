from __future__ import annotations

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from .config import AppConfig
from .data import build_dataset_bundle


def train_and_save_distilbert(config: AppConfig | None = None, epochs: int = 2, max_length: int = 256) -> None:
    cfg = config or AppConfig.from_env()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = cfg.models_dir / "distilbert_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataset_bundle(cfg.dataset_path, cfg.test_size, cfg.random_state)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        from_pt=True,
        use_safetensors=False,
    )

    train_enc = tokenizer(
        data.X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf",
    )
    labels = tf.convert_to_tensor(data.y_train.tolist())
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), labels)).shuffle(1024).batch(16)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.fit(train_dataset, epochs=epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train_and_save_distilbert()
