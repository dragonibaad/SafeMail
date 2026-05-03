# SafeMail

SafeMail is an AI-powered phishing email classifier that labels emails as **Phishing** or **Safe** using multiple ML models and exposes predictions through a Flask web app.

## Project Scope

- Binary classification: `Phishing` vs `Safe`
- Models:
  - TF-IDF + Logistic Regression
  - TF-IDF + Naive Bayes
  - Word2Vec + Random Forest
  - GloVe + Random Forest
  - DistilBERT (transformer fine-tuning)
- Outputs:
  - Final label
  - Confidence score
  - Suspicious indicators (URLs and urgency terms)
  - Highlighted suspicious text

## Dataset Contract

Expected CSV input schema:

- `Email Text`: raw email content
- `Email Type`: label string (`Safe Email` or `Phishing Email`)

Canonical label mapping:

- `Safe Email -> 0`
- `Phishing Email -> 1`

Canonical preprocessing (`src/safemail/data.py`):

- lowercase conversion
- URL removal
- non-letter character cleanup
- whitespace normalization
- removal of empty cleaned records

Default dataset path:

- `Email Dataset/Email_Dataset.csv`

You can override it with:

- `SAFEMAIL_DATASET_PATH=/path/to/file.csv`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local package imports:

```bash
export PYTHONPATH=src
```

## Training

Train classical + DistilBERT models:

```bash
python3 scripts/train_all.py
```

Artifacts are saved in `Models/`.
During classical training, SafeMail also downloads (once) and caches GloVe vectors into `Models/glove_vectors.kv`.

## Evaluation

Run unified evaluation and export report:

```bash
python3 scripts/evaluate_models.py
```

Output report:

- `reports/model_comparison.csv`

## Flask Web App

Run app:

```bash
python3 app.py
```

Features:

- single email prediction
- batch upload (`.csv` or `.txt`)
- model confidence display
- suspicious indicators and highlighted text
- basic batch dashboard stats

## Tests

Run tests:

```bash
pytest -q
```
