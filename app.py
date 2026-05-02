from __future__ import annotations

import csv
import io
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safemail.predict import SafeMailPredictor

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> SafeMailPredictor:
    return SafeMailPredictor()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_single():
    email_text = request.form.get("email_text", "")
    if not email_text.strip():
        return render_template("index.html", error="Please enter an email body.")

    result = get_predictor().predict_single(email_text)
    return render_template("index.html", single_result=result, email_text=email_text)


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    uploaded = request.files.get("batch_file")
    if not uploaded or not uploaded.filename:
        return render_template("index.html", error="Please upload a CSV/TXT file for batch prediction.")

    raw = uploaded.read().decode("utf-8", errors="ignore")
    emails: list[str] = []

    if uploaded.filename.lower().endswith(".csv"):
        reader = csv.DictReader(io.StringIO(raw))
        if "Email Text" in (reader.fieldnames or []):
            emails = [row.get("Email Text", "") for row in reader]
        else:
            reader = csv.reader(io.StringIO(raw))
            emails = [",".join(row) for row in reader]
    else:
        emails = [line.strip() for line in raw.splitlines() if line.strip()]

    results = get_predictor().predict_batch(emails)
    phishing_count = sum(1 for r in results if r["final_label"] == "Phishing")
    safe_count = len(results) - phishing_count
    phishing_percent = (phishing_count / len(results) * 100) if results else 0.0

    return render_template(
        "index.html",
        batch_results=results,
        batch_stats={
            "total": len(results),
            "phishing_count": phishing_count,
            "safe_count": safe_count,
            "phishing_percent": round(phishing_percent, 2),
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
