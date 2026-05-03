from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EMAIL_TEXT_COL = "Email Text"
EMAIL_TYPE_COL = "Email Type"
LABEL_COL = "label_num"
LABEL_MAP = {"Safe Email": 0, "Phishing Email": 1}
# For this dataset variant, numeric labels are inverted:
# 0 = phishing, 1 = safe
NUMERIC_LABEL_MAP = {0: 1, 1: 0}

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_REGEX = re.compile(r"[^a-zA-Z\s]")
SPACE_REGEX = re.compile(r"\s+")


@dataclass
class DatasetBundle:
    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    full_df: pd.DataFrame


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_REGEX.sub(" ", text)
    text = NON_LETTER_REGEX.sub(" ", text)
    return SPACE_REGEX.sub(" ", text).strip()


def load_raw_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    required = {EMAIL_TEXT_COL, EMAIL_TYPE_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df[EMAIL_TEXT_COL] = clean_df[EMAIL_TEXT_COL].fillna("").astype(str)
    clean_df["clean_text"] = clean_df[EMAIL_TEXT_COL].map(clean_text)

    def _to_binary_label(value: object) -> float:
        if pd.isna(value):
            return np.nan

        # Handle numeric labels directly (dataset uses 0=phishing, 1=safe).
        if isinstance(value, (int, float)):
            if int(value) in (0, 1):
                return NUMERIC_LABEL_MAP[int(value)]

        text = str(value).strip().lower()
        if text in {"0"}:
            return NUMERIC_LABEL_MAP[0]
        if text in {"1"}:
            return NUMERIC_LABEL_MAP[1]
        if text in {"safe email", "safe", "ham", "not phishing"}:
            return 0
        if text in {"phishing email", "phishing", "spam"}:
            return 1
        return np.nan

    clean_df[LABEL_COL] = clean_df[EMAIL_TYPE_COL].map(_to_binary_label)
    clean_df = clean_df.dropna(subset=[LABEL_COL]).copy()
    clean_df[LABEL_COL] = clean_df[LABEL_COL].astype(int)
    clean_df = clean_df[clean_df["clean_text"].str.len() > 0].copy()
    if clean_df.empty:
        raise ValueError(
            "No usable rows after preprocessing. Check 'Email Type' label format and 'Email Text' content."
        )
    return clean_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        df["clean_text"],
        df[LABEL_COL],
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COL],
    )


def build_dataset_bundle(dataset_path: Path, test_size: float, random_state: int) -> DatasetBundle:
    raw_df = load_raw_dataset(dataset_path)
    processed_df = preprocess_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(
        processed_df,
        test_size=test_size,
        random_state=random_state,
    )
    return DatasetBundle(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, full_df=processed_df)
