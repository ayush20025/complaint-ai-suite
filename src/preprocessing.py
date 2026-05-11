"""Data preprocessing pipeline for customer complaints."""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.utils import get_logger, normalize_text

logger = get_logger(__name__)

FALLBACK_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "to", "for", "and", "or", "of", "in", "on", "with", "my", "our", "your", "has", "have", "had", "been", "after", "from", "this", "that"}


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b(order|invoice|ticket)\s*#?\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text) if text else []


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token.lower() not in FALLBACK_STOPWORDS]


def lemmatize_text(tokens: Iterable[str]) -> List[str]:
    simplified: List[str] = []
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            simplified.append(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            simplified.append(token[:-1])
        else:
            simplified.append(token)
    return simplified


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


def preprocess_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if "complaint_text" not in df.columns:
        raise ValueError("Input dataset must include a 'complaint_text' column.")
    df["cleaned_text"] = df["complaint_text"].astype(str).map(preprocess_text)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessed dataset saved to %s", output_path)
    return df


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    raw_path = base_path / "data" / "raw" / "complaints.csv"
    processed_path = base_path / "data" / "processed" / "cleaned_complaints.csv"
    preprocess_dataset(raw_path, processed_path)
