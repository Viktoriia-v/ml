"""TF-IDF + Logistic Regression baseline.

The whole pipeline is wrapped into a single sklearn ``Pipeline`` so that we can
pickle it, ship it, and reuse it for SHAP / LIME explanations directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .. import config as C
from ..preprocessing import get_tokenizer


# ---------------------------------------------------------------------------
def build_pipeline(lang: str, cfg: C.TfidfLRConfig | None = None) -> Pipeline:
    cfg = cfg or C.TfidfLRConfig()
    tokenizer = get_tokenizer(lang)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,            # silence sklearn warning when tokenizer is set
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        n_jobs=-1,
        random_state=C.RANDOM_STATE,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lang: str,
    cfg: C.TfidfLRConfig | None = None,
) -> Pipeline:
    pipe = build_pipeline(lang, cfg)
    pipe.fit(train_df["text"].tolist(), train_df["label"].values)
    return pipe


def predict(pipe: Pipeline, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    return pipe.predict(df["text"].tolist())


def predict_proba(pipe: Pipeline, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    return pipe.predict_proba(df["text"].tolist())


# ---------------------------------------------------------------------------
def save(pipe: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)


# ---------------------------------------------------------------------------
def top_features(pipe: Pipeline, top_k: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the most positive (fake-indicating) and most negative (real)
    coefficients with their TF-IDF feature names. Useful for the explainability
    chapter — gives a free 'global' interpretation of the baseline.
    """
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    feat_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()
    order = np.argsort(coefs)
    fake = pd.DataFrame({"feature": feat_names[order[-top_k:][::-1]],
                         "coef": coefs[order[-top_k:][::-1]]})
    real = pd.DataFrame({"feature": feat_names[order[:top_k]],
                         "coef": coefs[order[:top_k]]})
    return fake, real
