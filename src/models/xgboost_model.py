"""XGBoost on TF-IDF features.

Same vectorizer family as the baseline, but with a non-linear classifier.
We bundle vectorizer + booster in a sklearn ``Pipeline`` for ergonomic SHAP.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .. import config as C
from ..preprocessing import get_tokenizer


def build_pipeline(lang: str, cfg: C.XGBConfig | None = None) -> Pipeline:
    cfg = cfg or C.XGBConfig()
    tokenizer = get_tokenizer(lang)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    clf = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=cfg.n_jobs,
        random_state=C.RANDOM_STATE,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lang: str,
    cfg: C.XGBConfig | None = None,
) -> Pipeline:
    pipe = build_pipeline(lang, cfg)
    # We fit the vectorizer on train, then transform both for early stopping.
    X_train = pipe.named_steps["tfidf"].fit_transform(train_df["text"].tolist())
    X_val = pipe.named_steps["tfidf"].transform(val_df["text"].tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.named_steps["clf"].fit(
            X_train,
            train_df["label"].values,
            eval_set=[(X_val, val_df["label"].values)],
            verbose=False,
        )
    return pipe


def predict(pipe: Pipeline, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    X = pipe.named_steps["tfidf"].transform(df["text"].tolist())
    return pipe.named_steps["clf"].predict(X)


def predict_proba(pipe: Pipeline, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    X = pipe.named_steps["tfidf"].transform(df["text"].tolist())
    return pipe.named_steps["clf"].predict_proba(X)


def save(pipe: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
