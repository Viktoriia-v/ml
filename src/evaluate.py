"""Metrics + confusion matrix plotting + result-table append."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")  # safe for headless training runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config as C


# ---------------------------------------------------------------------------
def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    model_name: str,
    lang: str,
    save_path: str | Path | None = None,
    normalize: bool = False,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=C.LABEL_NAMES,
        yticklabels=C.LABEL_NAMES,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — {lang.upper()}")
    fig.tight_layout()

    out = Path(save_path or C.CM_DIR / f"{model_name}_{lang}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
RESULTS_TABLE = C.TABLES_DIR / "all_results.csv"


def append_results(model_name: str, lang: str, metrics: Dict[str, float], extra: Dict | None = None) -> None:
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "model": model_name,
        "language": lang,
        **{k: round(float(v), 4) for k, v in metrics.items()},
    }
    if extra:
        row["extra"] = json.dumps(extra, ensure_ascii=False)
    if RESULTS_TABLE.exists():
        df = pd.read_csv(RESULTS_TABLE)
        # replace any prior row for this (model, language) — last write wins
        df = df[~((df["model"] == model_name) & (df["language"] == lang))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    RESULTS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_TABLE, index=False)


def evaluate_and_log(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    model_name: str,
    lang: str,
    extra: Dict | None = None,
    plot: bool = True,
) -> Dict[str, float]:
    metrics = compute_metrics(y_true, y_pred)
    append_results(model_name, lang, metrics, extra=extra)
    if plot:
        plot_confusion_matrix(y_true, y_pred, model_name=model_name, lang=lang)
    return metrics


# ---------------------------------------------------------------------------
def summary_pivot() -> pd.DataFrame:
    """Pivot the long result table into the model × language × metric layout
    that goes straight into the thesis.
    """
    if not RESULTS_TABLE.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_TABLE)
    metric_cols = [c for c in df.columns if c not in {"timestamp", "model", "language", "extra"}]
    pivot = df.pivot_table(index="model", columns="language", values=metric_cols, aggfunc="last")
    out = C.TABLES_DIR / "summary_pivot.csv"
    pivot.to_csv(out)
    return pivot
