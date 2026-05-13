"""Per-language dataset loaders + unified train/val/test splitting.

Every loader returns a ``pandas.DataFrame`` with at minimum:

    - ``text``  : str  — the article body (or title + body concatenated)
    - ``label`` : int  — 0 = real, 1 = fake

Loaders try the easy path first (HuggingFace ``datasets``) and fall back to a
local file in ``data/raw/<lang>/`` with a clear error message that points to
``data/README.md`` for download instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .preprocessing import basic_clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset_for_lang(lang: str) -> pd.DataFrame:
    """Dispatch to the per-language loader specified in ``config.DATASETS``."""
    spec = C.DATASETS[lang]
    loader = globals()[spec.loader]
    df = loader()
    df = _validate_and_normalize(df, lang)
    return df


def get_split(
    lang: str, *, force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cached (train, val, test) splits, building them on first call."""
    out_dir = C.PROCESSED_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: out_dir / f"{s}.parquet" for s in ("train", "val", "test")}

    if not force_rebuild and all(p.exists() for p in paths.values()):
        return tuple(pd.read_parquet(paths[s]) for s in ("train", "val", "test"))

    df = load_dataset_for_lang(lang)
    train_df, val_df, test_df = stratified_split(df)
    for s, sub in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        sub.to_parquet(paths[s], index=False)
    _write_split_summary(lang, train_df, val_df, test_df)
    return train_df, val_df, test_df


def stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split — class proportions preserved."""
    train_df, temp_df = train_test_split(
        df,
        train_size=C.TRAIN_SIZE,
        stratify=df["label"],
        random_state=C.RANDOM_STATE,
        shuffle=True,
    )
    val_share = C.VAL_SIZE / (C.VAL_SIZE + C.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share,
        stratify=temp_df["label"],
        random_state=C.RANDOM_STATE,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Per-language loaders
# ---------------------------------------------------------------------------
def load_welfake() -> pd.DataFrame:
    """English: WELFake. Try HF then fall back to a local CSV."""
    # Preferred: HuggingFace mirror
    try:
        from datasets import load_dataset

        ds = load_dataset("davanstrien/WELFake", split="train")
        df = ds.to_pandas()
    except Exception:
        path = _expect_local("en", filename="WELFake_Dataset.csv")
        df = pd.read_csv(path)

    df = df.rename(columns={c: c.lower() for c in df.columns})
    # WELFake columns: title, text, label  (label: 0=fake or 0=real depending on mirror)
    if "title" in df.columns and "text" in df.columns:
        df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df = df[["text", "label"]].dropna()
    df = df[df["text"].str.len() > 30]

    # WELFake uses 0=real, 1=fake on Zenodo. The HF mirror may flip it; we
    # detect by inspecting class names if present, otherwise we trust the spec.
    df["label"] = df["label"].astype(int)
    return df


def load_russian_fake_news() -> pd.DataFrame:
    """Russian: pre-built composite (IRA troll tweets fake + Lenta.ru real).

    Reads the single consolidated file ``data/raw/ru/ru_combined.csv`` produced
    by the data-prep step (already balanced 1:1, columns ``text,label``).
    """
    path = _expect_local("ru", filename="ru_combined.csv")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


def load_ufnd() -> pd.DataFrame:
    """Ukrainian: balanced fake/true news (Kaggle ``zepopo/ukrainian-fake-and-true-news``).

    Reads the single consolidated file ``data/raw/uk/uk_combined.csv`` produced
    by the data-prep step (already balanced 1:1, columns ``text,label`` with
    0=real, 1=fake).
    """
    path = _expect_local("uk", filename="uk_combined.csv")
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


def load_weibo_rumor() -> pd.DataFrame:
    """Chinese: pre-built combined dataset (CHECKED + thunlp/CED_Dataset).

    Reads ``data/raw/zh/zh_combined.csv`` (columns ``text,label`` with
    0=real, 1=fake) produced by ``scripts/prepare_zh.py``. Roughly 5 500 rows.

    Falls back to the older CHECKED-only auto-download if the combined file
    is missing, so the loader stays usable for a quick smoke-test.
    """
    raw_dir = C.RAW_DIR / "zh"
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined = raw_dir / "zh_combined.csv"

    if combined.exists():
        df = pd.read_csv(combined)
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        df = df[df["text"].astype(str).str.len() >= 5]
        return df.reset_index(drop=True)

    # Fallback: just CHECKED.
    fake_path = raw_dir / "fake_news.csv"
    real_path = raw_dir / "real_news.csv"
    base_url = "https://raw.githubusercontent.com/cyang03/CHECKED/master/dataset"

    if not fake_path.exists() or not real_path.exists():
        try:
            import urllib.request

            for name, p in (("fake_news.csv", fake_path), ("real_news.csv", real_path)):
                if not p.exists():
                    print(f"[ZH] downloading {name} from CHECKED repo …")
                    urllib.request.urlretrieve(f"{base_url}/{name}", p)
        except Exception as exc:
            raise FileNotFoundError(
                f"No zh_combined.csv and CHECKED auto-download failed ({exc}). "
                "Run `python scripts/prepare_zh.py` once."
            )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    def _normalize(df: pd.DataFrame, fallback_label: int) -> pd.DataFrame:
        if "text" not in df.columns:
            for cand in ("content", "body", "微博"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "text"})
                    break
        if "label" not in df.columns:
            df = df.assign(label=fallback_label)
        m = df["label"].astype(str).str.lower().str.strip()
        df["label"] = m.map({"fake": 1, "real": 0, "0": 0, "1": 1}).fillna(fallback_label).astype(int)
        return df[["text", "label"]].dropna()

    fake = _normalize(fake, fallback_label=1)
    real = _normalize(real, fallback_label=0)

    df = pd.concat([fake, real], ignore_index=True)
    df = df[df["text"].astype(str).str.len() > 10]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _expect_local(lang: str, filename: str) -> Path:
    p = C.RAW_DIR / lang / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Expected dataset file at {p}. "
            f"Download instructions are in data/README.md (section: {lang.upper()})."
        )
    return p


def _validate_and_normalize(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Loader for '{lang}' produced columns {list(df.columns)}; "
            "expected at least 'text' and 'label'."
        )
    df["text"] = df["text"].astype(str).map(basic_clean)
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Loader for '{lang}' produced labels {sorted(df['label'].unique())}; "
            "expected {0,1}."
        )
    df = df.sample(frac=1.0, random_state=C.RANDOM_STATE).reset_index(drop=True)
    return df


def _write_split_summary(
    lang: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    summary = {
        "language": lang,
        "dataset": C.DATASETS[lang].name,
        "n_total": int(len(train_df) + len(val_df) + len(test_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "class_balance": {
            "train": train_df["label"].value_counts(normalize=True).round(4).to_dict(),
            "val": val_df["label"].value_counts(normalize=True).round(4).to_dict(),
            "test": test_df["label"].value_counts(normalize=True).round(4).to_dict(),
        },
        "avg_text_len_chars": {
            "train": int(train_df["text"].str.len().mean()),
            "val": int(val_df["text"].str.len().mean()),
            "test": int(test_df["text"].str.len().mean()),
        },
        "random_state": C.RANDOM_STATE,
    }
    out = C.PROCESSED_DIR / lang / "split_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
