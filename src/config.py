"""Central configuration: paths, languages, model hyperparameters.

All numbers and string identifiers used across the pipeline live here so that
changing a hyperparameter never requires editing more than one file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
CM_DIR = RESULTS_DIR / "confusion_matrices"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

THESIS_DIR = ROOT / "thesis_excerpts"

for d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, CM_DIR, EXPLAIN_DIR, THESIS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ---------------------------------------------------------------------------
# Languages and dataset registry
# ---------------------------------------------------------------------------
LANGUAGES: List[str] = ["en", "ru", "uk", "zh"]


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a per-language dataset.

    `loader` is the name of a function inside ``src.data_utils`` that returns a
    ``pandas.DataFrame`` with at least the columns ``text`` and ``label``
    (label: 0 = real, 1 = fake).
    """

    name: str
    loader: str
    license: str
    url: str
    notes: str = ""


DATASETS: Dict[str, DatasetSpec] = {
    "en": DatasetSpec(
        name="WELFake",
        loader="load_welfake",
        license="CC BY 4.0",
        url="https://zenodo.org/record/4561253",
        notes="72k articles, balanced. Title+text concat.",
    ),
    "ru": DatasetSpec(
        name="Russian Troll Tweets (FiveThirtyEight) + Lenta.ru news",
        loader="load_russian_fake_news",
        license="CC0 (troll tweets) + research-use (Lenta.ru)",
        url="https://www.kaggle.com/datasets/fivethirtyeight/russian-troll-tweets",
        notes=(
            "Composite binary dataset: IRA troll tweets in Russian = fake; "
            "Lenta.ru articles = real. See data/README.md for download steps."
        ),
    ),
    "uk": DatasetSpec(
        name="Ukrainian Fake and True News (Kaggle)",
        loader="load_ufnd",
        license="see Kaggle page",
        url="https://www.kaggle.com/datasets/zepopo/ukrainian-fake-and-true-news",
        notes="Russo-Ukrainian war news with fake/true labels; Kaggle CLI download.",
    ),
    "zh": DatasetSpec(
        name="CHECKED + thunlp/CED_Dataset (Weibo, combined)",
        loader="load_weibo_rumor",
        license="academic research",
        url="https://github.com/cyang03/CHECKED + https://github.com/thunlp/Chinese_Rumor_Dataset",
        notes=(
            "Composite ~5 500 microblogs from CHECKED (COVID-19) and thunlp's "
            "CED_Dataset (general rumours, 2009-2017). Run "
            "`python scripts/prepare_zh.py` once to build data/raw/zh/zh_combined.csv."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
LABEL_NAMES = ["real", "fake"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class TfidfLRConfig:
    max_features: int = 50_000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    C: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"


@dataclass
class XGBConfig:
    max_features: int = 50_000
    ngram_range: tuple = (1, 2)
    n_estimators: int = 600
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    n_jobs: int = -1


@dataclass
class BiLSTMConfig:
    embedding_dim: int = 300
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    bidirectional: bool = True
    max_len: int = 256
    vocab_size: int = 50_000
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    fasttext_models: Dict[str, str] = field(
        default_factory=lambda: {
            # Facebook FastText cc.<lang>.300.bin
            "en": "facebook/fasttext-en-vectors",
            "ru": "facebook/fasttext-ru-vectors",
            "uk": "facebook/fasttext-uk-vectors",
            "zh": "facebook/fasttext-zh-vectors",
        }
    )


@dataclass
class TransformerConfig:
    model_name: str = "xlm-roberta-base"   # or bert-base-multilingual-cased / xlm-roberta-large
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 3
    fp16: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class LLMConfig:
    """OpenRouter is OpenAI-compatible — we point the OpenAI SDK at it."""
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.0
    max_tokens: int = 32
    n_few_shot: int = 4
    request_timeout_s: int = 60
    max_retries: int = 5
    sleep_between_requests_s: float = 0.0


# ---------------------------------------------------------------------------
# Convenience: pick up env vars eagerly so callers can rely on them
# ---------------------------------------------------------------------------
def load_dotenv_if_present() -> None:
    """Load .env file from repo root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


load_dotenv_if_present()


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise EnvironmentError(
            f"Environment variable {name} is required but not set. "
            f"Add it to your .env file (see .env.example)."
        )
    return val
