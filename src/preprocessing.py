"""Text preprocessing utilities — language-aware cleaning + tokenization.

Design:
    * For TF-IDF / XGBoost we apply *light* cleaning + language-specific
      tokenization (jieba for Chinese, whitespace+regex for the rest).
    * For transformers we do **no** preprocessing beyond stripping URLs and
      collapsing whitespace — the model's tokenizer handles the rest.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, List

# ---------------------------------------------------------------------------
# Generic cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MULTI_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^\w\sЀ-ӿ一-鿿]+", flags=re.UNICODE)


def basic_clean(text: str) -> str:
    """Light, lossless-ish cleanup: NFC, drop URLs/HTML/emails, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def heavy_clean(text: str) -> str:
    """Lowercase + strip punctuation. Use for TF-IDF on EN/RU/UK only."""
    text = basic_clean(text).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Language-specific tokenization for TF-IDF
# ---------------------------------------------------------------------------
def tokenize_zh(text: str) -> List[str]:
    """Chinese segmentation via jieba."""
    import jieba

    text = basic_clean(text)
    return [t for t in jieba.cut(text) if t.strip()]


def tokenize_whitespace(text: str) -> List[str]:
    return heavy_clean(text).split()


def get_tokenizer(lang: str) -> Callable[[str], List[str]]:
    """Return a callable tokenizer suitable for sklearn TfidfVectorizer."""
    if lang == "zh":
        return tokenize_zh
    return tokenize_whitespace


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def clean_series(texts: Iterable[str], *, heavy: bool = False) -> List[str]:
    fn = heavy_clean if heavy else basic_clean
    return [fn(t) for t in texts]
