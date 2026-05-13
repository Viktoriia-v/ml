"""LLM zero-shot and few-shot classification via OpenRouter.

OpenRouter exposes an OpenAI-compatible API, so we use the official ``openai``
SDK with a custom ``base_url``. This means a single client object works for
``anthropic/claude-3.5-sonnet``, ``openai/gpt-4o``, ``meta-llama/llama-3.1-70b``,
and so on — pick the model name from the OpenRouter catalog.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .. import config as C


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert media-literacy analyst. Classify a news article as either "
    "REAL or FAKE based on its content. Reply with strict JSON: "
    '{"label": "real|fake", "confidence": 0.0-1.0, "reason": "..."}'
)

USER_TEMPLATE_ZERO = (
    "Article (language hint: {lang_name}):\n\n"
    "{text}\n\n"
    "Return JSON only."
)

USER_TEMPLATE_FEW = (
    "Below are {k} labeled examples, then one new article to classify.\n\n"
    "{examples}\n"
    "--- TARGET ARTICLE (language hint: {lang_name}) ---\n"
    "{text}\n\n"
    "Return JSON only for the TARGET ARTICLE."
)

LANG_NAMES = {"en": "English", "ru": "Russian", "uk": "Ukrainian", "zh": "Chinese"}


# ---------------------------------------------------------------------------
def _client(cfg: C.LLMConfig):
    """Build an OpenAI-compatible client pointing at OpenRouter."""
    from openai import OpenAI

    api_key = C.get_env(cfg.api_key_env, required=True)
    extra_headers = {}
    if ref := C.get_env("OPENROUTER_HTTP_REFERER"):
        extra_headers["HTTP-Referer"] = ref
    if title := C.get_env("OPENROUTER_X_TITLE"):
        extra_headers["X-Title"] = title

    return OpenAI(
        api_key=api_key,
        base_url=cfg.base_url,
        default_headers=extra_headers or None,
        timeout=cfg.request_timeout_s,
    )


def _truncate(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + " […]"


def _format_examples(examples: List[Tuple[str, int]]) -> str:
    out = []
    for i, (txt, label) in enumerate(examples, start=1):
        name = "FAKE" if label == 1 else "REAL"
        out.append(f"Example {i} (label = {name}):\n{_truncate(txt, 800)}\n")
    return "\n".join(out)


_LABEL_RE = re.compile(r'"label"\s*:\s*"(real|fake)"', re.I)


def _parse_response(content: str) -> Dict:
    """Be liberal in what we accept. The model may wrap JSON in prose or code fences."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S | re.I)
    try:
        return json.loads(s)
    except Exception:
        m = _LABEL_RE.search(content)
        if m:
            return {"label": m.group(1).lower(), "confidence": None, "reason": ""}
    # Last resort: heuristic
    low = content.lower()
    if "fake" in low and "real" not in low:
        return {"label": "fake", "confidence": None, "reason": ""}
    if "real" in low and "fake" not in low:
        return {"label": "real", "confidence": None, "reason": ""}
    return {"label": "real", "confidence": None, "reason": "unparsed"}  # safe default


def _classify_one(client, cfg: C.LLMConfig, system: str, user: str) -> Dict:
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens if cfg.max_tokens > 64 else 256,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            return _parse_response(content)
        except Exception as exc:
            last_err = exc
            sleep_s = min(30.0, 2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM request failed after {cfg.max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_zero_shot(
    df: pd.DataFrame,
    lang: str,
    cfg: C.LLMConfig | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    cfg = cfg or C.LLMConfig()
    client = _client(cfg)
    rows = []
    iterator = tqdm(df.itertuples(index=False), total=len(df), disable=not show_progress)
    for row in iterator:
        user = USER_TEMPLATE_ZERO.format(
            lang_name=LANG_NAMES.get(lang, lang),
            text=_truncate(row.text),
        )
        result = _classify_one(client, cfg, SYSTEM_PROMPT, user)
        rows.append(result)
        if cfg.sleep_between_requests_s:
            time.sleep(cfg.sleep_between_requests_s)
    out = pd.DataFrame(rows)
    out["pred"] = (out["label"].str.lower() == "fake").astype(int)
    return out


def classify_few_shot(
    df: pd.DataFrame,
    lang: str,
    train_df: pd.DataFrame,
    cfg: C.LLMConfig | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Pick ``cfg.n_few_shot`` balanced examples from ``train_df`` once, reuse them."""
    cfg = cfg or C.LLMConfig()
    half = max(1, cfg.n_few_shot // 2)
    real_ex = train_df[train_df["label"] == 0].sample(n=half, random_state=C.RANDOM_STATE)
    fake_ex = train_df[train_df["label"] == 1].sample(n=cfg.n_few_shot - half, random_state=C.RANDOM_STATE)
    examples = list(zip(real_ex["text"], real_ex["label"])) + list(zip(fake_ex["text"], fake_ex["label"]))
    np.random.RandomState(C.RANDOM_STATE).shuffle(examples)
    examples_block = _format_examples(examples)

    client = _client(cfg)
    rows = []
    iterator = tqdm(df.itertuples(index=False), total=len(df), disable=not show_progress)
    for row in iterator:
        user = USER_TEMPLATE_FEW.format(
            k=cfg.n_few_shot,
            examples=examples_block,
            lang_name=LANG_NAMES.get(lang, lang),
            text=_truncate(row.text),
        )
        result = _classify_one(client, cfg, SYSTEM_PROMPT, user)
        rows.append(result)
        if cfg.sleep_between_requests_s:
            time.sleep(cfg.sleep_between_requests_s)
    out = pd.DataFrame(rows)
    out["pred"] = (out["label"].str.lower() == "fake").astype(int)
    return out


# ---------------------------------------------------------------------------
# sklearn-style adapters used by the unified evaluator
# ---------------------------------------------------------------------------
class LLMZeroShot:
    def __init__(self, cfg: C.LLMConfig | None = None, lang: str = "en"):
        self.cfg = cfg or C.LLMConfig()
        self.lang = lang
        self._last_results: pd.DataFrame | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._last_results = classify_zero_shot(df, self.lang, self.cfg)
        return self._last_results["pred"].values


class LLMFewShot:
    def __init__(self, train_df: pd.DataFrame, cfg: C.LLMConfig | None = None, lang: str = "en"):
        self.train_df = train_df
        self.cfg = cfg or C.LLMConfig()
        self.lang = lang
        self._last_results: pd.DataFrame | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._last_results = classify_few_shot(df, self.lang, self.train_df, self.cfg)
        return self._last_results["pred"].values


def explain_decision(text: str, predicted_label: str, cfg: C.LLMConfig | None = None) -> str:
    """Ask the LLM to explain WHY it labelled an article fake/real.

    Used in the explainability chapter to compare LLM rationales against SHAP
    attributions on the trained transformer.
    """
    cfg = cfg or C.LLMConfig()
    client = _client(cfg)
    prompt = (
        f"You labelled the following article as {predicted_label.upper()}. "
        "Explain in 3–5 sentences which concrete features of the text drove your decision. "
        "Cite specific phrases.\n\n"
        f"Article:\n{_truncate(text)}"
    )
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=0.3,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You are a careful media analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def config_to_dict(cfg: C.LLMConfig) -> Dict:
    return asdict(cfg)
