"""Explainability helpers — one function per model family.

The goal is *not* to plug everything together automatically, but to give the
notebooks a consistent set of building blocks:

    * ``explain_lr_global``     — top-N coefficients per class
    * ``explain_lime``          — single-example LIME explanation (LR or XGB)
    * ``explain_xgb_shap``      — SHAP TreeExplainer summary for XGBoost
    * ``explain_transformer_attention``  — BertViz head view
    * ``explain_transformer_ig`` — Captum integrated gradients
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# 1. Logistic Regression — global view (top features per class)
# ---------------------------------------------------------------------------
def explain_lr_global(pipe, top_k: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-export from baseline module for convenience."""
    from .models.baseline import top_features
    return top_features(pipe, top_k=top_k)


# ---------------------------------------------------------------------------
# 2. LIME — works for any sklearn-like estimator with predict_proba
# ---------------------------------------------------------------------------
def explain_lime(
    pipe,
    text: str,
    *,
    num_features: int = 12,
    num_samples: int = 1000,
    class_names: Iterable[str] = ("real", "fake"),
):
    """Returns a ``lime.explanation.Explanation`` object.

    Use ``.show_in_notebook()`` or ``.as_html()`` to render it.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=list(class_names), random_state=C.RANDOM_STATE)
    return explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda xs: pipe.predict_proba(list(xs)),
        num_features=num_features,
        num_samples=num_samples,
    )


# ---------------------------------------------------------------------------
# 3. XGBoost — SHAP TreeExplainer
# ---------------------------------------------------------------------------
def explain_xgb_shap(pipe, df: pd.DataFrame, max_display: int = 20):
    """Compute SHAP values on the TF-IDF feature space and return them.

    Plot with ``shap.summary_plot(shap_values, feature_names=...)``.
    """
    import shap

    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X = vec.transform(df["text"].tolist())
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    return {
        "shap_values": shap_values,
        "X": X,
        "feature_names": vec.get_feature_names_out(),
        "expected_value": explainer.expected_value,
        "max_display": max_display,
    }


# ---------------------------------------------------------------------------
# 4. Transformer — attention visualization (BertViz head view)
# ---------------------------------------------------------------------------
def explain_transformer_attention(model, tokenizer, text: str):
    """Returns the (tokens, attention) tuple. Pass to ``bertviz.head_view``.

    Example
    -------
    >>> from bertviz import head_view
    >>> tokens, attn = explain_transformer_attention(model, tokenizer, "...")
    >>> head_view(attn, tokens)
    """
    import torch

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, outputs.attentions


# ---------------------------------------------------------------------------
# 5. Transformer — Captum integrated gradients
# ---------------------------------------------------------------------------
def explain_transformer_ig(
    model,
    tokenizer,
    text: str,
    target_class: int = 1,
    n_steps: int = 50,
):
    """Per-token attribution scores via integrated gradients.

    Returns a list of (token, score) tuples sorted by absolute score.
    """
    import torch
    from captum.attr import IntegratedGradients

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids)
    baseline = torch.zeros_like(input_embeds)

    def forward(embeds):
        return model(inputs_embeds=embeds, attention_mask=attention_mask).logits

    ig = IntegratedGradients(forward)
    attributions, _ = ig.attribute(
        input_embeds, baselines=baseline, target=target_class,
        n_steps=n_steps, return_convergence_delta=True,
    )
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return list(zip(tokens, scores.tolist()))


# ---------------------------------------------------------------------------
# 6. LLM — explain its own decision
# ---------------------------------------------------------------------------
def explain_llm(text: str, predicted_label: str) -> str:
    from .models.llm import explain_decision
    return explain_decision(text, predicted_label)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def attribution_table(token_scores: List[Tuple[str, float]], top_k: int = 20) -> pd.DataFrame:
    df = pd.DataFrame(token_scores, columns=["token", "score"])
    df["abs_score"] = df["score"].abs()
    return df.nlargest(top_k, "abs_score").reset_index(drop=True)


def save_explanation_html(html: str, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
