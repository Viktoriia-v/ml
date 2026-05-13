"""Unified training entry point.

Examples
--------
    # baseline on all 4 languages
    python -m src.train --model baseline --lang all

    # XGBoost only on Ukrainian
    python -m src.train --model xgboost --lang uk

    # Transformer fine-tuning (GPU recommended)
    python -m src.train --model xlmr-base --lang en --epochs 3

    # LLM zero-shot via OpenRouter
    python -m src.train --model llm-zero --lang ru --llm-model anthropic/claude-3.5-sonnet --llm-test-cap 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from . import config as C
from .data_utils import get_split
from .evaluate import evaluate_and_log
from .models import baseline, bilstm, llm, transformer, xgboost_model

MODEL_REGISTRY = {
    "baseline": baseline,
    "xgboost": xgboost_model,
    "bilstm": bilstm,
    "mbert": transformer,
    "xlmr-base": transformer,
    "xlmr-large": transformer,
    "llm-zero": llm,
    "llm-few": llm,
}

TRANSFORMER_MAP = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
}


# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate fake-news classifiers.")
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--lang", required=True, help="One of en/ru/uk/zh, or 'all'.")
    p.add_argument("--epochs", type=int, default=None, help="Override transformer/BiLSTM epochs.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--llm-model", default=None, help="OpenRouter model id, e.g. anthropic/claude-3.5-sonnet")
    p.add_argument("--llm-test-cap", type=int, default=None,
                   help="Only run on the first N test examples (cost control).")
    p.add_argument("--force-rebuild-splits", action="store_true")
    return p.parse_args(argv)


def _select_languages(arg: str) -> List[str]:
    if arg == "all":
        return list(C.LANGUAGES)
    if arg not in C.LANGUAGES:
        raise SystemExit(f"--lang must be one of {C.LANGUAGES + ['all']}, got '{arg}'")
    return [arg]


# ---------------------------------------------------------------------------
def run_one(model_key: str, lang: str, args: argparse.Namespace) -> dict:
    print(f"\n=== {model_key} :: {lang.upper()} ===")
    train_df, val_df, test_df = get_split(lang, force_rebuild=args.force_rebuild_splits)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    if model_key in {"baseline", "xgboost"}:
        module = MODEL_REGISTRY[model_key]
        model = module.train(train_df, val_df, lang)
        y_pred = module.predict(model, test_df, lang)

    elif model_key == "bilstm":
        cfg = C.BiLSTMConfig()
        if args.epochs:
            cfg.epochs = args.epochs
        if args.batch_size:
            cfg.batch_size = args.batch_size
        bundle = bilstm.train(train_df, val_df, lang, cfg=cfg)
        y_pred = bilstm.predict(bundle, test_df, lang)

    elif model_key in {"mbert", "xlmr-base", "xlmr-large"}:
        cfg = C.TransformerConfig(model_name=TRANSFORMER_MAP[model_key])
        if args.epochs:
            cfg.epochs = args.epochs
        if args.batch_size:
            cfg.batch_size = args.batch_size
        bundle = transformer.train(train_df, val_df, lang, cfg=cfg)
        y_pred = transformer.predict(bundle, test_df, lang)

    elif model_key in {"llm-zero", "llm-few"}:
        cfg = C.LLMConfig()
        if args.llm_model:
            cfg.model = args.llm_model
        sub = test_df.head(args.llm_test_cap) if args.llm_test_cap else test_df
        if model_key == "llm-zero":
            res = llm.classify_zero_shot(sub, lang, cfg=cfg)
        else:
            res = llm.classify_few_shot(sub, lang, train_df=train_df, cfg=cfg)
        y_pred = res["pred"].values
        test_df = sub  # for correct y_true alignment

    else:
        raise ValueError(model_key)

    metrics = evaluate_and_log(
        test_df["label"].values, y_pred,
        model_name=model_key, lang=lang,
    )
    print("  metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    langs = _select_languages(args.lang)
    for lang in langs:
        run_one(args.model, lang, args)


if __name__ == "__main__":
    main()
