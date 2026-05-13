"""Fine-tuning pipeline for mBERT / XLM-R via HuggingFace Transformers Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .. import config as C


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True), preserve_index=False
    )


def _build_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lang: str,
    cfg: C.TransformerConfig | None = None,
    output_dir: str | Path | None = None,
) -> Dict:
    cfg = cfg or C.TransformerConfig()
    set_seed(C.RANDOM_STATE)

    out_dir = Path(output_dir or (C.RESULTS_DIR / "checkpoints" / f"{cfg.model_name.replace('/', '_')}_{lang}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    train_ds = _to_hf_dataset(train_df).map(tok, batched=True, remove_columns=["text"])
    val_ds = _to_hf_dataset(val_df).map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=C.ID2LABEL,
        label2id=C.LABEL2ID,
    )

    # Newer `transformers` (>=4.46) renamed `evaluation_strategy` -> `eval_strategy`
    # and removed `tokenizer=` from Trainer in favour of `processing_class=`.
    # We try the modern names first and fall back to the legacy names.
    args_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        report_to=["none"],
        logging_steps=50,
        seed=C.RANDOM_STATE,
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **args_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_build_metrics(),
    )
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    return {"trainer": trainer, "tokenizer": tokenizer, "model": trainer.model, "config": cfg}


# ---------------------------------------------------------------------------
def predict(bundle: Dict, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    return predict_proba(bundle, df).argmax(axis=-1)


def predict_proba(bundle: Dict, df: pd.DataFrame, lang: str | None = None) -> np.ndarray:
    tokenizer = bundle["tokenizer"]
    cfg: C.TransformerConfig = bundle["config"]
    ds = _to_hf_dataset(df).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length),
        batched=True,
        remove_columns=["text"],
    )
    trainer: Trainer = bundle["trainer"]
    raw = trainer.predict(ds)
    logits = raw.predictions
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def save(bundle: Dict, path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    bundle["model"].save_pretrained(p)
    bundle["tokenizer"].save_pretrained(p)


def load(path: str | Path) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    cfg = C.TransformerConfig(model_name=str(path))
    try:
        trainer = Trainer(model=model, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(model=model, tokenizer=tokenizer)
    return {"trainer": trainer, "tokenizer": tokenizer, "model": model, "config": cfg}
