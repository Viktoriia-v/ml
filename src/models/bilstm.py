"""BiLSTM classifier with pretrained FastText embeddings.

Vocabulary is built from training data, capped at ``cfg.vocab_size``. Words not
in the FastText model get a small random vector. Shorter texts are right-padded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .. import config as C
from ..preprocessing import get_tokenizer

PAD, UNK = "<pad>", "<unk>"


# ---------------------------------------------------------------------------
@dataclass
class Vocab:
    stoi: Dict[str, int]

    @property
    def itos(self) -> List[str]:
        return [w for w, _ in sorted(self.stoi.items(), key=lambda kv: kv[1])]

    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.stoi[PAD]] * (max_len - len(ids))
        return ids


def build_vocab(texts: List[str], tokenize, max_size: int) -> Vocab:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most = [w for w, _ in counter.most_common(max_size - 2)]
    stoi = {PAD: 0, UNK: 1}
    for w in most:
        stoi[w] = len(stoi)
    return Vocab(stoi)


# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, tokenize, max_len: int):
        self.tokens = [tokenize(t) for t in df["text"].tolist()]
        self.labels = df["label"].values.astype(np.int64)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        ids = self.vocab.encode(self.tokens[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx])


# ---------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float, bidirectional: bool,
                 pad_idx: int, num_classes: int = 2,
                 pretrained_emb: torch.Tensor | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)            # (B, T, E)
        out, _ = self.lstm(emb)            # (B, T, 2H)
        # masked mean pooling — exclude pad positions
        mask = (x != 0).unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return self.fc(self.dropout(pooled))


# ---------------------------------------------------------------------------
def _load_fasttext_vectors(lang: str, vocab: Vocab, dim: int) -> torch.Tensor:
    """Load FastText vectors for words in ``vocab``.

    Strategy:
        1. Try ``gensim`` to load ``cc.<lang>.300.bin`` if ``FASTTEXT_PATH_<LANG>``
           env var points at it.
        2. Otherwise fall back to random init — the model still trains, just
           without the embedding head start.
    """
    import os

    weights = np.random.normal(0, 0.1, size=(len(vocab.stoi), dim)).astype(np.float32)
    weights[vocab.stoi[PAD]] = 0.0

    env_var = f"FASTTEXT_PATH_{lang.upper()}"
    path = os.environ.get(env_var)
    if not path:
        print(f"[BiLSTM] {env_var} not set — using random init for embeddings.")
        return torch.tensor(weights)

    try:
        from gensim.models.fasttext import load_facebook_vectors

        ft = load_facebook_vectors(path)
    except Exception as exc:
        print(f"[BiLSTM] failed to load FastText from {path}: {exc} — using random init.")
        return torch.tensor(weights)

    hits = 0
    for word, idx in vocab.stoi.items():
        if word in (PAD, UNK):
            continue
        try:
            weights[idx] = ft.get_vector(word)
            hits += 1
        except KeyError:
            pass
    print(f"[BiLSTM] FastText hit rate: {hits}/{len(vocab.stoi)} = {hits/len(vocab.stoi):.1%}")
    return torch.tensor(weights)


# ---------------------------------------------------------------------------
def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lang: str,
    cfg: C.BiLSTMConfig | None = None,
) -> Dict:
    cfg = cfg or C.BiLSTMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenize = get_tokenizer(lang)

    vocab = build_vocab(train_df["text"].tolist(), tokenize, cfg.vocab_size)
    pretrained = _load_fasttext_vectors(lang, vocab, cfg.embedding_dim)

    train_ds = TextDataset(train_df, vocab, tokenize, cfg.max_len)
    val_ds = TextDataset(val_df, vocab, tokenize, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = BiLSTMClassifier(
        vocab_size=len(vocab.stoi),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        pad_idx=vocab.stoi[PAD],
        pretrained_emb=pretrained,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += float(loss) * x.size(0)
        val_acc = _eval(model, val_loader, device)
        print(f"[BiLSTM] epoch {epoch+1}/{cfg.epochs} "
              f"loss={total_loss/len(train_ds):.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_state = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "vocab": vocab, "config": cfg, "device": device}


@torch.no_grad()
def _eval(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.size(0)
    return correct / max(total, 1)


def _logits_for(bundle: Dict, df: pd.DataFrame, lang: str) -> np.ndarray:
    cfg: C.BiLSTMConfig = bundle["config"]
    vocab: Vocab = bundle["vocab"]
    tokenize = get_tokenizer(lang)
    ds = TextDataset(df, vocab, tokenize, cfg.max_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size)
    model = bundle["model"]
    device = bundle["device"]
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict(bundle: Dict, df: pd.DataFrame, lang: str) -> np.ndarray:
    return _logits_for(bundle, df, lang).argmax(axis=-1)


def predict_proba(bundle: Dict, df: pd.DataFrame, lang: str) -> np.ndarray:
    logits = _logits_for(bundle, df, lang)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def save(bundle: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": bundle["model"].state_dict(),
            "vocab_stoi": bundle["vocab"].stoi,
            "config": bundle["config"].__dict__,
        },
        path,
    )


def load(path: str | Path, lang: str) -> Dict:
    # weights_only=False because we also persist a vocab dict + config dict.
    # Safe here because we only load checkpoints we wrote ourselves.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = C.BiLSTMConfig(**payload["config"])
    vocab = Vocab(payload["vocab_stoi"])
    model = BiLSTMClassifier(
        vocab_size=len(vocab.stoi),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        pad_idx=vocab.stoi[PAD],
    )
    model.load_state_dict(payload["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return {"model": model, "vocab": vocab, "config": cfg, "device": device}
