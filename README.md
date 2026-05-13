# Multilingual Fake News Classification

> Practical part of the Master's thesis. Comparison of fake/real news classifiers
> across **English, Russian, Ukrainian, and Chinese** — from TF-IDF baselines
> through transformers to LLM zero/few-shot prompting.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research question

How does the quality of fake/real news classification change as we move from
simple models (TF-IDF + Logistic Regression) to modern transformers and LLMs,
and what patterns emerge across different languages?

## Models

| # | Family            | Model                                   |
|---|-------------------|-----------------------------------------|
| 1 | Baseline          | TF-IDF + Logistic Regression            |
| 2 | Classical ML      | XGBoost on TF-IDF features              |
| 3 | Neural net        | BiLSTM + FastText multilingual embeds   |
| 4 | Transformer       | mBERT (`bert-base-multilingual-cased`)  |
| 5 | Transformer       | XLM-RoBERTa-base                        |
| 6 | Transformer       | XLM-RoBERTa-large *(if GPU permits)*    |
| 7 | LLM (API)         | Zero-shot **and** few-shot via OpenRouter |

7 models × 4 languages = **28 experiments**.

## Repository layout

```
thesis-fake-news/
├── README.md                 # this file
├── requirements.txt
├── .env.example              # copy to .env and fill in your keys
├── data/
│   ├── raw/                  # raw datasets (gitignored)
│   ├── processed/            # cleaned, split datasets (gitignored)
│   └── README.md             # dataset sources, licenses, download instructions
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_lr.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_bilstm.ipynb
│   ├── 05_transformers.ipynb
│   ├── 06_llm.ipynb
│   └── 07_explainability.ipynb
├── src/
│   ├── config.py             # paths, hyperparameters, language config
│   ├── data_utils.py         # dataset loading + train/val/test split
│   ├── preprocessing.py      # text cleaning, tokenization
│   ├── train.py              # unified training entry points
│   ├── evaluate.py           # metrics + confusion matrices
│   ├── explain.py            # SHAP / LIME / attention helpers
│   └── models/
│       ├── baseline.py       # TF-IDF + LR
│       ├── xgboost_model.py
│       ├── bilstm.py
│       ├── transformer.py
│       └── llm.py            # OpenRouter zero/few-shot
├── results/
│   ├── tables/
│   ├── confusion_matrices/
│   └── explainability/
└── thesis_excerpts/          # tables and figures for the thesis text
```

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Data

See [`data/README.md`](data/README.md) for per-language sources. Many datasets
load directly via `datasets.load_dataset(...)` and require no manual download;
others (Kaggle, GitHub releases) need a one-time download.

### 4. Run an experiment

```bash
# Baseline on all 4 languages, results into results/tables/baseline_lr.csv
python -m src.train --model baseline --lang all

# XGBoost only on Ukrainian
python -m src.train --model xgboost --lang uk

# Transformer fine-tuning (GPU recommended)
python -m src.train --model xlmr-base --lang en --epochs 3

# LLM zero-shot via OpenRouter
python -m src.train --model llm-zero --lang ru --llm-model anthropic/claude-3.5-sonnet
```

Or open the notebooks in `notebooks/` — they call into `src/` and write
results to the same locations.

## Reproducibility

- Fixed `random_state = 42` everywhere
- Fixed train/val/test split per language, persisted to `data/processed/<lang>/`
- All hyperparameters are in `src/config.py`
- Results are written to `results/tables/<model>_<lang>.csv`

## Limitations

This work uses **different datasets for each language** (Variant A in the spec).
Direct comparison of absolute metrics across languages is therefore **not
methodologically sound** — domains, label noise, and difficulty differ.
The discussion compares **relative gains** (baseline → transformer) instead.
This is documented explicitly in the Limitations section of the thesis.

## License

Code: MIT (see [`LICENSE`](LICENSE)).
Datasets retain their original licenses — see [`data/README.md`](data/README.md).
